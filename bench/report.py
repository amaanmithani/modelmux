"""Render bench/results/*.json into README.md (between the RESULTS markers) and
internal/server/static/facts.json. Every number in either comes from here."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "bench" / "results"
ov = json.loads((R / "overhead.json").read_text())
fa = json.loads((R / "faults.json").read_text())
se = json.loads((R / "semantic.json").read_text())
lookup = re.findall(r"BenchmarkSemanticLookup-\d+\s+\d+\s+(\d+) ns/op", (R / "semantic-lookup.txt").read_text())
lookup_ms = sorted(int(x) for x in lookup)[len(lookup) // 2] / 1e6


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def rng(o):
    return f"{o['median']:.2f} ms ({o['min']:.2f}–{o['max']:.2f})"


V = ov["variants"]
lines = [
    "### Gateway overhead",
    "",
    f"{ov['method']} Machine: {ov['machine']['cpu']}, {ov['machine']['cores']} cores. "
    "Overhead cells show the median across rounds, with the min–max in brackets.",
    "",
    "| path | direct p50 / p99 | through ModelMux p50 / p99 | added at p50 | added at p99 |",
    "|---|---|---|---|---|",
]
for key in ("json_public", "json_tenant", "stream_ttfb"):
    v = V[key]
    lines.append(
        f"| {v['what']} | {v['direct_ms']['p50']:.2f} / {v['direct_ms']['p99']:.2f} ms | "
        f"{v['gateway_ms']['p50']:.2f} / {v['gateway_ms']['p99']:.2f} ms | {rng(v['overhead_ms']['p50'])} | {rng(v['overhead_ms']['p99'])} |")
gp = ov.get("gateway_process", {})
lines += [
    "",
    f"Every run held {V['json_public']['achieved_rps']:.0f} req/s with {V['json_public']['failed_rate']:.0%} errors and "
    f"{sum(v['dropped_iterations'] for v in V.values())} dropped iterations. Gateway process: peak RSS "
    f"{gp.get('peak_rss_mb', '?')} MB, median {gp.get('median_cpu_percent', 0):.0f}% of one core. The upstream is a local "
    "stub with fixed latency, so these numbers isolate what the gateway adds; against a real provider the added "
    "milliseconds are the same, but they're a much smaller fraction of the total.",
    "",
    "### Fallback under injected faults",
    "",
    f"{fa['method']}.",
    "",
    "| scenario | success rate | recovered / recoverable | HTTP requests that reached `a` | p99 |",
    "|---|---|---|---|---|",
]


def span(lo, hi, fmt="{}"):
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


for s in fa["scenarios"]:
    if s["name"] == "both-dead":
        rec = "— (nothing healthy to fall back to)"
    else:
        rec = ("all" if s["every_recoverable_request_recovered"] else "NOT all") + \
              f" ({span(s['recoverable_min'], s['recoverable_max'])} per run)"
    succ = span(s["success_rate_min"] * 100, s["success_rate_max"] * 100, "{:.1f}") + "%"
    lines.append(f"| **{s['name']}**: {s['description']} | {succ} | {rec} | "
                 f"{span(s['upstream_requests_to_a_min'], s['upstream_requests_to_a_max'])} | "
                 f"{span(s['latency_p99_ms_min'], s['latency_p99_ms_max'], '{:.1f}')} ms |")
mid = next(s for s in fa["scenarios"] if s["name"] == "midstream")
healthy = [s for s in fa["scenarios"] if s["name"] != "both-dead"]
all_ok = all(s["every_recoverable_request_recovered"] for s in healthy)
lines += [
    "",
    f"Streams that fail *after* their first byte can't be retried by any proxy, because the client already has "
    f"partial output. In the midstream scenario that was {span(mid['unrecoverable_midstream_min'], mid['unrecoverable_midstream_max'])} "
    "streams per run. ModelMux ends them with an in-band error event and no `[DONE]`, and never caches them.",
    "",
    "### Semantic cache: how much can it safely hit?",
    "",
    f"{se['method']}",
    "",
    "| embedding model | held-out AUC | no guard: τ → hits on paraphrases, false hits | with lexical guard | embed p50 |",
    "|---|---|---|---|---|",
]


def cell(m):
    if not isinstance(m, dict) or m.get("threshold_chosen_on_calibration") is None:
        return "no threshold meets the budget"
    h, f = m["heldout_pairwise_hit_rate_on_duplicates"], m["heldout_pairwise_false_hit_rate_on_non_duplicates"]
    return (f"τ={m['threshold_chosen_on_calibration']:.3f} → {pct(h['rate'])} [{pct(h['ci95_lo'])}–{pct(h['ci95_hi'])}], "
            f"{pct(f['rate'])} [{pct(f['ci95_lo'])}–{pct(f['ci95_hi'])}]")


for m in se["models"]:
    lines.append(f"| {m['model']} | {m['heldout_auc']:.3f} | {cell(m['unguarded'])} | {cell(m['guarded'])} | "
                 f"{m['embed_latency_ms_p50']:.0f} ms |")

guarded = [m for m in se["models"] if isinstance(m["guarded"], dict) and m["guarded"].get("threshold_chosen_on_calibration")]
best = max(guarded, key=lambda m: m["guarded"]["heldout_pairwise_hit_rate_on_duplicates"]["rate"])
bg = best["guarded"]
nomic = next((m for m in se["models"] if m["model"] == "nomic-embed-text"), None)
ex_model = (nomic or best)["model"]
ex = (nomic or best)["top_non_duplicates"][0]
lines += [
    "",
    f"In the cache simulation (every held-out question indexed, each paraphrase looked up by nearest neighbour), "
    f"{best['model']} with the guard answered {pct(bg['cache_sim_correct_answers_for_duplicate_queries']['rate'])} of "
    f"paraphrase queries correctly and gave a wrong cached answer to {pct(bg['cache_sim_wrong_answers_over_all_queries']['rate'])} "
    "of all queries.",
    "",
    f"Why the guard exists: embeddings score near-identical questions about different entities as the same question. "
    f"The top held-out non-duplicate for {ex_model} is *\"{ex['q1']}\"* vs *\"{ex['q2']}\"*, "
    f"cosine {ex['cosine']:.4f}" + (", which the guard blocks." if ex["guard_blocks_it"] else ". The guard does not catch this one.") +
    " A safe semantic cache catches only a minority of paraphrases, which is why it's off by default and off in the public demo.",
    "",
    f"Scan cost at capacity (2,000 entries × 1,024 dims, flat inner product): **{lookup_ms:.2f} ms** per lookup "
    "(`go test -bench SemanticLookup ./internal/cache`), small next to the embedding call itself.",
]

readme = ROOT / "README.md"
text = readme.read_text()
block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: block, text, flags=re.S)
readme.write_text(text)

jp = V["json_public"]["overhead_ms"]["p99"]
bo = next(s for s in fa["scenarios"] if s["name"] == "a-down-breaker-on")
facts = {
    "overhead_p99_ms": jp["median"], "overhead_p99_min_ms": jp["min"], "overhead_p99_max_ms": jp["max"],
    "overhead_p50_ms": V["json_public"]["overhead_ms"]["p50"]["median"], "rps": V["json_public"]["achieved_rps"],
    "stub_latency": ov["method"].split("fixed ")[1].split(" latency")[0],
    "fallback_scenarios": len(healthy), "fallback_all_recovered": all_ok,
    "midstream_unrecoverable_pct": round(100 * (1 - (mid["success_rate_min"] + mid["success_rate_max"]) / 2), 1),
    "breaker_hits_min": bo["upstream_requests_to_a_min"], "breaker_hits_max": bo["upstream_requests_to_a_max"],
    "breaker_requests": bo["runs"][0]["requests"],
    "semantic_model": best["model"],
    "semantic_hit": bg["heldout_pairwise_hit_rate_on_duplicates"]["rate"],
    "semantic_wrong": bg["cache_sim_wrong_answers_over_all_queries"]["rate"],
}
(ROOT / "internal/server/static/facts.json").write_text(json.dumps(facts, indent=2) + "\n")
print("README results + facts.json updated")
