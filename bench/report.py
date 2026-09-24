"""Render bench/results/*.json into README.md between the RESULTS markers.
Every number in the README's results section comes from here."""
import json, re
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ov = json.loads((R / "overhead.json").read_text())
fa = json.loads((R / "faults.json").read_text())
se = json.loads((R / "semantic.json").read_text())
lookup = re.findall(r"BenchmarkSemanticLookup-\d+\s+\d+\s+(\d+) ns/op", (R / "semantic-lookup.txt").read_text())
lookup_ms = sorted(int(x) for x in lookup)[len(lookup) // 2] / 1e6

o = ov["overhead_ms"]
lines = [
    "### Gateway overhead",
    "",
    f"{ov['method']}. Machine: {ov['machine']['cpu']}, {ov['machine']['cores']} cores.",
    "",
    "| | p50 | p95 | p99 |",
    "|---|---|---|---|",
    f"| direct to upstream | {ov['direct_ms']['p50']:.2f} ms | {ov['direct_ms']['p95']:.2f} ms | {ov['direct_ms']['p99']:.2f} ms |",
    f"| through ModelMux | {ov['gateway_ms']['p50']:.2f} ms | {ov['gateway_ms']['p95']:.2f} ms | {ov['gateway_ms']['p99']:.2f} ms |",
    f"| **added by ModelMux** | **{o['p50']:.2f} ms** | **{o['p95']:.2f} ms** | **{o['p99']:.2f} ms** |",
    "",
    f"Achieved {ov['achieved_rps']['gateway']:.0f} req/s with {ov['failed_rate']['gateway']:.0%} errors and "
    f"{ov['dropped_iterations']['gateway']} dropped iterations. Gateway process: peak RSS "
    f"{ov['gateway_process']['peak_rss_mb']} MB, median {ov['gateway_process']['median_cpu_percent']:.0f}% of one core.",
    "",
    "### Fallback under injected faults",
    "",
    f"{fa['method']}.",
    "",
    "| scenario | success | recovered / recoverable | HTTP requests that reached `a` | p99 |",
    "|---|---|---|---|---|",
]
for s in fa["scenarios"]:
    rec = f"{s['recovered']}/{s['recoverable_failures']}" if s["recoverable_failures"] else "—"
    lines.append(f"| {s['name']}: {s['description']} | {s['success_rate']:.1%} | {rec} | {s['upstream_requests_to_a']} | {s['latency_p99_ms']:.1f} ms |")
mid = next(s for s in fa["scenarios"] if s["name"] == "midstream")
lines += [
    "",
    f"Streams that fail *after* the first byte can't be retried by any proxy, because the client already has "
    f"partial output. ModelMux ends those with an in-band error event and no `[DONE]` "
    f"({mid['in_band_stream_errors']} in the midstream run), and never caches them.",
    "",
    "### Semantic cache: how much can it safely hit?",
    "",
    f"{se['method']}.",
    "",
    "| embedding model | AUC | no guard: τ, hit rate @ ≤1% false hits | with lexical guard | embed p50 |",
    "|---|---|---|---|---|",
]
def cell(rec):
    return "no threshold qualifies" if rec is None else f"τ={rec['threshold']:.3f}, {rec['hit_rate_on_duplicates']:.1%} ({rec['false_hit_rate_on_non_duplicates']:.1%} false)"
for m in se["models"]:
    lines.append(f"| {m['model']} | {m['auc']:.3f} | {cell(m['unguarded']['recommended'])} | {cell(m['guarded']['recommended'])} | {m['embed_latency_ms_p50']:.0f} ms |")
lines += [
    "",
    "Embeddings score entity swaps as near-identical: with nomic-embed-text, *\"best budget hotels in Udaipur\"* vs "
    "*\"…in Munnar\"* has cosine ≈ 1.0. The lexical guard (`cache.GuardKey`) requires the same leading question word and the same "
    "capitalised/numeric tokens. Even so, a safe semantic cache catches only about a fifth of true paraphrases. "
    "That's why it's off by default, and off in the public demo.",
    "",
    f"Scan cost at capacity (2,000 entries × 1,024 dims, flat inner product): **{lookup_ms:.2f} ms** per lookup "
    "(`go test -bench SemanticLookup ./internal/cache`), small next to the embedding call itself.",
]
readme = Path(__file__).resolve().parent.parent / "README.md"
text = readme.read_text()
block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
text = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: block, text, flags=re.S) \
    if "<!-- RESULTS:START -->" in text else text.rstrip() + "\n\n## Results\n\n" + block + "\n"
readme.write_text(text)
print("README results updated")
