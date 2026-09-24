"""Aggregate per-round k6 summaries into bench/results/overhead.json (stdout).

Overhead is computed per round as gateway minus direct (both measured in the
same round), then summarised as the median and min-max across rounds."""
import datetime, json, os, platform, statistics, subprocess, sys

rounds, rate, duration, latency = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
names = ["direct-json", "gateway-json", "gateway-json-tenant", "direct-stream", "gateway-stream"]
runs = {n: [json.load(open(f"bench/results/overhead-{n}-{r}.tmp.json")) for r in range(1, rounds + 1)] for n in names}

def cpu():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()

def compare(gw, direct, what):
    out = {"what": what, "gateway_ms": {}, "direct_ms": {}, "overhead_ms": {}}
    for k in ("p50", "p95", "p99"):
        diffs = [g["latency_ms"][k] - d["latency_ms"][k] for g, d in zip(runs[gw], runs[direct])]
        out["direct_ms"][k] = round(statistics.median(r["latency_ms"][k] for r in runs[direct]), 3)
        out["gateway_ms"][k] = round(statistics.median(r["latency_ms"][k] for r in runs[gw]), 3)
        out["overhead_ms"][k] = {"median": round(statistics.median(diffs), 3), "min": round(min(diffs), 3), "max": round(max(diffs), 3)}
    out["achieved_rps"] = round(statistics.median(r["achieved_rps"] for r in runs[gw]), 1)
    out["failed_rate"] = max(r["failed_rate"] for r in runs[gw] + runs[direct])
    out["dropped_iterations"] = sum(r["dropped_iterations"] for r in runs[gw] + runs[direct])
    return out

out = {
    "what": "added latency of ModelMux vs calling the upstream directly",
    "method": f"k6 constant-arrival-rate {rate} req/s for {duration} per run; {rounds} rounds, each running every variant; "
              f"overhead = gateway minus direct within the same round, reported as median and min-max across rounds. "
              f"Stub upstream with fixed {latency} latency over HTTP, caches off, k6 + gateway + stub on one machine.",
    "machine": {"cpu": cpu(), "cores": os.cpu_count(), "os": platform.platform()},
    "date": datetime.date.today().isoformat(),
    "variants": {
        "json_public": compare("gateway-json", "direct-json", "non-streaming, anonymous caller (no limits configured)"),
        "json_tenant": compare("gateway-json-tenant", "direct-json", "non-streaming, API-key tenant with rate limit + budget active"),
        "stream_ttfb": compare("gateway-stream", "direct-stream", "streaming, time to first byte"),
    },
    "rounds": runs,
}
try:
    samples = [l.split() for l in open("bench/results/overhead-proc.tmp") if l.strip()]
    out["gateway_process"] = {"peak_rss_mb": round(max(int(x[0]) for x in samples) / 1024, 1),
                              "median_cpu_percent": statistics.median(float(x[1]) for x in samples),
                              "note": "sampled once per second with ps during gateway-json runs; 100% = one core"}
except FileNotFoundError:
    pass
print(json.dumps(out, indent=2))
