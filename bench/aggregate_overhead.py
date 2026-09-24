"""Aggregate per-round k6 summaries into bench/results/overhead.json (stdout)."""
import json, platform, statistics, subprocess, sys, datetime

rounds, rate, duration, latency = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
runs = {t: [json.load(open(f"bench/results/overhead-{t}-{r}.tmp.json")) for r in range(1, rounds + 1)]
        for t in ("direct", "gateway")}

def med(t, key):
    return statistics.median(run["latency_ms"][key] for run in runs[t])

def cpu():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()

out = {
    "what": "added latency of ModelMux vs calling the upstream directly",
    "method": f"k6 constant-arrival-rate {rate} req/s for {duration}, {rounds} alternating rounds per path, "
              f"median of rounds; stub upstream with fixed {latency} latency over HTTP; caches off; "
              "k6, gateway and stub on the same machine",
    "machine": {"cpu": cpu(), "cores": __import__("os").cpu_count(), "os": platform.platform()},
    "date": datetime.date.today().isoformat(),
    "direct_ms": {k: round(med("direct", k), 3) for k in ("p50", "p95", "p99")},
    "gateway_ms": {k: round(med("gateway", k), 3) for k in ("p50", "p95", "p99")},
    "achieved_rps": {t: round(statistics.median(r["achieved_rps"] for r in runs[t]), 1) for t in runs},
    "failed_rate": {t: max(r["failed_rate"] for r in runs[t]) for t in runs},
    "dropped_iterations": {t: sum(r["dropped_iterations"] for r in runs[t]) for t in runs},
    "rounds": runs,
}
try:
    samples = [l.split() for l in open("bench/results/overhead-proc.tmp") if l.strip()]
    rss = [int(x[0]) for x in samples]
    cpu_pct = [float(x[1]) for x in samples]
    out["gateway_process"] = {"peak_rss_mb": round(max(rss) / 1024, 1),
                              "median_cpu_percent": statistics.median(cpu_pct),
                              "note": "sampled once per second with ps during gateway rounds; 100% = one core"}
except FileNotFoundError:
    pass
out["overhead_ms"] = {k: round(out["gateway_ms"][k] - out["direct_ms"][k], 3) for k in ("p50", "p95", "p99")}
print(json.dumps(out, indent=2))
