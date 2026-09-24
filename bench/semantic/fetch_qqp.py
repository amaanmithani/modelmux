"""Deterministically sample balanced Quora Question Pairs (GLUE QQP validation
split) into a calibration set (used only to pick the threshold) and a held-out
set (used only to report). Writes bench/data/qqp_{calib,heldout}.jsonl, which
are gitignored: this script and its seed make the sample reproducible."""
# /// script
# dependencies = ["datasets>=2.19"]
# ///
import json, random
from pathlib import Path
from datasets import load_dataset

CALIB, HELDOUT = 2000, 1000
ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
rng = random.Random(20260924)
dup = [r for r in ds if r["label"] == 1]
non = [r for r in ds if r["label"] == 0]
n = (CALIB + HELDOUT) // 2
d, x = rng.sample(dup, n), rng.sample(non, n)
splits = {"calib": d[: CALIB // 2] + x[: CALIB // 2], "heldout": d[CALIB // 2 :] + x[CALIB // 2 :]}
out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(parents=True, exist_ok=True)
for name, rows in splits.items():
    rng.shuffle(rows)
    with (out / f"qqp_{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"idx": r["idx"], "q1": r["question1"], "q2": r["question2"], "duplicate": r["label"] == 1}) + "\n")
    print(f"{name}: {len(rows)} pairs")
