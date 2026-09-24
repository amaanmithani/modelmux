"""Deterministically sample a balanced set of Quora Question Pairs (GLUE QQP
validation split) to bench/data/qqp_sample.jsonl. The data itself is not
committed; this script and the seed make the sample reproducible."""
# /// script
# dependencies = ["datasets>=2.19"]
# ///
import json, random, sys
from pathlib import Path
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
rng = random.Random(20260924)
dup = [r for r in ds if r["label"] == 1]
non = [r for r in ds if r["label"] == 0]
rows = rng.sample(dup, N // 2) + rng.sample(non, N // 2)
rng.shuffle(rows)
out = Path(__file__).resolve().parents[1] / "data" / "qqp_sample.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps({"idx": r["idx"], "q1": r["question1"], "q2": r["question2"], "duplicate": r["label"] == 1}) + "\n")
print(f"wrote {len(rows)} pairs to {out}")
