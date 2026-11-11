"""Needle insertion study.

  python3 experiments/needle_study.py [n_subjects] [n_trials] [n_workers]

For every virtual patient (seeds 0..n-1, never used for tuning), every trial (a random moment in
the breathing cycle), every strategy and three mean insertion speeds (10, 20, 40 mm/s) it runs one
insertion and stores the miss distance. Resumable: results accumulate in outputs/needle_study.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import random_subject      # noqa: E402
from breathus.needle import run_insertion          # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "needle_study.json")
METHODS = ("naive", "mean", "cv", "ekf", "oracle")
SPEEDS = (0.02, 0.01, 0.04)


def job(args):
    method, speed, seed, trial = args
    res = run_insertion(random_subject(seed), method, speed=speed, trial=trial)
    return f"{method}|{speed}|{seed}|{trial}", res


if __name__ == "__main__":
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(m, v, s, k) for v in SPEEDS for s in range(n_sub) for k in range(n_trials) for m in METHODS
            if f"{m}|{v}|{s}|{k}" not in done]
    print(f"{len(todo)} insertions to do, {len(done)} already done", flush=True)
    with Pool(n_workers, maxtasksperchild=40) as pool:
        for i, (key, res) in enumerate(pool.imap_unordered(job, todo)):
            done[key] = res
            if i % 10 == 0 or i == len(todo) - 1:
                json.dump(done, open(OUT, "w"))
            print(key, round(res["miss_mm"], 2), flush=True)
    json.dump(done, open(OUT, "w"))
    print("FINISHED", flush=True)
