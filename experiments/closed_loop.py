"""Closed-loop experiments with the simulated Franka.

  python3 experiments/closed_loop.py [n_subjects] [n_workers]

For each virtual patient (seed 0 .. n-1, never used for tuning) it runs:
  tracking comparison at 100 ms latency : static, delayed, cv, ekf, oracle   (force control on z)
  tracking comparison at 300 ms latency : cv, ekf
  force comparison                      : impedance (soft, 300 N/m) and impedance (stiff, 1500 N/m)
Results are appended to outputs/closed_loop.json after every run, so an interrupted
job can simply be started again and continues where it stopped.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import random_subject      # noqa: E402
from breathus.sim import run_episode               # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "closed_loop.json")
DURATION = 60.0                                     # 20 s warm-up + 40 s scored

CONDITIONS = (
    [dict(name=f"{tm}_L100", force_mode="force", track_mode=tm, latency=0.1)
     for tm in ("static", "delayed", "cv", "ekf", "oracle")]
    + [dict(name=f"{tm}_L300", force_mode="force", track_mode=tm, latency=0.3) for tm in ("cv", "ekf")]
    + [dict(name="imp300", force_mode="impedance", track_mode="ekf", latency=0.1, imp_stiffness=300.0),
       dict(name="imp1500", force_mode="impedance", track_mode="ekf", latency=0.1, imp_stiffness=1500.0)]
)


def job(args):
    seed, cond = args
    kw = {k: v for k, v in cond.items() if k != "name"}
    res = run_episode(random_subject(seed), duration=DURATION, **kw)
    return f"{cond['name']}|{seed}", res


if __name__ == "__main__":
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(s, c) for s in range(n_sub) for c in CONDITIONS if f"{c['name']}|{s}" not in done]
    print(f"{len(todo)} runs to do, {len(done)} already done", flush=True)
    with Pool(n_workers) as pool:
        for key, res in pool.imap_unordered(job, todo):
            done[key] = res
            json.dump(done, open(OUT, "w"), indent=1)
            print(key, {k: round(v, 3) for k, v in res.items()}, flush=True)
