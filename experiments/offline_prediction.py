"""Prediction-only experiments (no robot): how well can each predictor say where the
target is NOW, from measurements that arrive `latency` seconds late?

  python3 experiments/offline_prediction.py tune    -> grid search on TUNING subjects (seeds 100-111)
  python3 experiments/offline_prediction.py sweep   -> latency sweep on TEST subjects (seeds 0-39)

Tuning and test subjects are different people (different seeds) on purpose:
settings are chosen on one group and reported on another.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import Breathing, random_subject          # noqa: E402
from breathus.estimators import CVKalman, HarmonicEKF, MeasurementStream  # noqa: E402
from breathus.sim import CV_Q, EKF_Q_AMP, EKF_Q_OMEGA              # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def prediction_rmse_mm(sub, make_estimator, latency, T=80.0, warmup=20.0):
    """RMS error [mm] between the predicted and the true target position, after warm-up.
    make_estimator=None means 'no prediction': use the newest raw measurement as it is."""
    b = Breathing(sub, T + 5)
    truth = lambda t: b.target(t)[0]
    stream = MeasurementStream(truth, latency=latency, seed=sub.seed)
    est = make_estimator() if make_estimator else None
    last, err = None, []
    for t in np.arange(0.0, T, 0.01):
        for z, t_acq in stream.poll(t):
            last = z
            if est:
                est.update(z, t_acq)
        if t < warmup or last is None:
            continue
        pred = est.predict(t)[0] if est else last
        err.append(pred - truth(t))
    return 1e3 * float(np.sqrt(np.mean(np.square(err))))


def tune():
    subs = [random_subject(s) for s in range(100, 112)]
    for L in (0.1, 0.3):
        print(f"latency {L} s")
        for q in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            e = [prediction_rmse_mm(s, lambda: CVKalman(q=q), L) for s in subs]
            print(f"  CV  q={q:g}: mean {np.mean(e):.3f} mm, worst {np.max(e):.2f}")
        for qa in (1e-8, 1e-7, 1e-6):
            for qw in (1e-4, 1e-3, 1e-2, 1e-1):
                e = [prediction_rmse_mm(s, lambda: HarmonicEKF(q_amp=qa, q_omega=qw), L) for s in subs]
                print(f"  EKF q_amp={qa:g} q_omega={qw:g}: mean {np.mean(e):.3f} mm, worst {np.max(e):.2f}")


def sweep(n_subjects=40):
    subs = [random_subject(s) for s in range(n_subjects)]
    methods = {
        "none": None,
        "cv": lambda: CVKalman(q=CV_Q),
        "ekf": lambda: HarmonicEKF(q_amp=EKF_Q_AMP, q_omega=EKF_Q_OMEGA),
    }
    latencies = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    res = {name: [[prediction_rmse_mm(s, mk, L) for s in subs] for L in latencies]
           for name, mk in methods.items()}
    res["latencies"] = latencies
    res["target_pp_mm"] = [1e3 * s.target_pp for s in subs]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "offline_sweep.json"), "w"))
    for name in methods:
        print(name, [round(float(np.mean(r)), 3) for r in res[name]])


if __name__ == "__main__":
    {"tune": tune, "sweep": sweep}[sys.argv[1] if len(sys.argv) > 1 else "sweep"]()
