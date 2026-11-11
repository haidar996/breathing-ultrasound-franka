"""Quick sanity checks (about 30 s).   Run:  python3 tests/test_basics.py   (or pytest)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import Breathing, SubjectParams, nominal_subject   # noqa: E402
from breathus.estimators import CVKalman, HarmonicEKF, MeasurementStream   # noqa: E402
from breathus.sim import load_model, run_episode, solve_ik                 # noqa: E402


def test_breathing_velocity_matches_position():
    """The analytic ds/dt must agree with a numerical derivative of s."""
    b = Breathing(nominal_subject())
    for t in np.linspace(1, 60, 200):
        h = 1e-5
        num = (b.s(t + h)[0] - b.s(t - h)[0]) / (2 * h)
        assert abs(num - b.s(t)[1]) < 1e-4


def test_force_sensor_reads_probe_weight_in_free_air():
    """Hanging in the air, the sensor must show only the probe's weight (0.3 kg -> 2.943 N)."""
    m = load_model()
    d = mujoco.MjData(m)
    solve_ik(m, d, np.array([0.5, 0.0, 0.45]))
    for _ in range(300):
        d.ctrl[:7] = d.qfrc_bias[:7] - 20 * d.qvel[:7]
        mujoco.mj_step(m, d)
    assert abs(d.sensor("ft_force").data[2] + 0.3 * 9.81) < 0.02


def test_ekf_finds_the_breathing_rate():
    """On a clean periodic signal the EKF must lock onto the true rate (18 breaths/min here,
    starting from its 15/min prior) and predict better than straight-line extrapolation."""
    sub = SubjectParams(rate_cpm=18, period_cv=0, amp_cv=0, drift_frac=0, seed=3)
    b = Breathing(sub)
    truth = lambda t: b.target(t)[0]
    stream = MeasurementStream(truth, latency=0.3, seed=3)
    ekf, cv = HarmonicEKF(), CVKalman(q=1e-4)
    e_ekf, e_cv = [], []
    for t in np.arange(0, 60, 0.01):
        for z, ta in stream.poll(t):
            ekf.update(z, ta)
            cv.update(z, ta)
        if t > 30:
            e_ekf.append(ekf.predict(t)[0] - truth(t))
            e_cv.append(cv.predict(t)[0] - truth(t))
    assert abs(ekf.rate_cpm - 18) < 0.5
    assert np.std(e_ekf) < np.std(e_cv)


def test_closed_loop_holds_force_and_follows_target():
    r = run_episode(nominal_subject(), "force", "ekf", duration=30.0)
    assert abs(r["force_mean"] - 5.0) < 0.1
    assert r["force_rmse"] < 0.5
    assert r["pct_lost"] == 0.0
    assert r["track_rmse_mm"] < 1.5


def test_needle_reaches_target_with_perfect_knowledge():
    """With the future known, the miss must be only the robot's own execution error (< 1 mm)."""
    from breathus.needle import run_insertion
    r = run_insertion(nominal_subject(), "oracle")
    assert r["miss_mm"] < 1.0
    assert r["rcm_max_mm"] < 2.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
