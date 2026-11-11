"""Needle insertion towards a target that moves with breathing.

The needle goes in along ONE straight line through ONE skin entry point (once it is inside
tissue it must not be swung sideways). So the robot can only choose two things:
    WHERE to aim   (the point the tip will reach), and
    WHEN to start  (so that the target is at that point when the tip arrives).
This is an interception problem, like catching a moving ball.

Strategies (argument `method`):
    'naive'   aim at where the target is seen now, start immediately
    'mean'    aim at the average target position of the last 12 s, start immediately
    'cv'      aim at  p + v * T  (straight-line prediction over the insertion time), limited to the
              range of positions seen in the last 12 s, start immediately
    'cv_raw'  the same without the limit (it extrapolates far outside the body)
    'ekf'     wait for the calm end-of-exhale phase predicted by the harmonic EKF, aim at the
              EKF prediction for the arrival time
    'oracle'  same waiting rule, but with perfect knowledge of the future (upper bound)

World axes: x across the patient, y feet -> head, z up.
"""
import os

import mujoco
import numpy as np

from .breathing import Breathing, SubjectParams
from .estimators import CVKalman, HarmonicEKF, MeasurementStream
from .sim import CV_Q, EKF_Q_AMP, EKF_Q_OMEGA, _rot_err

SCENE = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "franka_emika_panda",
                     "needle_scene.xml")

# ---- geometry and tissue model (assumptions are listed in docs/research_notes.md) ------------
ENTRY_XY = np.array([0.5, 0.0])     # skin entry point (world x, y) [m]
SKIN_Z0 = 0.32                      # skin height at exhale [m]
TARGET_DEPTH = 0.06                 # target depth below the exhale skin [m]
AP_RATIO = 0.5                      # target up/down motion = AP_RATIO * skin motion
STANDOFF = 0.02                     # needle tip waits this far outside the skin [m]
F_CUT, F_FRICTION = 0.5, 20.0       # axial resistance = F_CUT + F_FRICTION * depth  [N], [N/m]
K_LATERAL = 300.0                   # tissue pushes the shaft back to the entry point [N/m]
T_HISTORY = 20.0                    # the estimators have watched the target this long before 'ready'


def frame_from_axis(u):
    """Rotation matrix whose z-axis is the needle direction u (roll chosen to stay near home)."""
    z = u / np.linalg.norm(u)
    x = np.array([0.0, -1.0, 0.0])
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def min_jerk(tau):
    """Smooth 0 -> 1 profile (position, velocity) for tau in 0..1."""
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5, 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4


def run_insertion(sub: SubjectParams, method="ekf", speed=0.02, trial=0, latency=0.1,
                  meas_rate=25.0, meas_noise=0.0005, log=False, frame_cb=None):
    """Simulate one insertion. `speed` is the mean insertion speed [m/s]."""
    m = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    breath = Breathing(sub, 80.0)
    period = 60.0 / sub.rate_cpm
    rng = np.random.default_rng(70_000 + 1000 * sub.seed + trial)
    t_ready = T_HISTORY + rng.uniform(0, period)          # random moment in the breathing cycle
    t0 = t_ready - 1.0                                     # robot simulation starts 1 s earlier

    # ---- true target motion (y, z) and true skin height -----------------------------------
    def target(t):
        s, _ = breath.s(t)
        return np.array([ENTRY_XY[0],
                         ENTRY_XY[1] + sub.target_pp * (0.5 - s),      # centred under the entry point
                         SKIN_Z0 - TARGET_DEPTH + AP_RATIO * sub.surface_pp * s])
    skin_z = lambda t: SKIN_Z0 + breath.surface(t)[0]
    z_skin_mean = float(np.mean([skin_z(t) for t in np.arange(0, T_HISTORY, 0.05)]))
    entry = np.array([ENTRY_XY[0], ENTRY_XY[1], z_skin_mean])          # planned entry point

    # ---- sensing: delayed, noisy target position; one filter pair per axis -------------------
    streams = [MeasurementStream(lambda t, i=i: target(t)[i], rate=meas_rate, latency=latency,
                                 noise_std=meas_noise, seed=sub.seed * 10 + i + 100 * trial) for i in (1, 2)]
    r = meas_noise ** 2
    cvs = [CVKalman(q=CV_Q, r=r) for _ in range(2)]
    ekfs = [HarmonicEKF(q_amp=EKF_Q_AMP, q_omega=EKF_Q_OMEGA, r=r) for _ in range(2)]
    history = [[], []]                                     # (t_acquired, z) for the 'mean' strategy

    def sense(t):
        for i in range(2):
            for z, ta in streams[i].poll(t):
                cvs[i].update(z, ta)
                ekfs[i].update(z, ta)
                history[i].append((ta, z))

    sense(t0)                                              # everything measured before the robot starts

    # ---- strategy: aim point for an insertion that would start at time t and last T ----------
    def aim_point(t, T):
        if method == "oracle":
            return target(t + T)
        if method == "naive":
            yz = [cvs[i].x[0] for i in range(2)]
        elif method == "mean":
            yz = [np.mean([z for ta, z in history[i] if ta > t - 12.0]) for i in range(2)]
        elif method in ("cv", "cv_raw"):
            yz = [cvs[i].predict(t + T)[0] for i in range(2)]
            if method == "cv":                             # keep the aim inside the range seen so far;
                for i in range(2):                         # 'cv_raw' shows what happens without this
                    seen = [z for ta, z in history[i] if ta > t - 12.0]
                    yz[i] = float(np.clip(yz[i], min(seen), max(seen)))
        else:
            yz = [ekfs[i].predict(t + T)[0] for i in range(2)]
        return np.array([ENTRY_XY[0], yz[0], yz[1]])

    def calm_at_arrival(t, T):
        """Waiting rule: is the target predicted to be resting at end-of-exhale when the tip
        arrives, and still resting 0.3 s later (margin for timing error)?"""
        if method == "oracle":
            pred, P = (lambda tt: target(tt)[1]), period
        else:
            pred, P = (lambda tt: ekfs[0].predict(tt)[0]), 2 * np.pi / ekfs[0].x[6]
        future = np.array([pred(t + h) for h in np.arange(0.0, 1.5 * P, 0.05)])
        threshold = future.max() - 0.10 * (future.max() - future.min())    # top 10 % = exhale rest
        return pred(t + T) >= threshold and pred(t + T + 0.3) >= threshold

    def plan(t):
        """Return aim point, start point (tip outside the skin), needle direction, length, duration."""
        T = 2.0
        for _ in range(3):                                 # T depends on the path length: iterate
            A = aim_point(t, T)
            u = (A - entry) / np.linalg.norm(A - entry)
            S = entry - STANDOFF * u
            length = np.linalg.norm(A - S)
            T = length / speed
        return A, S, u, length, T

    # ---- robot start pose: tip at the start point, needle along the planned line -----------
    sid = m.site("tcp").id
    body_needle = m.body("needle").id
    qa_breath = m.jnt_qposadr[m.joint("breath").id]
    a_breath = m.actuator("breath_servo").id
    mocap_id = m.body("target").mocapid[0]
    A, S, u, length, T = plan(t_ready)
    d.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(300):                                   # damped least-squares IK
        mujoco.mj_forward(m, d)
        e = np.concatenate([S - d.site_xpos[sid], _rot_err(d.site_xmat[sid].reshape(3, 3), frame_from_axis(u))])
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        d.qpos[:7] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
    q_home = d.qpos[:7].copy()
    d.qpos[qa_breath], d.qvel[:] = breath.surface(t0)[0], 0.0
    d.time = t0
    mujoco.mj_forward(m, d)

    KP, KD, KPR, KDR, KN, DN = 900.0, 60.0, 900.0, 60.0, 20.0, 6.0
    Minv_full, eye_nv = np.zeros((m.nv, m.nv)), np.eye(m.nv)
    t_start = t_arrive = None
    pierce = None                                          # (x, y) where the needle went through the skin
    rcm_max, result, L = 0.0, None, []

    step = -1
    while True:
        step += 1
        t = d.time
        d.ctrl[a_breath] = breath.surface(t)[0]
        tgt = target(t)
        d.mocap_pos[mocap_id] = tgt
        sense(t)

        # ---- decide: keep aiming, or start the insertion -----------------------------------
        if t_start is None:
            if step % 20 == 0:                             # planning runs at 50 Hz
                A, S, u, length, T = plan(t)
                go = t >= t_ready and (method in ("naive", "mean", "cv", "cv_raw") or calm_at_arrival(t, T))
                if go or t > t_ready + 3 * period:         # safety: never wait more than 3 breaths
                    t_start, t_arrive = t, t + T
            p_des, v_des = S, np.zeros(3)
        else:
            s, ds = min_jerk((t - t_start) / T)
            p_des, v_des = S + s * length * u, ds / T * length * u
        Rd = frame_from_axis(u)

        # ---- task-space torque control (same structure as the ultrasound controller) ----------
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        mujoco.mj_solveM(m, d, Minv_full, eye_nv)
        Minv = Minv_full[:7, :7]
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-6 * np.eye(6))
        p, v = d.site_xpos[sid].copy(), J @ d.qvel[:7]
        R = d.site_xmat[sid].reshape(3, 3)
        acc = np.concatenate([KP * (p_des - p) + KD * (v_des - v[:3]), KPR * _rot_err(R, Rd) - KDR * v[3:]])
        Jbar = Minv @ J.T @ Lam
        tau = J.T @ (Lam @ acc) + (np.eye(7) - J.T @ Jbar.T) @ (KN * (q_home - d.qpos[:7]) - DN * d.qvel[:7])
        d.ctrl[:7] = np.clip(tau + d.qfrc_bias[:7], m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])

        # ---- needle-tissue forces (simple model) + entry-point bookkeeping --------------------
        d.qfrc_applied[:] = 0.0
        axis = R[:, 2]
        zs = skin_z(t)
        rcm = 0.0
        if p[2] < zs:                                      # tip is below the skin
            if pierce is None:
                pierce = p[:2].copy()
            depth = (zs - p[2]) / max(-axis[2], 1e-6)      # length of shaft inside tissue
            entry_now = np.array([pierce[0], pierce[1], zs])
            off = entry_now - p                            # from tip to the entry point
            lateral = off - (off @ axis) * axis            # sideways distance shaft <-> entry point
            rcm = float(np.linalg.norm(lateral))
            rcm_max = max(rcm_max, rcm)
            # tanh instead of sign: no chattering when the needle is (almost) at rest
            f_axial = -np.tanh((v[:3] @ axis) / 0.001) * (F_CUT + F_FRICTION * depth) * axis
            mujoco.mj_applyFT(m, d, f_axial, np.zeros(3), p, body_needle, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, K_LATERAL * lateral, np.zeros(3), p + (off @ axis) * axis,
                              body_needle, d.qfrc_applied)

        if log:
            L.append((t, *p, *tgt, *A, rcm, 0.0 if t_start is None else 1.0))
        if frame_cb is not None:
            frame_cb(m, d)

        # ---- score at the planned arrival time; then hold 0.3 s and stop ----------------------
        if t_arrive is not None and result is None and t >= t_arrive:
            err = p - tgt
            result = dict(miss_mm=1e3 * float(np.linalg.norm(err)), miss_y_mm=1e3 * float(err[1]),
                          miss_z_mm=1e3 * float(err[2]),
                          aim_error_mm=1e3 * float(np.linalg.norm(A - tgt)),       # planning part
                          exec_error_mm=1e3 * float(np.linalg.norm(p - A)),        # robot part
                          wait_s=float(t_start - t_ready), insert_s=float(T))
        if result is not None and t >= t_arrive + 0.3:
            break
        mujoco.mj_step(m, d)
        if d.time < t or not np.isfinite(d.qpos).all():    # MuJoCo reset itself: the simulation blew up
            raise RuntimeError(f"unstable simulation ({method}, seed {sub.seed}, trial {trial})")

    result["rcm_max_mm"] = 1e3 * rcm_max
    if log:
        result["log"] = np.array(L)
    return result
