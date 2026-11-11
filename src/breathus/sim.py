"""Closed-loop simulation: Franka + ultrasound probe on a breathing patient.

One call to run_episode() simulates one patient with one controller setting and
returns the metrics (and, optionally, the full time log).

Axes (world frame):  x = across the patient,  y = feet -> head,  z = up.
The probe points straight down (-z).
  z axis : contact force regulation   (force_mode  = 'force' | 'impedance')
  y axis : follow the internal target (track_mode  = 'static' | 'delayed' | 'cv' | 'ekf' | 'oracle')
  x axis and orientation: held fixed.
"""
import os

import mujoco
import numpy as np

from .breathing import Breathing, SubjectParams
from .estimators import CVKalman, HarmonicEKF, MeasurementStream

SCENE = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "franka_emika_panda",
                     "breathing_us_scene.xml")

# ---- fixed settings (see docs/research_notes.md for where each number comes from) ----------
F_DES = 5.0            # desired contact force [N]
CONTACT_XY = np.array([0.5, 0.0])   # where the probe touches the abdomen (world x, y) [m]
T_WARMUP = 20.0        # estimators converge / calibration; not scored [s]
FT_NOISE = 0.05        # force sensor noise, std [N]
FT_CUTOFF = 30.0       # force sensor low-pass cut-off [Hz]
G = 9.81

# tuned predictor settings (experiments/tune_predictors.py, tuning subjects only)
CV_Q = 1e-4
EKF_Q_AMP = 1e-7
EKF_Q_OMEGA = 1e-3


def load_model():
    return mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))


def solve_ik(m, d, pos, n_iter=200):
    """Damped least-squares IK: put the probe tip at `pos`, pointing straight down."""
    sid = m.site("tcp").id
    d.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    Rd = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])        # tip z-axis = world -z
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(n_iter):
        mujoco.mj_forward(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        e = np.concatenate([pos - d.site_xpos[sid], _rot_err(R, Rd)])
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        d.qpos[:7] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
    mujoco.mj_forward(m, d)
    return d.qpos[:7].copy()


def _rot_err(R, Rd):
    """Small-angle orientation error (world frame) that rotates R towards Rd."""
    e = np.zeros(3)
    for i in range(3):
        a, b = R[:, i], Rd[:, i]
        e += (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return 0.5 * e


def run_episode(sub: SubjectParams, force_mode="force", track_mode="ekf", latency=0.1,
                duration=80.0, meas_rate=25.0, meas_noise=0.0005, imp_stiffness=300.0,
                kf_p=0.8, kf_i=20.0, log=False, frame_cb=None):
    m = load_model()
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    breath = Breathing(sub, duration + 5)

    # ---- ids ---------------------------------------------------------------------------
    sid = m.site("tcp").id
    j_breath, j_tissue = m.joint("breath").id, m.joint("tissue").id
    qa_breath, qa_tissue = m.jnt_qposadr[j_breath], m.jnt_qposadr[j_tissue]
    m.jnt_stiffness[j_tissue] = sub.tissue_k
    a_breath = m.actuator("breath_servo").id
    ft_adr = m.sensor("ft_force").adr[0]
    tip_geom, pad_geom = m.geom("probe_tip").id, m.geom("skin_pad").id
    probe_mass = float(m.body("probe").mass[0])
    mocap_id = m.body("target").mocapid[0]
    target0 = np.array([CONTACT_XY[0], CONTACT_XY[1], 0.27])

    # ---- start pose: probe 3 cm above the skin, pointing down -----------------------------
    skin_z0 = 0.32                                            # skin height at exhale [m]
    d.qpos[qa_breath] = breath.surface(0.0)[0]
    q_home = solve_ik(m, d, np.array([CONTACT_XY[0], CONTACT_XY[1], skin_z0 + 0.03]))
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    Rd = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])

    # ---- sensing of the internal target (late + noisy) and the predictors -----------------
    y0 = CONTACT_XY[1]
    target_y = lambda t: y0 + breath.target(t)[0]
    stream = MeasurementStream(target_y, rate=meas_rate, latency=latency, noise_std=meas_noise,
                               seed=sub.seed)
    r = meas_noise ** 2
    cv = CVKalman(q=CV_Q, r=r)
    ekf = HarmonicEKF(q_amp=EKF_Q_AMP, q_omega=EKF_Q_OMEGA, r=r)

    # ---- controller gains -----------------------------------------------------------------
    KP, KD = 900.0, 60.0          # motion axes (x, y): task-space PD, natural freq 30 rad/s
    KPR, KDR = 900.0, 60.0        # orientation
    KF_P, KF_I = kf_p, kf_i       # force PI
    DZ = 120.0                    # damping along the probe axis [N s/m]
    KN, DN = 20.0, 6.0            # null-space posture
    rng = np.random.default_rng(90_000 + sub.seed)
    alpha = 1 - np.exp(-2 * np.pi * FT_CUTOFF * dt)

    f_filt, f_int, in_contact, t_contact = 0.0, 0.0, False, None
    z_sum, z_cnt, z_des = 0.0, 0, None
    jp, jr, M = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((m.nv, m.nv))
    eye_nv = np.eye(m.nv)
    n_steps = int(duration / dt)
    L = {k: np.zeros(n_steps) for k in ("t", "f_true", "f_meas", "y_tcp", "y_target", "y_ref",
                                        "z_tcp", "z_surface", "ekf_rate")} if log else None
    f_log, e_log = [], []

    for i in range(n_steps):
        t = d.time
        # ---- patient: breathing servo + draw the internal target ----------------------------
        d.ctrl[a_breath] = breath.surface(t)[0]
        yt, vt = target_y(t), breath.target(t)[1]
        d.mocap_pos[mocap_id] = [target0[0], yt, target0[2] + breath.surface(t)[0]]

        # ---- sensing --------------------------------------------------------------------
        for z, t_acq in stream.poll(t):
            cv.update(z, t_acq)
            ekf.update(z, t_acq)
        R = d.site_xmat[sid].reshape(3, 3)
        # force along the probe axis, with the probe's own weight removed (gravity compensation)
        f_raw = d.sensordata[ft_adr + 2] + probe_mass * G * (-R[2, 2])
        f_filt += alpha * (f_raw + FT_NOISE * rng.standard_normal() - f_filt)

        # ---- kinematics / dynamics -----------------------------------------------------
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        mujoco.mj_solveM(m, d, M, eye_nv)                         # M now holds inverse(mass matrix)
        Minv = M[:7, :7]                                         # robot block (patient is a separate tree)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-6 * np.eye(6))   # task-space inertia
        p, v = d.site_xpos[sid], J @ d.qvel[:7]

        # ---- y reference: where should the probe be to stay over the target? ----------------
        y_ref, vy_ref = y0, 0.0
        if track_mode == "oracle":
            y_ref, vy_ref = yt, vt
        elif track_mode in ("delayed", "cv") and cv.x is not None:
            y_ref, vy_ref = (cv.x[0], 0.0) if track_mode == "delayed" else cv.predict(t)
        elif track_mode == "ekf" and ekf.x is not None:
            y_ref, vy_ref = ekf.predict(t)
        y_ref = float(np.clip(y_ref, y0 - 0.05, y0 + 0.05))      # safety box: +-5 cm

        # ---- task-space command -----------------------------------------------------------
        acc = np.zeros(6)
        acc[0] = KP * (CONTACT_XY[0] - p[0]) - KD * v[0]
        acc[1] = KP * (y_ref - p[1]) + KD * (vy_ref - v[1])
        acc[3:] = KPR * _rot_err(R, Rd) - KDR * v[3:]
        wrench = Lam @ acc

        # z axis: contact force
        if not in_contact and f_filt > 1.0:
            in_contact, t_contact = True, t
        if in_contact:
            ramp = min(1.0, (t - t_contact) / 1.0)                # 2 N -> F_DES over 1 s
            f_des = 2.0 + (F_DES - 2.0) * ramp
        else:
            f_des = 2.0                                           # gentle approach
        use_impedance = force_mode == "impedance" and z_des is not None
        if use_impedance:
            fz = imp_stiffness * (z_des - p[2]) - DZ * v[2]
        else:
            e_f = f_des - f_filt
            if in_contact:
                f_int = float(np.clip(f_int + KF_I * e_f * dt, -5, 5))
            fz = -(f_des + KF_P * e_f * in_contact + f_int) - DZ * v[2]
            # impedance baseline: learn the mean contact height, then switch at t = 12 s
            if force_mode == "impedance" and in_contact:
                if 4.0 <= t < 12.0:
                    z_sum, z_cnt = z_sum + p[2], z_cnt + 1
                elif t >= 12.0 and z_cnt:
                    z_des = z_sum / z_cnt - F_DES / imp_stiffness
        wrench[2] += fz

        # ---- joint torques: task + null-space posture + gravity/Coriolis ---------------------
        Jbar = Minv @ J.T @ Lam
        tau = J.T @ wrench + (np.eye(7) - J.T @ Jbar.T) @ (KN * (q_home - d.qpos[:7]) - DN * d.qvel[:7])
        d.ctrl[:7] = np.clip(tau + d.qfrc_bias[:7], m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])

        # ---- ground truth for scoring (never used by the controller) -----------------------
        f_true = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if {con.geom1, con.geom2} == {tip_geom, pad_geom}:
                w6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, w6)
                f_true += w6[0]
        if t >= T_WARMUP:
            f_log.append(f_true)
            e_log.append(p[1] - yt)
        if log:
            for k, val in (("t", t), ("f_true", f_true), ("f_meas", f_filt), ("y_tcp", p[1]),
                           ("y_target", yt), ("y_ref", y_ref), ("z_tcp", p[2]),
                           ("z_surface", skin_z0 + d.qpos[qa_breath] + d.qpos[qa_tissue]),
                           ("ekf_rate", ekf.rate_cpm or np.nan)):
                L[k][i] = val
        if frame_cb is not None:
            frame_cb(m, d)
        mujoco.mj_step(m, d)

    if not f_log:                                  # episode shorter than the warm-up: nothing to score
        return dict(log=L) if log else {}
    f, e = np.array(f_log), np.array(e_log) * 1e3
    out = dict(
        force_rmse=float(np.sqrt(np.mean((f - F_DES) ** 2))), force_mean=float(f.mean()),
        force_max=float(f.max()), force_min=float(f.min()),
        pct_over_150=float(100 * np.mean(f > 1.5 * F_DES)), pct_lost=float(100 * np.mean(f < 0.5)),
        track_rmse_mm=float(np.sqrt(np.mean(e ** 2))), track_p95_mm=float(np.percentile(np.abs(e), 95)),
        ekf_rate_cpm=float(ekf.rate_cpm or np.nan),
    )
    if log:
        out["log"] = L
    return out
