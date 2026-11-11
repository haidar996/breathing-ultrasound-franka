"""Turn the saved results into figures and a results table.

  python3 experiments/make_figures.py

Reads  outputs/closed_loop.json, outputs/offline_sweep.json, outputs/demo_log.npz (whatever exists)
Writes outputs/fig_*.png and outputs/results_table.md
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
C = {"static": "#8a8a8a", "none": "#8a8a8a", "delayed": "#d98b2b", "cv": "#2b7bba", "ekf": "#c0392b",
     "oracle": "#2e8b57"}


def boot_ci(x, n=5000, seed=0):
    """95 % bootstrap confidence interval of the mean (resampling subjects)."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def fig_offline():
    p = os.path.join(OUT, "offline_sweep.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    L = 1e3 * np.array(r["latencies"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    labels = {"none": "no prediction (newest measurement)", "cv": "constant-velocity KF  (p + v·t)",
              "ekf": "harmonic EKF (periodic model)"}
    for name in ("none", "cv", "ekf"):
        a = np.array(r[name])                              # latency x subject
        mean = a.mean(axis=1)
        ci = np.array([boot_ci(row) for row in a])
        ax.plot(L, mean, "o-", color=C[name], label=labels[name])
        ax.fill_between(L, ci[:, 0], ci[:, 1], color=C[name], alpha=0.2, lw=0)
    ax.set_xlabel("measurement latency [ms]")
    ax.set_ylabel("prediction error, RMS [mm]")
    ax.set_title(f"Predicting the target position ({a.shape[1]} virtual patients, mean and 95 % CI)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_latency_sweep.png"), dpi=160)


def table_closed_loop():
    p = os.path.join(OUT, "closed_loop.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    conds = sorted({k.split("|")[0] for k in r})
    seeds = sorted({int(k.split("|")[1]) for k in r})
    full = [s for s in seeds if all(f"{c}|{s}" in r for c in conds)]     # subjects with every condition
    lines = [f"Subjects with all conditions finished: {len(full)} (seeds {full})", "",
             "| condition | tracking RMS [mm] (95 % CI) | force RMSE [N] (95 % CI) | mean force [N] | max force [N] | time >150 % [%] | contact lost [%] |",
             "|---|---|---|---|---|---|---|"]
    for c in conds:
        g = lambda key: np.array([r[f"{c}|{s}"][key] for s in full])
        tr, fr = g("track_rmse_mm"), g("force_rmse")
        tci, fci = boot_ci(tr), boot_ci(fr)
        lines.append(f"| {c} | {tr.mean():.2f} ({tci[0]:.2f}–{tci[1]:.2f}) | {fr.mean():.2f} ({fci[0]:.2f}–{fci[1]:.2f}) "
                     f"| {g('force_mean').mean():.2f} | {g('force_max').max():.2f} | {g('pct_over_150').mean():.2f} | {g('pct_lost').mean():.2f} |")
    # paired comparison EKF vs CV (same subjects)
    for tag in ("L100", "L300"):
        if f"cv_{tag}" in conds and f"ekf_{tag}" in conds and full:
            diff = np.array([r[f"cv_{tag}|{s}"]["track_rmse_mm"] - r[f"ekf_{tag}|{s}"]["track_rmse_mm"] for s in full])
            ci = boot_ci(diff)
            lines += ["", f"Paired difference CV − EKF, {tag}: {diff.mean():.2f} mm (95 % CI {ci[0]:.2f} to {ci[1]:.2f}); "
                          f"EKF better in {int((diff > 0).sum())}/{len(full)} subjects"]
    open(os.path.join(OUT, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # bar figure: tracking error per method at 100 ms
    order = [c for c in ("static_L100", "delayed_L100", "cv_L100", "ekf_L100", "oracle_L100") if c in conds]
    if order and full:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for i, c in enumerate(order):
            v = np.array([r[f"{c}|{s}"]["track_rmse_mm"] for s in full])
            ax.bar(i, v.mean(), color=C[c.split("_")[0]], width=0.6)
            ax.plot(np.full(len(v), i), v, "k.", ms=4, alpha=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["probe not\nmoving", "follow, no\nprediction", "p + v·t\n(CV-KF)", "harmonic\nEKF", "perfect\nknowledge"])
        ax.set_ylabel("probe-to-target error, RMS [mm]")
        ax.set_title(f"Following the moving target, 100 ms latency ({len(full)} patients)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_tracking.png"), dpi=160)


def fig_demo():
    p = os.path.join(OUT, "demo_log.npz")
    if not os.path.exists(p):
        return
    L = np.load(p)
    t = L["t"]
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 7), sharex=True)
    ax[0].plot(t, 1e3 * (L["z_surface"] - L["z_surface"][0]), color="#555")
    ax[0].set_ylabel("skin height [mm]")
    ax[0].set_title("One patient, force control + harmonic EKF (100 ms latency)")
    ax[1].plot(t, L["f_true"], color=C["ekf"])
    ax[1].axhline(5, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("contact force [N]")
    ax[1].set_ylim(0, 8)
    ax[2].plot(t, 1e3 * L["y_target"], color="k", label="target (truth)")
    ax[2].plot(t, 1e3 * L["y_tcp"], color=C["ekf"], lw=1, label="probe")
    ax[2].set_ylabel("head–feet position [mm]")
    ax[2].set_xlabel("time [s]")
    ax[2].legend(frameon=False, ncol=2)
    for a in ax:
        a.axvspan(0, 20, color="#eee", zorder=0)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_demo.png"), dpi=160)


if __name__ == "__main__":
    fig_offline()
    table_closed_loop()
    fig_demo()
