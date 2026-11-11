"""Table and figure for the needle study.   python3 experiments/needle_figures.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
METHODS = ("naive", "mean", "cv", "ekf", "oracle")
LABEL = {"naive": "aim at current\nposition", "mean": "aim at average\nposition", "cv": "p + v·t\n(CV-KF)",
         "ekf": "wait for exhale\n(harmonic EKF)", "oracle": "wait for exhale\n(perfect knowledge)"}
COLOR = {"naive": "#8a8a8a", "mean": "#d98b2b", "cv": "#2b7bba", "ekf": "#c0392b", "oracle": "#2e8b57"}

r = json.load(open(os.path.join(OUT, "needle_study.json")))
keys = [k.split("|") for k in r]
speeds = sorted({float(k[1]) for k in keys})
# keep only (speed, patient, trial) cases where every strategy finished -> paired comparison
cases = {v: sorted({(int(k[2]), int(k[3])) for k in keys if float(k[1]) == v
                    and all(f"{m}|{k[1]}|{k[2]}|{k[3]}" in r for m in METHODS)}) for v in speeds}


def per_patient(method, v, field="miss_mm"):
    """Average the trials of each patient first: patients, not trials, are the independent units."""
    by = {}
    for s, k in cases[v]:
        by.setdefault(s, []).append(r[f"{method}|{v}|{s}|{k}"][field])
    return np.array([np.mean(x) for x in by.values()])


def boot_ci(x, n=5000):
    rng = np.random.default_rng(0)
    return np.percentile(rng.choice(x, (n, len(x))).mean(axis=1), [2.5, 97.5])


lines = []
for v in speeds:
    if not cases[v]:
        continue
    n_pat = len({s for s, _ in cases[v]})
    lines += [f"### Mean insertion speed {1e3 * v:.0f} mm/s — {n_pat} patients, {len(cases[v])} insertions per strategy", "",
              "| strategy | miss distance, mean [mm] (95 % CI) | worst insertion [mm] | within 2 mm [%] | wait [s] | entry-point side shift, max [mm] |",
              "|---|---|---|---|---|---|"]
    for m in METHODS:
        x = per_patient(m, v)
        allm = np.array([r[f"{m}|{v}|{s}|{k}"]["miss_mm"] for s, k in cases[v]])
        ci = boot_ci(x)
        lines.append(f"| {m} | {x.mean():.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {allm.max():.2f} | {100 * np.mean(allm <= 2):.0f} "
                     f"| {per_patient(m, v, 'wait_s').mean():.1f} | {per_patient(m, v, 'rcm_max_mm').mean():.2f} |")
    for base in ("naive", "mean", "cv"):
        dlt = per_patient(base, v) - per_patient("ekf", v)
        ci = boot_ci(dlt)
        lines.append(f"\nPaired difference {base} − ekf: {dlt.mean():.2f} mm (95 % CI {ci[0]:.2f} to {ci[1]:.2f}); "
                     f"ekf better in {int((dlt > 0).sum())}/{len(dlt)} patients")
    lines.append("")
open(os.path.join(OUT, "needle_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
w = 0.8 / len(speeds)
for j, v in enumerate(speeds):
    for i, m in enumerate(METHODS):
        if not cases[v]:
            continue
        x = per_patient(m, v)
        ci = boot_ci(x)
        ax.bar(i + (j - (len(speeds) - 1) / 2) * w, x.mean(), w * 0.9, color=COLOR[m], alpha=0.45 + 0.25 * j,
               yerr=[[x.mean() - ci[0]], [ci[1] - x.mean()]], capsize=2)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=8)
ax.set_ylabel("needle tip miss distance [mm]")
ax.set_title("Needle insertion to a breathing target — bars per strategy: " + ", ".join(f"{1e3 * v:.0f}" for v in speeds) + " mm/s")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_needle.png"), dpi=160)
