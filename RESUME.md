# RESUME — read this first in a new chat

Project: `~/breathing-ultrasound-franka` (started 2026-11-11). Medical-robotics portfolio project
for the Strasbourg HealthTech master application (Franka Panda, MuJoCo, simulation only).

## State (updated 2026-11-11, end of session)

**Part 1 — ultrasound probe on a breathing patient: DONE.**
Code `src/breathus/{breathing,estimators,sim}.py`; results in `README.md`, `outputs/results_table.md`.

**Part 2 — needle insertion to the moving target: DONE (first version).**
Code `src/breathus/needle.py`; study 720 insertions in `outputs/needle_study.json`;
results in `README.md`, `outputs/needle_table.md`, `outputs/fig_needle.png`.

All experiments are finished; nothing is running. `python3 tests/test_basics.py` = 5 checks.

## Open items (none started)
- Read the papers marked "unverified" in `docs/research_notes.md` before quoting them
  (Ginhoux/Gangloff T-RO 2005, Lujan 1999, Kaneko & Horie 2012).
- Every needle/tissue number is an assumption — find sources.
- Ideas: recorded breathing data; safety rule / step-wise gated insertion; joint friction +
  predictive force control; simulated ultrasound image.
- Not a git repository yet.

## Cautions
- Laptop has 8 GB RAM: run experiments with 1–2 workers, not during other heavy jobs.
- Scripts set `OMP_NUM_THREADS=1` before importing numpy (otherwise ~10x slower).
- MuJoCo option `cone="elliptic"` in the scenes is required (sliding-contact dropouts otherwise).
- Needle strategy `cv_raw` (no range limit) can make the simulation unstable; it raises an error.
- When stopping jobs with `pkill -f`, use a pattern like `needle_stud[y]` in a command by itself.
