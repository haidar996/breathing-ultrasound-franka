# Research notes — where the idea and every number come from

Legend: ✅ read by me in the paper's own text (PDF → `pdftotext`) · 🔎 seen only in a search-result
summary, **not** verified in the full text · ⚠️ my own assumption (no source).

## 1. Why this topic

| Fact | Status | Source |
|---|---|---|
| Keeping a steady probe contact is hard "where instinctive movements of the body, such as respiration" occur | ✅ | Compliant US end-effector paper, arXiv 2410.03086 |
| Compliant force control alone can compensate small breathing motion (Esteban 2018); constant-force control used for long liver/prostate monitoring (Ipsen 2021); visual servoing with a **predictive controller** used to stabilise respiratory motion (Nadeau 2011) | ✅ | Jiang, Salcudean, Navab, *Robotic Ultrasound Imaging: State-of-the-Art and Future Perspectives*, Medical Image Analysis 2023, arXiv 2307.05545, §5.1.1 and §5.3.1 |
| ICube (Strasbourg) treated breathing and heartbeat as periodic disturbances and cancelled them with predictive control (repetitive GPC for breathing, adaptive disturbance predictor for the heart), tested in vivo | 🔎 | Ginhoux, Gangloff, de Mathelin et al., *Active filtering of physiological motion in robotized surgery using predictive control*, IEEE Trans. Robotics 2005 — **read this paper before quoting it** |
| HealthTech master: coordinator Prof. Florent Nageotte; M2 elective "Medical robotics"; research axis "systems for assisting diagnosis and surgical procedures — interventional imaging, instrumentation, robotics" | ✅ (your own dossier, 15 Oct 2026) | `~/uni/FRANce/Université de Strasbourg/Université de Strasbourg.md` |

## 2. Numbers used in the simulation

| Quantity | Value used | Status | Source / reason |
|---|---|---|---|
| Abdominal wall motion, quiet breathing, supine | 9.9 mm, 14.6 breaths/min (nominal subject) | ✅ as quoted by 2410.03086 from Kaneko & Horie, *Respiratory Care* 2012 (I did not read Kaneko & Horie itself) | 2410.03086 says "amplitude of 9.9 mm" in one place and "9.9 mm displacement" in another; I use it as **peak-to-peak**. Random subjects cover 5–15 mm. |
| Contact force | 5 N | ✅ | 2410.03086 tests 2.5, 5, 10, 15 N |
| "% of time above 150 % of target force" metric | same definition | ✅ | 2410.03086, Table III |
| Real-robot reference point: UR3e arm alone, 5 N target, breathing phantom | RMSE 4.30 N; with their compliant end-effector 0.77 N | ✅ | 2410.03086, Table III. **Not comparable 1:1 with this simulation** (different robot, real friction, slower control loop). |
| Breathing shape `cos^(2n)` | n ∈ {1, 2, 3} | 🔎 | Lujan et al., Med. Phys. 26(5), 1999 (formula seen in search summary and well known; paper not read) |
| Internal target (liver) head–feet motion | 15 mm nominal; 5–25 mm random | 🔎 / ⚠️ | Search summary: one 4DCT study reports 5.7 ± 3.4 mm, others "several centimetres"; there is no single agreed value. The range is my choice. |
| Breath-to-breath variability | period ±5 %, depth ±10 %, drift 10 % | ⚠️ | chosen so the signal is not perfectly periodic; not fitted to patient data |
| Soft-tissue stiffness under the probe | 1000 N/m nominal; 500–2000 N/m random | ⚠️ | order of magnitude only (5 N ↔ 2.5–10 mm indentation) |
| Image-tracking sensor: 25 Hz, 0.5 mm noise, 100 ms latency (300 ms as a stress case) | ⚠️ | typical-looking values, no source |
| Force sensor noise 0.05 N, 30 Hz low-pass | ⚠️ | assumption |
| Probe–skin friction 0.1 (ultrasound gel) | ⚠️ | assumption |

## 3. What is idealised (say this openly)

- No ultrasound **image** is simulated. The "image tracker" is the true target position plus noise and delay.
- Tissue is one spring–damper under a flat plate, not a deformable body. The skin does not slide with breathing.
- Surface motion and organ motion are perfectly in phase (real organs lag and show hysteresis).
- The robot has ideal torque control: joint damping exists in the model (1 N·m·s/rad, unknown to the
  controller) but there is no Coulomb friction, no torque ripple, no communication delay.
- Breathing is synthetic. No recorded patient traces were used.
- The patient only breathes: no coughing, talking or shifting.

## 4. Part 2 — needle insertion: assumptions

Nothing in this table comes from a paper I read; every value is ⚠️ my assumption. Find sources
(or measure) before presenting any of them as realistic.

| Quantity | Value used | Comment |
|---|---|---|
| Target depth below the skin | 60 mm | fixed for all patients |
| Target up/down motion | 0.5 × skin motion | head–feet motion as in Part 1 |
| Mean insertion speed | 10, 20, 40 mm/s | swept, because the right clinical value is not known to me |
| Needle | rigid, 150 mm, no bending | real needles deflect in tissue |
| Axial tissue resistance | 0.5 N + 20 N/m × inserted depth | simple friction-like model, opposes motion |
| Sideways tissue stiffness at the entry point | 300 N/m | pushes the shaft back to the puncture point |
| Entry point | fixed on the skin above the middle of the target's path | chosen from the known motion range |
| "Calm phase" rule | start when the predicted target is in the top 10 % of its range (end of exhale) at arrival and 0.3 s later | my design |

Extra idealisations for Part 2:
- The organ does **not** drag or bend the needle, and the needle does not push the target away.
  Only the position of the tip against the target at the arrival time is scored.
- The whole insertion is one continuous stroke. Clinically, insertion is often done during a
  breath-hold ("insertions are generally done during a short patient's apnea" — 🔎 seen in a
  search summary about ICube-related work, not verified).
- A reader from interventional radiology (e.g. ICube's steerable-needle and finite-element
  needle–tissue work) will find this tissue model very simple. Say so first.
