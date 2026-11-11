"""One demo episode on the nominal patient.

  python3 experiments/demo.py            -> saves outputs/demo_log.npz (for the figure)
  python3 experiments/demo.py video      -> also renders outputs/demo.mp4
  python3 experiments/demo.py view       -> opens the interactive MuJoCo viewer (needs a screen)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import nominal_subject     # noqa: E402
from breathus.sim import run_episode               # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
mode = sys.argv[1] if len(sys.argv) > 1 else "log"
cb, finish = None, lambda: None

if mode == "video":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import imageio
    import mujoco
    state = {}
    writer = imageio.get_writer(os.path.join(OUT, "demo.mp4"), fps=25)

    def cb(m, d):
        if "r" not in state:
            state["r"] = mujoco.Renderer(m, 480, 720)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.42, 0.0, 0.33]
            cam.distance, cam.azimuth, cam.elevation = 1.35, 140, -18
            state["cam"] = cam
        if int(round(d.time * 1000)) % 40 == 0:            # 25 frames per simulated second
            state["r"].update_scene(d, state["cam"])
            writer.append_data(state["r"].render())
    finish = writer.close

elif mode == "view":
    import mujoco.viewer
    state = {}

    def cb(m, d):
        if "v" not in state:
            state["v"] = mujoco.viewer.launch_passive(m, d)
            state["t0"] = time.time()
        if int(round(d.time * 1000)) % 20 == 0:
            state["v"].sync()
            time.sleep(max(0.0, d.time - (time.time() - state["t0"])))   # real-time pacing

duration = 30.0 if mode == "video" else 60.0
res = run_episode(nominal_subject(), "force", "ekf", latency=0.1, duration=duration, log=True, frame_cb=cb)
finish()
log = res.pop("log")
print(res)
if mode == "log":
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "demo_log.npz"), **{k: v[::10] for k, v in log.items()})
