"""One needle insertion on the nominal patient, rendered to outputs/needle_demo.mp4.

  python3 experiments/needle_demo.py [strategy]      (default: ekf)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
import sys

import imageio
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from breathus.breathing import nominal_subject     # noqa: E402
from breathus.needle import run_insertion          # noqa: E402

method = sys.argv[1] if len(sys.argv) > 1 else "ekf"
out = os.path.join(os.path.dirname(__file__), "..", "outputs", f"needle_demo_{method}.mp4")
writer = imageio.get_writer(out, fps=25)
state = {}


def cb(m, d):
    if "r" not in state:
        state["r"] = mujoco.Renderer(m, 480, 720)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.5, 0.0, 0.33]                   # close-up, looking along the patient's side
        cam.distance, cam.azimuth, cam.elevation = 0.55, 180, -8
        state["cam"] = cam
    if int(round(d.time * 1000)) % 40 == 0:
        state["r"].update_scene(d, state["cam"])
        writer.append_data(state["r"].render())


res = run_insertion(nominal_subject(), method, speed=0.02, frame_cb=cb)
writer.close()
print({k: round(v, 2) for k, v in res.items()})
