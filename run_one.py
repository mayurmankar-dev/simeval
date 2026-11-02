"""Run the push controller on one scene from your scene dataset.

    python run_one.py --scene 0              # headless: fast, prints the result
    mjpython run_one.py --scene 0 --watch    # watch it live (macOS needs mjpython)
"""
import argparse, json, time
import numpy as np
import mujoco
from pathlib import Path
from env import PushEnv
from controller import run_push

p = argparse.ArgumentParser()
p.add_argument("--scenes", default="scenes_seed42.json")
p.add_argument("--scene", type=int, default=0)
p.add_argument("--watch", action="store_true")
args = p.parse_args()

scenes = json.loads((Path(__file__).parent / args.scenes).read_text())
s = next(x for x in scenes if x["scene_id"] == args.scene)
env = PushEnv()
env.reset(s)

if args.watch:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = env.model.camera("overview").id
        def on_step():                       # sync the window and run at ~real time
            if int(env.data.time / env.model.opt.timestep) % 10 == 0:
                viewer.sync()
                time.sleep(10 * env.model.opt.timestep)
        trace, n_realign = run_push(env, on_step=on_step)
        time.sleep(1.5)
else:
    trace, n_realign = run_push(env)

cube, target = env.cube_pos(), env.target_pos()
dist = np.linalg.norm(cube[:2] - target[:2])
print(f"scene {args.scene}: mass={s['mass']:.2f}kg friction={s['friction']:.2f}")
print(f"final phase: {trace[-1][1]}  time: {trace[-1][0]:.1f}s  re-alignments: {n_realign}")
print(f"final distance to target: {dist*100:.1f} cm  cube height: {cube[2]:.3f} m")
