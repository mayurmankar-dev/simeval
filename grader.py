"""Grader: turns one rollout into a verdict + metrics.

The grader is deliberately independent of the controller:
  - the controller stops at 2 cm, but the grader re-measures after letting physics settle,
  - it checks things the controller never looks at (cube on table, upright, time budget).
"""
import mujoco
import numpy as np
from env import TABLE_TOP, HALF

SUCCESS_DIST = 0.03      # cube centre within 3 cm of target centre
MAX_TILT_DEG = 15        # cube must still be flat on the table
SETTLE_TIME = 0.5        # seconds of physics after the rollout before measuring


def settle(env, seconds=SETTLE_TIME):
    """Let the cube come to rest (it may still be sliding when the controller stops)."""
    for _ in range(int(seconds / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)


def grade(env, scene, trace, n_realign, max_time=15.0):
    settle(env)
    cube, target = env.cube_pos(), env.target_pos()
    final_dist = float(np.linalg.norm(cube[:2] - target[:2]))
    start_dist = float(np.hypot(scene["target_x"] - scene["cube_x"], scene["target_y"] - scene["cube_y"]))

    # tilt = angle between the cube's local z axis and world up
    z_axis = env.data.body("cube").xmat.reshape(3, 3)[:, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(z_axis[2]), -1, 1))))

    xy = np.array([[p[2], p[3]] for p in trace])
    cube_path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
    net_move = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    duration = float(trace[-1][0])
    on_table = cube[2] > TABLE_TOP

    # --- verdict: the first failure reason that applies (order matters) ---
    if not on_table:
        outcome = "fell_off_table"
    elif tilt_deg > MAX_TILT_DEG:
        outcome = "tipped_over"
    elif final_dist <= SUCCESS_DIST:
        outcome = "success"
    elif duration >= max_time - 0.05:
        outcome = "timeout_near" if final_dist < 0.08 else "timeout_far"
    else:
        outcome = "stopped_short"

    return {
        "scene_id": scene["scene_id"],
        "outcome": outcome,
        "success": outcome == "success",
        "final_dist_cm": round(final_dist * 100, 2),
        "start_dist_cm": round(start_dist * 100, 2),
        "progress": round(1 - final_dist / start_dist, 3),     # 1.0 = perfect, <0 = moved away
        "tilt_deg": round(tilt_deg, 1),
        "duration_s": round(duration, 2),
        "n_realign": n_realign,
        "cube_path_cm": round(cube_path * 100, 1),
        "path_efficiency": round(net_move / cube_path, 3) if cube_path > 0 else 0.0,  # 1.0 = straight line
        # scene parameters copied in, so every result is traceable to its scene
        "mass": scene["mass"], "friction": scene["friction"], "cube_yaw": scene["cube_yaw"],
    }


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    from env import PushEnv
    from controller import run_push

    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scenes_seed42.json")
    p.add_argument("--scene", type=int, default=0)
    args = p.parse_args()
    scenes = json.loads((Path(__file__).parent / args.scenes).read_text())
    s = next(x for x in scenes if x["scene_id"] == args.scene)

    env = PushEnv()
    env.reset(s)
    trace, n_realign = run_push(env)
    result = grade(env, s, trace, n_realign)
    for k, v in result.items():
        print(f"  {k:16s} {v}")