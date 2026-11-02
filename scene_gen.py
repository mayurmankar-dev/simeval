import argparse, json
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

ROOT = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(ROOT / "task_scene.xml"))
data = mujoco.MjData(model)

# Look up IDs once (MuJoCo stores everything in arrays indexed by ID)
cube_body = model.body("cube").id
cube_geom = model.geom("cube_geom").id
target_site = model.site("target").id
qadr = model.jnt_qposadr[model.body_jntadr[cube_body]]  # where the cube's pose lives in qpos

TABLE_TOP, HALF = 0.40, 0.02   # table height, half cube size (m)


def sample_scene(rng, scene_id):
    """Randomly sample one scene's parameters."""
    cube_xy = rng.uniform([0.35, -0.15], [0.55, 0.15])
    while True:  # target must be at least 10 cm from the cube
        target_xy = rng.uniform([0.35, -0.25], [0.70, 0.25])
        if np.linalg.norm(target_xy - cube_xy) > 0.10:
            break
    return {
        "scene_id": scene_id,
        "cube_x": float(cube_xy[0]), "cube_y": float(cube_xy[1]),
        "cube_yaw": float(rng.uniform(-np.pi, np.pi)),
        "target_x": float(target_xy[0]), "target_y": float(target_xy[1]),
        "mass": float(rng.uniform(0.05, 0.5)),       # kg
        "friction": float(rng.uniform(0.3, 1.2)),    # sliding friction
    }


def apply_scene(s):
    """Write a scene's parameters into the MuJoCo model/state."""
    mujoco.mj_resetData(model, data)
    yaw = s["cube_yaw"]
    data.qpos[qadr:qadr + 3] = [s["cube_x"], s["cube_y"], TABLE_TOP + HALF]
    data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # rotation about z
    model.body_mass[cube_body] = s["mass"]
    model.body_inertia[cube_body] = s["mass"] * (2 * (2 * HALF) ** 2) / 12  # box inertia
    model.geom_friction[cube_geom, 0] = s["friction"]
    model.site_pos[target_site] = [s["target_x"], s["target_y"], TABLE_TOP + 0.001]
    mujoco.mj_forward(model, data)


def is_valid():
    """Settle for 0.5 s; the cube must still be on the table."""
    for _ in range(250):
        mujoco.mj_step(model, data)
    return data.body("cube").xpos[2] > TABLE_TOP


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--view", type=int, default=None, help="open scene with this id in the viewer")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    scenes = [sample_scene(rng, i) for i in range(args.n)]

    if args.view is not None:
        apply_scene(scenes[args.view])
        mujoco.viewer.launch(model, data)
    else:
        valid = []
        for s in scenes:
            apply_scene(s)
            if is_valid():
                valid.append(s)
        out = ROOT / f"scenes_seed{args.seed}.json"
        out.write_text(json.dumps(valid, indent=2))
        print(f"{len(valid)}/{len(scenes)} valid scenes saved to {out.name}")
        for s in valid[:5]:
            print(f"  #{s['scene_id']:2d} cube=({s['cube_x']:.2f},{s['cube_y']:.2f}) "
                  f"target=({s['target_x']:.2f},{s['target_y']:.2f}) "
                  f"mass={s['mass']:.2f}kg friction={s['friction']:.2f}")