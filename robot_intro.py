import mujoco
import numpy as np
from pathlib import Path

XML = Path(__file__).parent / "mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

print("Joints:")
for i in range(model.njnt):
    lo, hi = model.jnt_range[i]
    print(f"  {model.joint(i).name:12s} range=({lo:+.2f}, {hi:+.2f}) rad")

print("\nActuators (what data.ctrl controls):")
for i in range(model.nu):
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  ctrl[{i}] {model.actuator(i).name:10s} range=({lo:+.2f}, {hi:+.2f})")

# Start from the saved 'home' pose
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)                    # forward kinematics
print("\nHand position at home:", np.round(data.body("hand").xpos, 3))

# Command joint 1 (the base) to rotate by +0.8 rad, then simulate 2 s
data.ctrl[0] += 0.8
for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
print("Hand position after base rotation:", np.round(data.body("hand").xpos, 3))
print(f"Joint 1 angle: target={data.ctrl[0]:.3f}, actual={data.qpos[0]:.3f}")