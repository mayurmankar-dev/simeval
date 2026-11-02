import mujoco
from pathlib import Path

XML = Path(__file__).parent / "first_scene.xml"   # full path (avoids the macOS issue)
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

print("timestep:", model.opt.timestep, "s")
print("bodies:", model.nbody, "| geoms:", model.ngeom)

steps_per_print = int(0.1 / model.opt.timestep)    # print every 0.1 s
for step in range(int(1.0 / model.opt.timestep)):  # simulate 1 second
    mujoco.mj_step(model, data)
    if step % steps_per_print == 0:
        z = data.body("cube").xpos[2]               # cube's height (x, y, z → index 2)
        print(f"t={data.time:.2f}s  height={z:.3f} m")