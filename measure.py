import mujoco
import numpy as np
from pathlib import Path

XML = Path(__file__).parent / "task_scene.xml"
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

for _ in range(500):                       # let the cube settle (1 s)
    mujoco.mj_step(model, data)

cube = data.body("cube").xpos
target = data.site("target").xpos
dist_xy = np.linalg.norm(cube[:2] - target[:2])   # distance on the table plane

print("cube   (x, y, z):", np.round(cube, 3))
print("target (x, y, z):", np.round(target, 3))
print(f"distance to target: {dist_xy*100:.1f} cm")