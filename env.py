"""Builds the Panda + table scene and applies scene parameters from scene_gen."""
import mujoco
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
PANDA_XML = ROOT / "mujoco_menagerie/franka_emika_panda/panda.xml"
TABLE_TOP, HALF = 0.40, 0.02
TCP_OFFSET = 0.11   # fingertip point: 11 cm below the 'hand' body along its z axis


def build_model():
    """Attach the Panda (robot) to our task scene (table, cube, target) using MjSpec."""
    scene = mujoco.MjSpec.from_file(str(ROOT / "task_scene.xml"))
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    frame = scene.worldbody.add_frame(pos=[0, 0, 0])   # robot base at the origin
    frame.attach_body(panda.body("link0"), "panda/", "")
    return scene.compile()


class PushEnv:
    def __init__(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.cube_body = m.body("cube").id
        self.cube_geom = m.geom("cube_geom").id
        self.target_site = m.site("target").id
        self.hand = m.body("panda/hand").id
        self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.cube_body]]
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"panda/joint{i}").id] for i in range(1, 8)])
        self.arm_dofadr = np.array([m.jnt_dofadr[m.joint(f"panda/joint{i}").id] for i in range(1, 8)])
        self.home = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    def reset(self, s):
        m, d = self.model, self.data
        # 1) model parameters for this scene (physics properties)
        m.body_mass[self.cube_body] = s["mass"]
        m.body_inertia[self.cube_body] = s["mass"] * (2 * (2 * HALF) ** 2) / 12
        m.geom_friction[self.cube_geom, 0] = s["friction"]
        m.site_pos[self.target_site] = [s["target_x"], s["target_y"], TABLE_TOP + 0.001]
        mujoco.mj_setConst(m, d)   # recompute derived constants after changing mass (else: unstable!)
        # 2) initial state
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = self.home        # robot at home pose
        d.ctrl[:7] = self.home
        d.ctrl[7] = 0                             # gripper closed: fingers form a pusher
        yaw = s["cube_yaw"]
        q = self.cube_qadr
        d.qpos[q:q + 3] = [s["cube_x"], s["cube_y"], TABLE_TOP + HALF]
        d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)

    # --- measurements ---
    def cube_pos(self):
        return self.data.body("cube").xpos.copy()

    def target_pos(self):
        return self.data.site("target").xpos.copy()

    def tcp_pos(self):
        d = self.data
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ [0, 0, TCP_OFFSET]
