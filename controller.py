"""Scripted push controller: inverse kinematics + a phase-based plan (approach, descend, push, re-align)."""
import mujoco
import numpy as np
from env import TABLE_TOP, HALF, TCP_OFFSET


def ik_step(env, goal, damping=0.05, max_step=0.02):
    """One damped-least-squares IK step: move the fingertip toward `goal`,
    keep the gripper pointing straight down, and stay near the home posture."""
    m, d = env.model, env.data
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, env.tcp_pos(), env.hand)
    J = np.vstack([jacp, jacr])[:, env.arm_dofadr]            # 6 x 7

    pos_err = goal - env.tcp_pos()
    z_axis = d.xmat[env.hand].reshape(3, 3)[:, 2]
    rot_err = np.cross(z_axis, [0, 0, -1])                       # tilt away from "down"
    err = np.concatenate([pos_err, rot_err])

    JJt = J @ J.T + damping ** 2 * np.eye(6)
    dq = J.T @ np.linalg.solve(JJt, err)
    # null space: drift back toward home posture without disturbing the task
    N = np.eye(7) - np.linalg.pinv(J) @ J
    dq += N @ (0.05 * (env.home - d.qpos[env.arm_qadr]))
    dq = np.clip(dq, -max_step, max_step)
    d.ctrl[:7] = d.qpos[env.arm_qadr] + dq


def run_push(env, max_time=15.0, success_dist=0.02, on_step=None):
    """Push the cube onto the target with a closed-loop plan:
    approach (hover behind cube) -> descend -> push (re-aligning if the cube drifts sideways).
    Returns a trace of (time, phase, cube_x, cube_y, cube_z) for the grader."""
    m, d = env.model, env.data
    phase = "approach"
    hover_z = TABLE_TOP + 0.10
    push_z = TABLE_TOP + 0.005        # fingertips just above the table; palm clears the cube top
    standoff = HALF + 0.015           # pusher distance from cube centre while pushing
    trace, n_realign, lift_xy = [], 0, None
    steps_per_ctrl = 5                # control at 100 Hz, physics at 500 Hz
    for step in range(int(max_time / m.opt.timestep)):
        if step % steps_per_ctrl == 0:
            cube, target, tcp = env.cube_pos(), env.target_pos(), env.tcp_pos()
            to_target = target[:2] - cube[:2]
            dist = np.linalg.norm(to_target)
            u = to_target / max(dist, 1e-6)                 # desired push direction
            behind = cube[:2] - u * 0.08                     # stand-off point behind the cube

            if phase == "approach":
                goal = np.r_[behind, hover_z]
                if np.linalg.norm(tcp - goal) < 0.01:
                    phase = "descend"
            elif phase == "descend":
                goal = np.r_[behind, push_z]
                if np.linalg.norm(tcp - goal) < 0.01:
                    phase = "push"
            elif phase == "push":
                rel = tcp[:2] - cube[:2]
                lateral = rel - (rel @ u) * u                # sideways misalignment
                if dist < success_dist:
                    phase = "done"
                elif np.linalg.norm(lateral) > 0.025 or rel @ u > -HALF:
                    # cube drifted off-line (or we overshot): lift and go around again
                    phase, n_realign = "lift", n_realign + 1
                    lift_xy = tcp[:2] - u * 0.03
                goal = np.r_[cube[:2] - u * standoff + u * 0.03 - 0.5 * lateral, push_z]
            if phase == "lift":
                goal = np.r_[lift_xy, hover_z]               # back off and rise, then re-approach
                if tcp[2] > hover_z - 0.01:
                    phase = "approach"
            if phase == "done":
                goal = np.r_[tcp[:2], hover_z]               # lift off so we don't disturb the cube
            ik_step(env, goal)
            trace.append((d.time, phase, *cube[:3]))
        mujoco.mj_step(m, d)
        if on_step:
            on_step()
        if phase == "done" and env.tcp_pos()[2] > TABLE_TOP + 0.08:
            break
    return trace, n_realign
