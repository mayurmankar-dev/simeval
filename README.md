# SimEval: scene generation and graded rollouts for robot pushing in MuJoCo

A small evaluation harness for a robot manipulation task. It generates randomized simulation scenes, runs a robot controller in each one, grades every rollout with explicit rules, and reports the success rate with confidence intervals plus a breakdown of how it fails.

**Task:** a Franka Emika Panda arm pushes a 4 cm cube across a table onto a target marker.

![Outcomes](results/report/outcomes.png)

## Results (200 scenes, seed 0)

**Success rate: 147/200 = 73.5%** (95% Wilson CI: 67.0%–79.1%)

A rollout succeeds if, after 0.5 s of settling, the cube's centre is within 3 cm of the target, the cube is still on the table, and it is tilted less than 15°.

| Outcome | Count | Share |
|---|---|---|
| success | 147 | 73.5% |
| timeout_near (out of time, within 8 cm) | 27 | 13.5% |
| fell_off_table | 15 | 7.5% |
| timeout_far | 5 | 2.5% |
| tipped_over | 5 | 2.5% |
| stopped_short | 1 | 0.5% |

**Findings**
- **Push distance matters.** Success holds at about 80% for pushes up to ~23 cm, then drops to 57% for the longest pushes (28–47 cm).
- **Cube mass (0.05–0.5 kg) and friction (0.3–1.2) show no clear effect at this sample size.** The per-bin confidence intervals overlap heavily.
- **The largest failure mode is `timeout_near`** (13.5%): the cube is close but the controller runs out of time. That points to speed rather than strategy.
- **Fell-off-table failures (7.5%) are a controller bug, not physics.** All 15 cubes fell off the edge nearest the robot, during the re-approach after a re-alignment. When the cube is close to the robot, the stand-off point behind it lies near the arm's base, where the inverse kinematics can't lift the gripper while keeping it vertical. The gripper slides at table height toward the robot, and because the stand-off point is recomputed from the moving cube, it keeps pushing the cube until it drops off the edge (see `results/report/fell_off_table_scene35.mp4`). Planned fix: rise to hover height before moving sideways, and freeze the stand-off point during the approach.

![Success by distance](results/report/success_by_distance.png)
![Failure map](results/report/failure_map.png)

## How it works

| Stage | File | What it does |
|---|---|---|
| Scene generation | `scene_gen.py` | Samples cube position/yaw, target, mass and friction from a seeded RNG; rejects invalid scenes; saves parameters to JSON |
| Environment | `env.py` | Attaches the Panda (MuJoCo Menagerie) to the task scene with `MjSpec`; applies scene parameters |
| Controller | `controller.py` | Scripted pusher: damped least-squares inverse kinematics plus a closed-loop plan (approach → descend → push → re-align if the cube drifts) |
| Grader | `grader.py` | Independent of the controller: re-measures after settling, checks table/tilt/time, assigns one outcome per rollout, records metrics |
| Benchmark | `benchmark.py` | Runs all scenes in parallel, saves raw results (JSON/CSV), reports Wilson confidence intervals and per-property breakdowns |
| Report | `report.py` | Charts, example videos per outcome, markdown summary |

**Design choices**
- **Seeded and deterministic:** the same seed and MuJoCo version reproduce identical numbers on different machines.
- **The grader doesn't trust the controller:** the controller stops at 2 cm; the grader uses its own 3 cm rule after physics settles and checks failure modes the controller never looks at.
- **Every result is traceable:** each row stores its scene parameters, so a failure can be traced to the exact scene and replayed (`python run_one.py --scene N`).

**Problems found while building it**
- The Panda's fingers (~4.4 cm) are shorter than a 5 cm cube, so the palm jammed the cube. Switched to a 4 cm cube.
- Changing the cube's mass at runtime made contacts unstable until derived model constants were recomputed with `mj_setConst`.
- An open-loop push drifted off-line; the controller now re-aligns when the cube moves sideways or it overshoots.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh            # add --videos to also render example rollouts
```
Runs in under a minute on a laptop (videos take a few minutes). Watch a single rollout live on macOS with `mjpython run_one.py --scene 19 --watch`.

## Limitations
- Simulation only; no sim-to-real validation.
- One task and one scripted controller; the eval harness is the focus, not the policy.
- 200 scenes give roughly ±6 percentage points of uncertainty on the headline rate.
- Contact parameters are MuJoCo defaults, not calibrated against real hardware.

## Built with
MuJoCo 3.17, [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda model), NumPy, Matplotlib.