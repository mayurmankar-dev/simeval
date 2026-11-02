# SimEval report: `scenes_seed0`

**Task:** Franka Panda pushes a 4 cm cube onto a target. **Scenes:** 200 randomized (cube pose, target, mass 0.05–0.5 kg, friction 0.3–1.2).

**Success rate: 147/200 = 73.5%** (95% CI 67.0%–79.1%). Success = cube centre within 3 cm of target after 0.5 s settling, on the table, tilt < 15°.

| Outcome | Count | Share |
|---|---|---|
| success | 147 | 73.5% |
| timeout_near | 27 | 13.5% |
| fell_off_table | 15 | 7.5% |
| timeout_far | 5 | 2.5% |
| tipped_over | 5 | 2.5% |
| stopped_short | 1 | 0.5% |

![Outcomes](outcomes.png)

![Success by distance](success_by_distance.png)

![Failure map](failure_map.png)

