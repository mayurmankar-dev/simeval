#!/usr/bin/env bash
# Reproduce every number in the README from scratch.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d mujoco_menagerie ]; then
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
fi

python scene_gen.py --n 200 --seed 0        # 200 randomized scenes -> scenes_seed0.json
python benchmark.py --scenes scenes_seed0.json   # rollouts + grades -> results/
python report.py --scenes scenes_seed0.json "$@" # charts (+ videos with --videos) -> results/report/
