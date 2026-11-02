"""Benchmark: run the controller on every scene, grade each rollout, and summarise.

    python benchmark.py --scenes scenes_seed0.json
"""
import argparse, csv, json, time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
_env = None   # one simulation per worker process


def _init_worker():
    global _env
    from env import PushEnv
    _env = PushEnv()


def _run_scene(scene):
    from controller import run_push
    from grader import grade
    _env.reset(scene)
    trace, n_realign = run_push(_env)
    return grade(_env, scene, trace, n_realign)


def wilson_ci(k, n, z=1.96):
    """95% confidence interval for a success rate k/n (Wilson score interval)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def rate_line(label, rows):
    k, n = sum(r["success"] for r in rows), len(rows)
    lo, hi = wilson_ci(k, n)
    return f"  {label:28s} {k:3d}/{n:<3d} = {k/n:5.1%}   95% CI [{lo:5.1%}, {hi:5.1%}]"


def breakdown(results, key, label, unit=""):
    """Split scenes into 3 equal-size bins by `key` and show success rate per bin."""
    vals = np.array([r[key] for r in results])
    edges = np.quantile(vals, [0, 1/3, 2/3, 1])
    print(f"\nSuccess by {label}:")
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        rows = [r for r in results if (lo <= r[key] <= hi if i == 2 else lo <= r[key] < hi)]
        print(rate_line(f"{lo:.2f}-{hi:.2f}{unit}", rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scenes_seed0.json")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = p.parse_args()

    scenes = json.loads((ROOT / args.scenes).read_text())
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        results = pool.map(_run_scene, scenes)
    results.sort(key=lambda r: r["scene_id"])
    print(f"Ran {len(results)} rollouts in {time.time() - t0:.0f}s on {args.workers} workers")

    # save raw results: every number in the report can be traced back to these rows
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.scenes).stem
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("\nOverall:")
    print(rate_line("success rate", results))

    print("\nOutcomes:")
    for outcome, n in Counter(r["outcome"] for r in results).most_common():
        print(f"  {outcome:28s} {n:3d}  ({n/len(results):.1%})")

    ok = [r for r in results if r["success"]]
    if ok:
        print("\nAmong successes (median):")
        print(f"  time {np.median([r['duration_s'] for r in ok]):.1f}s, "
              f"re-alignments {np.median([r['n_realign'] for r in ok]):.0f}, "
              f"final distance {np.median([r['final_dist_cm'] for r in ok]):.1f} cm, "
              f"path efficiency {np.median([r['path_efficiency'] for r in ok]):.2f}")

    breakdown(results, "mass", "cube mass", " kg")
    breakdown(results, "friction", "friction")
    breakdown(results, "start_dist_cm", "start distance", " cm")
    print(f"\nRaw results saved to results/{stem}.json and results/{stem}.csv")