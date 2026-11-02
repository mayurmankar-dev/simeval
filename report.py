"""Eval report: charts + failure videos + a markdown summary, from benchmark results.

    python report.py --scenes scenes_seed0.json            # charts + report.md
    python report.py --scenes scenes_seed0.json --videos   # also record failure videos
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark import wilson_ci

ROOT = Path(__file__).parent
# chart styling: light surface, quiet axes, one hue per job
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def chart_success_by_distance(results, out):
    d = np.array([r["start_dist_cm"] for r in results])
    edges = np.quantile(d, np.linspace(0, 1, 6))          # 5 equal-size bins
    labels, rates, lo_err, hi_err, ns = [], [], [], [], []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        rows = [r for r in results if (lo <= r["start_dist_cm"] <= hi if i == 4 else lo <= r["start_dist_cm"] < hi)]
        k, n = sum(r["success"] for r in rows), len(rows)
        p = k / n; ci = wilson_ci(k, n)
        labels.append(f"{lo:.0f}–{hi:.0f}"); rates.append(p * 100)
        lo_err.append((p - ci[0]) * 100); hi_err.append((ci[1] - p) * 100); ns.append(n)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(5)
    ax.bar(x, rates, width=0.6, color=BLUE, zorder=2)
    ax.errorbar(x, rates, yerr=[lo_err, hi_err], fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=3)
    for xi, r, n in zip(x, rates, ns):
        ax.text(xi, 2, f"{r:.0f}%\nn={n}", ha="center", va="bottom", color="white", fontsize=9, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Start distance, cube to target (cm)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title("Success rate by push distance (error bars: 95% CI)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_outcomes(results, out):
    counts = Counter(r["outcome"] for r in results).most_common()
    names = [c[0].replace("_", " ") for c in counts][::-1]
    vals = [c[1] for c in counts][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(names, vals, height=0.6, color=BLUE, zorder=2)
    for i, v in enumerate(vals):
        share = v / len(results)
        ax.text(v + 1.5, i, f"{v}  ({share:.0%})" if share >= 0.01 else f"{v}  (<1%)", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("Rollouts")
    ax.set_title(f"Outcomes across {len(results)} scenes", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def chart_failure_map(results, scenes, out):
    """Top-down view of the table: where each target was, coloured by outcome group."""
    by_id = {s["scene_id"]: s for s in scenes}
    groups = {
        "success": ("Success", BLUE, "o"),
        "timeout": ("Timeout", AQUA, "s"),
        "lost": ("Fell off / tipped over", ORANGE, "^"),
    }
    def group(o):
        return "success" if o == "success" else "lost" if o in ("fell_off_table", "tipped_over") else "timeout"
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_patch(plt.Rectangle((0.2, -0.4), 0.6, 0.8, fill=False, edgecolor=INK_2, linewidth=1.2))
    ax.text(0.21, 0.37, "table", color=INK_2, fontsize=9)
    ax.plot(0, 0, marker="P", markersize=11, color=INK, linestyle="none")
    ax.text(0.02, 0.02, "robot base", color=INK, fontsize=9)
    for key, (label, color, marker) in groups.items():
        pts = np.array([[by_id[r["scene_id"]]["target_x"], by_id[r["scene_id"]]["target_y"]]
                        for r in results if group(r["outcome"]) == key])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], marker=marker, markersize=7, linestyle="none", color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, alpha=0.9, label=f"{label} ({len(pts)})")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("x (m)  →  away from robot"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Target positions by outcome (top-down view)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def record_video(scene, path, fps=30):
    import imageio.v2 as imageio
    import mujoco
    from env import PushEnv
    from controller import run_push
    env = PushEnv()
    env.reset(scene)
    renderer = mujoco.Renderer(env.model, 480, 640)
    frames, every = [], int(round(1 / (fps * env.model.opt.timestep)))
    step = [0]
    def on_step():
        step[0] += 1
        if step[0] % every == 0:
            renderer.update_scene(env.data, camera="overview")
            frames.append(renderer.render())
    run_push(env, on_step=on_step)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scenes_seed0.json")
    p.add_argument("--videos", action="store_true", help="record example rollouts per outcome")
    p.add_argument("--per-outcome", type=int, default=2)
    args = p.parse_args()

    stem = Path(args.scenes).stem
    scenes = json.loads((ROOT / args.scenes).read_text())
    results = json.loads((ROOT / "results" / f"{stem}.json").read_text())
    out_dir = ROOT / "results" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    chart_success_by_distance(results, out_dir / "success_by_distance.png")
    chart_outcomes(results, out_dir / "outcomes.png")
    chart_failure_map(results, scenes, out_dir / "failure_map.png")
    print("Charts saved to results/report/")

    videos = []
    if args.videos:
        by_id = {s["scene_id"]: s for s in scenes}
        for outcome in [o for o, _ in Counter(r["outcome"] for r in results).most_common()]:
            for r in [r for r in results if r["outcome"] == outcome][:args.per_outcome]:
                name = f"{outcome}_scene{r['scene_id']}.mp4"
                record_video(by_id[r["scene_id"]], out_dir / name)
                videos.append((name, r))
                print("  recorded", name)

    k, n = sum(r["success"] for r in results), len(results)
    lo, hi = wilson_ci(k, n)
    lines = [
        f"# SimEval report: `{stem}`", "",
        f"**Task:** Franka Panda pushes a 4 cm cube onto a target. **Scenes:** {n} randomized "
        "(cube pose, target, mass 0.05–0.5 kg, friction 0.3–1.2).", "",
        f"**Success rate: {k}/{n} = {k/n:.1%}** (95% CI {lo:.1%}–{hi:.1%}). "
        "Success = cube centre within 3 cm of target after 0.5 s settling, on the table, tilt < 15°.", "",
        "| Outcome | Count | Share |", "|---|---|---|",
    ]
    for o, c in Counter(r["outcome"] for r in results).most_common():
        lines.append(f"| {o} | {c} | {c/n:.1%} |")
    lines += ["", "![Outcomes](outcomes.png)", "", "![Success by distance](success_by_distance.png)", "",
              "![Failure map](failure_map.png)", ""]
    if videos:
        lines += ["## Example rollouts", ""]
        for name, r in videos:
            lines.append(f"- [{name}]({name}): final distance {r['final_dist_cm']} cm, "
                         f"{r['n_realign']} re-alignments, mass {r['mass']:.2f} kg, friction {r['friction']:.2f}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("Report written to results/report/report.md")