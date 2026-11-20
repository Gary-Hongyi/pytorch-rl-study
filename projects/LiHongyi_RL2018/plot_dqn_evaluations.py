"""Plot historical independent evaluations from the curated CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {"DQN Baseline": "#0F4D92", "Double DQN": "#B64342"}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.linewidth": 1.2,
        "savefig.dpi": 300,
    })


def save_figure(fig, name):
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(directory / name, facecolor="white")
    plt.close(fig)


def main():
    configure_style()
    with (ROOT / "results/dqn_evaluation_records.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    conditions = "Training seed 42 | Test epsilon 0.025 | 100 episodes | Unclipped reward"
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for algorithm, color in COLORS.items():
        rows = sorted((r for r in records if r["algorithm"] == algorithm
                       and r["test_seed"] == "11037"), key=lambda r: int(r["training_steps"]))
        ax.plot([int(r["training_steps"]) for r in rows],
                [float(r["mean100_reward"]) for r in rows],
                marker="o", linewidth=2, color=color, label=algorithm)
    ax.axhline(40, color="#767676", linestyle="--", label="Target reward = 40")
    ax.text(0.99, 0.04, "Double DQN at 2,000,000 steps: no evaluation available",
            transform=ax.transAxes, ha="right", fontsize=9)
    ax.set(title="Breakout independent evaluation | Test seed 11037",
           xlabel="Training steps", ylabel="Mean episode reward (Mean100)", ylim=(0, 50))
    ticks = [500000, 1000000, 1250000, 1500000, 1750000, 2000000]
    ax.set_xticks(ticks, [f"{t / 1e6:g}M" for t in ticks])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left")
    fig.text(0.5, 0.01, conditions, ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.18)
    save_figure(fig, "dqn_eval_mean100_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5.4))
    seeds = [11037, 2026]
    x = np.arange(len(seeds))
    for i, (algorithm, color) in enumerate(COLORS.items()):
        values = [float(next(r["mean100_reward"] for r in records
                      if r["algorithm"] == algorithm and int(r["test_seed"]) == seed
                      and int(r["training_steps"]) == 1750000)) for seed in seeds]
        bars = ax.bar(x + (i - 0.5) * 0.34, values, width=0.34,
                      color=color, edgecolor="black", linewidth=0.7,
                      hatch="" if i == 0 else "//", label=algorithm)
        ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.axhline(40, color="#767676", linestyle="--", label="Target reward = 40")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set(title="Breakout test-seed comparison | 1,750,000 training steps",
           xlabel="Test seed", ylabel="Mean episode reward (Mean100)", ylim=(0, 55))
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.5, 0.01, conditions, ha="center", fontsize=9)
    save_figure(fig, "dqn_eval_seed_robustness.png")


if __name__ == "__main__":
    main()
