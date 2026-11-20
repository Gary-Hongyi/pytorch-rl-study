"""Compare archived Pong histories descriptively without running an environment."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_history(slug):
    path = ROOT / "results" / f"pong_{slug}_training_records.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = np.asarray([int(row["episode"]) for row in rows])
    steps = np.asarray([int(row["cumulative_environment_steps"]) for row in rows])
    rewards = np.asarray([float(row["episode_reward"]) for row in rows])
    if (len(rows) < 30 or not np.array_equal(episodes, np.arange(1, len(rows) + 1))
            or not np.isfinite(rewards).all() or steps[0] <= 0
            or not np.all(np.diff(steps) > 0)):
        raise ValueError(f"Invalid training history in {path}")
    avg30 = np.convolve(rewards, np.ones(30) / 30, mode="valid")
    saved_avg30 = np.asarray([float(row["avg30_reward"]) for row in rows[29:]])
    if not np.allclose(avg30, saved_avg30, rtol=0, atol=1e-9):
        raise ValueError(f"Avg30 does not match episode rewards in {path}")
    return episodes[29:], steps[29:], avg30


def main():
    output = ROOT / "figures/pong_pg_vs_actor_critic_descriptive.png"
    if output.exists():
        raise FileExistsError(f"Preserving existing figure: {output}")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for slug, label, color in [
        ("vanilla_pg", "Vanilla PG", "#0F4D92"),
        ("actor_critic", "Actor-Critic (MC value baseline)", "#B64342"),
    ]:
        episodes, steps, avg30 = load_history(slug)
        axes[0].plot(episodes, avg30, color=color, linewidth=1.3, label=label)
        axes[1].plot(steps, avg30, color=color, linewidth=1.3, label=label)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_xlim(left=0)
        ax.set_ylim(-23, 7)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set(xlabel="Episode number", ylabel="Training episode reward (Avg30)",
                title="Episode-indexed histories")
    axes[1].set(xlabel="Cumulative environment steps", title="Interaction-indexed histories")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    fig.suptitle("Pong learning histories | Descriptive comparison", fontsize=15, y=0.97)
    fig.text(0.5, 0.89, "Historical environment configurations not fully verified",
             ha="center", fontsize=11, color="#87412A")
    fig.text(0.5, 0.025,
             "Trailing 30 episodes | Unclipped reward | Full recorded histories | Different training budgets",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.16, wspace=0.13)
    output.parent.mkdir(exist_ok=True)
    with output.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=300, facecolor="white", metadata={
            "Description": "Descriptive training Avg30 from results/pong_vanilla_pg_training_records.csv "
                           "and results/pong_actor_critic_training_records.csv. "
                           "Historical environment configurations not fully verified.",
        })
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
