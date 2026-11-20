"""Extract existing local training histories and plot episode reward and Avg30."""
import csv
from pathlib import Path

import numpy as np
import torch
from plot_dqn_evaluations import configure_style, save_figure
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    configure_style()
    experiments = [
        ("modern_pg", "Vanilla PG", "vanilla_pg", "#0F4D92"),
        ("modern_pg_improved", "Actor-Critic (MC value baseline)", "actor_critic", "#B64342"),
    ]
    for directory, label, slug, color in experiments:
        source = ROOT / directory / "checkpoints/pong_pg_latest.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        rewards = np.asarray(checkpoint["episode_rewards"], dtype=float)
        steps = np.asarray(checkpoint["episode_timesteps"], dtype=np.int64)
        count = int(checkpoint["episodes_completed"])
        if (rewards.ndim != 1 or steps.ndim != 1 or len(rewards) != count
                or len(steps) != count or count < 30 or not np.isfinite(rewards).all()
                or steps[0] <= 0 or not np.all(np.diff(steps) > 0)
                or steps[-1] != int(checkpoint["total_steps"])):
            raise ValueError(f"Invalid or incomplete training history in {source}")
        episodes = np.arange(1, count + 1)
        avg30 = np.convolve(rewards, np.ones(30) / 30, mode="valid")
        output = ROOT / "results" / f"pong_{slug}_training_records.csv"
        output.parent.mkdir(exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "episode_reward", "cumulative_environment_steps", "avg30_reward"])
            for index, (episode, reward, step) in enumerate(zip(episodes, rewards, steps)):
                writer.writerow([episode, reward, step, avg30[index - 29] if index >= 29 else ""])
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.plot(episodes, rewards, color=color, alpha=0.22, linewidth=0.55,
                label="Episode reward")
        ax.plot(episodes[29:], avg30, color=color, linewidth=1.8, label="Avg30 (trailing 30 episodes)")
        ax.set(title=f"Pong training | {label}", xlabel="Episode number",
               ylabel="Episode reward", ylim=(-23, 23), xlim=(1, count))
        ax.grid(axis="y", alpha=0.18)
        ax.legend(loc="upper left")
        fig.text(0.5, 0.01, f"{count:,} completed episodes | Training history | Unclipped episode reward",
                 ha="center", fontsize=9)
        save_figure(fig, f"pong_{slug}_training.png")
        print(f"{label}: {count} episodes, {steps[-1]} cumulative steps, final Avg30 {avg30[-1]:.3f}")


if __name__ == "__main__":
    main()
