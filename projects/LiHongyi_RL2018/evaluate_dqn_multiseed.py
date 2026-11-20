"""Evaluate existing 1.75M snapshots using the project's unchanged test protocol."""
import argparse
import csv
import gc
import hashlib
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modern_dqn"))
from environment import Environment
from agent_dqn import Agent_DQN
from q_network import QNetwork

SEEDS = [11037, 2026, 2027, 10007, 27182, 31415, 4242, 7777, 8888, 12345]
STEPS = 1_750_000
EPISODES = 100
EPSILON = 0.025
MODELS = {
    "DQN Baseline": ROOT / "modern_dqn/checkpoints/dqn_baseline_cpu_01_latest_1750000.pt",
    "Double DQN": ROOT / "modern_double_dqn/checkpoints/double_dqn_cpu_01_latest_1750000.pt",
}
FIELDS = ["algorithm", "eval_seed", "episode", "episode_reward", "checkpoint_name",
          "checkpoint_path", "checkpoint_sha256", "training_steps", "test_epsilon"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_digest(network):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                    for v in network.state_dict().values())).hexdigest()


def write_new(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows, hashes):
    groups = {}
    for row in rows:
        algorithm, seed = row["algorithm"], int(row["eval_seed"])
        if (algorithm not in MODELS or seed not in SEEDS
                or row["checkpoint_sha256"] != hashes[algorithm]
                or row["checkpoint_path"] != str(MODELS[algorithm])
                or int(row["training_steps"]) != STEPS
                or float(row["test_epsilon"]) != EPSILON
                or not np.isfinite(float(row["episode_reward"]))):
            raise ValueError("Episode data do not match the selected protocol/checkpoint")
        groups.setdefault((algorithm, seed), []).append(row)
    for key, group in groups.items():
        if [int(row["episode"]) for row in group] != list(range(1, EPISODES + 1)):
            raise ValueError(f"Incomplete/duplicate episode group: {key}")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reuse complete groups from this script")
    options = parser.parse_args()
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    detail = result_dir / "dqn_eval_multiseed_1750000_episodes.csv"
    summary_path = result_dir / "dqn_eval_multiseed_1750000_summary.csv"
    pair_path = result_dir / "dqn_eval_multiseed_1750000_pairwise.csv"
    for path in [summary_path, pair_path]:
        if path.exists():
            raise FileExistsError(path)
    if detail.exists() and not options.resume:
        raise FileExistsError(f"Use --resume for this script's existing episode data: {detail}")

    # CPU execution with one thread avoids small-CNN thread-pool overhead.
    # No training components or replay/optimizer state are instantiated.
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks, hashes, initial_weights = {}, {}, {}
    for algorithm, path in MODELS.items():
        hashes[algorithm] = digest(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["total_steps"]) != STEPS:
            raise ValueError(f"Incorrect checkpoint step count: {path}")
        network = QNetwork(action_dim=4).to(device)
        network.load_state_dict(checkpoint["online"])
        network.eval().requires_grad_(False)
        networks[algorithm] = network
        initial_weights[algorithm] = weight_digest(network)
        del checkpoint
        gc.collect()
        print(f"VERIFIED {algorithm}: steps={STEPS}, device={device}, sha256={hashes[algorithm]}", flush=True)

    rows = []
    if detail.exists():
        with detail.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Unexpected episode CSV schema")
            rows = list(reader)
    else:
        write_new(detail, FIELDS, [])
    completed = validate_rows(rows, hashes)
    started = time.monotonic()
    for seed in SEEDS:
        for algorithm, path in MODELS.items():
            if (algorithm, seed) in completed:
                continue
            env = Environment("BreakoutNoFrameskip-v4", args=None, atari_wrapper=True, test=True)
            try:
                # Reuse the exact original make_action/preprocessing methods.
                agent = Agent_DQN.__new__(Agent_DQN)
                agent.env = env
                agent.action_dim = env.action_space.n
                agent.device = device
                agent.q_network = networks[algorithm]
                agent.epsilon = EPSILON
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                group = []
                with torch.inference_mode():
                    for episode in range(1, EPISODES + 1):
                        state = env.reset()
                        agent.init_game_setting()
                        done, reward_sum = False, 0.0
                        while not done:
                            action = agent.make_action(state, test=True)
                            state, reward, done, info = env.step(action)
                            reward_sum += float(reward)
                        group.append(dict(algorithm=algorithm, eval_seed=seed, episode=episode,
                                          episode_reward=reward_sum, checkpoint_name=path.name,
                                          checkpoint_path=str(path), checkpoint_sha256=hashes[algorithm],
                                          training_steps=STEPS, test_epsilon=EPSILON))
                        if episode % 20 == 0:
                            print(f"PROGRESS seed={seed} {algorithm} episodes={episode}/100 "
                                  f"mean={np.mean([r['episode_reward'] for r in group]):.2f} "
                                  f"elapsed={time.monotonic()-started:.0f}s", flush=True)
                validate_rows(group, hashes)
                # Only persist complete groups; interrupted evaluations can restart their seed.
                buffer = io.StringIO(newline="")
                csv.DictWriter(buffer, fieldnames=FIELDS).writerows(group)
                with detail.open("a", newline="", encoding="utf-8") as handle:
                    handle.write(buffer.getvalue())
                    handle.flush()
                rows.extend(group)
            finally:
                env.close()

    groups = validate_rows(rows, hashes)
    if len(groups) != len(SEEDS) * len(MODELS):
        raise ValueError("Expected twenty complete model/seed groups")
    summary = []
    means = {}
    for seed in SEEDS:
        for algorithm, path in MODELS.items():
            rewards = np.array([float(r["episode_reward"]) for r in groups[(algorithm, seed)]])
            mean = float(rewards.mean())
            means[(algorithm, seed)] = mean
            summary.append(dict(algorithm=algorithm, eval_seed=seed, checkpoint_name=path.name,
                                checkpoint_path=str(path), episodes=len(rewards), mean_reward=mean,
                                std_reward=float(rewards.std(ddof=0)), median_reward=float(np.median(rewards)),
                                min_reward=float(rewards.min()), max_reward=float(rewards.max()),
                                reward_gt_40=mean > 40, checkpoint_sha256=hashes[algorithm],
                                training_steps=STEPS, test_epsilon=EPSILON, device=str(device)))
    pairs = [dict(eval_seed=seed, dqn_mean_reward=means[("DQN Baseline", seed)],
                  double_dqn_mean_reward=means[("Double DQN", seed)],
                  difference_double_minus_dqn=means[("Double DQN", seed)] - means[("DQN Baseline", seed)])
             for seed in SEEDS]
    for algorithm, path in MODELS.items():
        if digest(path) != hashes[algorithm] or weight_digest(networks[algorithm]) != initial_weights[algorithm]:
            raise RuntimeError("Checkpoint or inference weights changed")
    write_new(summary_path, list(summary[0]), summary)
    write_new(pair_path, list(pairs[0]), pairs)
    print("COMPLETE: 10 seeds, 20 groups, 2000 episodes; checkpoint and model hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
