
"""HW4-2 DQN Agent for Breakout.

Includes:
- Online / Target Q-Network
- Epsilon-greedy action selection
- Replay Buffer
- TD Learning
- Target Network synchronization
- Training loop
- Checkpoint save / load
- CSV training logs
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .q_network import QNetwork
    from .replay_buffer import ReplayBuffer
except ImportError:
    from q_network import QNetwork
    from replay_buffer import ReplayBuffer


class Agent_DQN:

    def __init__(self, env, args):
        self.env = env
        self.args = args

        # ===== 1. Environment and device =====

        self.action_dim = self.env.action_space.n

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== 2. Online Network =====

        self.q_network = QNetwork(
            action_dim=self.action_dim
        ).to(self.device)

        # ===== 3. Target Network =====

        self.target_network = QNetwork(
            action_dim=self.action_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        for param in self.target_network.parameters():
            param.requires_grad = False

        # ===== 4. Optimizer =====

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=1.5e-4
        )

        # ===== 5. Hyperparameters =====

        self.gamma = float(
            getattr(args, "gamma", 0.99)
        )

        self.batch_size = 32

        self.update_interval = 4
        self.target_update_interval = 1000

        # ===== 6. Epsilon-greedy =====

        self.epsilon_start = 1.0
        self.epsilon_end = 0.025
        self.epsilon = self.epsilon_start

        self.epsilon_decay_steps = int(
            getattr(args, "epsilon_decay_steps", 1_000_000)
        )

        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")

        # ===== 7. Replay Buffer =====

        self.replay_buffer = ReplayBuffer(
            capacity=10_000,
            observation_shape=self.env.observation_space.shape
        )

        # ===== 8. Training configuration =====

        self.train_steps = int(
            getattr(args, "train_steps", 100)
        )

        self.total_steps = 0

        # ===== 9. Experiment records =====

        self.episode_rewards = []
        self.episode_timesteps = []

        # ===== 10. Checkpoint and logging =====

        self.run_name = str(
            getattr(args, "run_name", "dqn_baseline")
        )

        self.checkpoint_interval = int(
            getattr(args, "checkpoint_interval", 10_000)
        )

        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

        self.log_interval = int(
            getattr(args, "log_interval", 25)
        )

        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")

        root = Path(__file__).resolve().parent

        self.checkpoint_path = (
            root / "checkpoints" / f"{self.run_name}_latest.pt"
        )

        self.log_path = (
            root / "logs" / f"{self.run_name}.csv"
        )

    # ==========================================
    # Observation preprocessing
    # ==========================================

    def _observation_to_tensor(self, observation):
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        )

        # HWC -> CHW -> BCHW
        state = state.permute(2, 0, 1)
        state = state.unsqueeze(0)

        return state.contiguous()

    # ==========================================
    # Official Agent interface
    # ==========================================

    def init_game_setting(self):
        # No additional per-episode Agent state
        # currently needs resetting.
        pass

    def make_action(self, observation, test=True):
        if test:
            self.q_network.eval()
        else:
            self.q_network.train()

        # Exploration
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)

        # Exploitation
        else:
            state = self._observation_to_tensor(observation)

            with torch.no_grad():
                q_values = self.q_network(state)

            action = q_values.argmax(dim=1).item()

        return int(action)

    # ==========================================
    # Replay Buffer
    # ==========================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated
        )

    def _sample_batch_to_tensors(self):
        batch = self.replay_buffer.sample(self.batch_size)

        (
            states, actions, rewards,
            next_states, terminateds, truncateds
        ) = batch

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        ).permute(0, 3, 1, 2).contiguous()

        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        ).permute(0, 3, 1, 2).contiguous()

        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )

        terminateds = torch.as_tensor(
            terminateds, dtype=torch.bool, device=self.device
        )

        truncateds = torch.as_tensor(
            truncateds, dtype=torch.bool, device=self.device
        )

        return (
            states, actions, rewards,
            next_states, terminateds, truncateds
        )

    # ==========================================
    # TD Learning
    # ==========================================

    def _compute_current_q_values(self, states, actions):
        q_values = self.q_network(states)

        return q_values.gather(
            dim=1,
            index=actions
        )

    def _compute_td_targets(
        self,
        rewards,
        next_states,
        terminateds
    ):
        with torch.no_grad():
            next_q_values = self.target_network(next_states)

            max_next_q_values = (
                next_q_values.max(dim=1).values
            )

            bootstrap_mask = (~terminateds).float()

            td_targets = (
                rewards
                + self.gamma * bootstrap_mask * max_next_q_values
            )

        return td_targets

    def _learn_from_replay(self):
        (
            states, actions, rewards,
            next_states, terminateds, truncateds
        ) = self._sample_batch_to_tensors()

        self.q_network.train()

        current_q = self._compute_current_q_values(
            states, actions
        ).squeeze(1)

        td_targets = self._compute_td_targets(
            rewards, next_states, terminateds
        )

        loss = torch.nn.functional.mse_loss(
            current_q, td_targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _sync_target_network(self):
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    # ==========================================
    # CSV logging
    # ==========================================

    def _log_episode(self, last_loss):
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        episode = len(self.episode_rewards)
        reward = self.episode_rewards[-1]

        avg30 = (
            sum(self.episode_rewards[-30:]) / 30
            if episode >= 30
            else ""
        )

        # A new experiment starts a new log.
        # Resumed experiments append to existing logs.
        mode = "w" if episode == 1 else "a"

        new_file = (
            mode == "w"
            or not self.log_path.exists()
            or self.log_path.stat().st_size == 0
        )

        with open(
            self.log_path,
            mode,
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(f)

            if new_file:
                writer.writerow([
                    "episode",
                    "total_steps",
                    "reward",
                    "avg30",
                    "epsilon",
                    "last_loss"
                ])

            writer.writerow([
                episode,
                self.episode_timesteps[-1],
                reward,
                avg30,
                self.epsilon,
                last_loss if last_loss is not None else ""
            ])

    # ==========================================
    # Main training loop
    # ==========================================

    def train(self):
        state, _ = self.env.reset()

        episode_reward = 0.0
        last_loss = None

        for _ in range(self.train_steps):

            # 1. Choose action
            action = self.make_action(state, test=False)

            # 2. Environment interaction
            next_state, reward, terminated, truncated, info = (
                self.env.step(action)
            )

            # 3. Store transition
            self.store_transition(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated
            )

            self.total_steps += 1

            # 4. Linear epsilon decay
            fraction = min(
                self.total_steps / self.epsilon_decay_steps,
                1.0
            )

            self.epsilon = (
                self.epsilon_start
                + fraction * (
                    self.epsilon_end - self.epsilon_start
                )
            )

            # 5. Update Online Network
            if (
                self.total_steps % self.update_interval == 0
                and len(self.replay_buffer) >= self.batch_size
            ):
                last_loss = self._learn_from_replay()

            # 6. Synchronize Target Network
            if self.total_steps % self.target_update_interval == 0:
                self._sync_target_network()

            # 7. Accumulate unclipped reward
            episode_reward += float(
                info.get("raw_reward", reward)
            )

            # 8. Handle episode boundary
            if terminated or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.total_steps)

                # NEW: write one CSV record
                self._log_episode(last_loss)

                state, _ = self.env.reset()
                episode_reward = 0.0
            else:
                state = next_state

            # 9. Print progress
            if self.total_steps % self.log_interval == 0:
                print(
                    f"Step={self.total_steps}, "
                    f"Buffer={len(self.replay_buffer)}, "
                    f"Epsilon={self.epsilon:.4f}, "
                    f"Loss={last_loss}"
                )

            # 10. NEW: periodic checkpoint
            if self.total_steps % self.checkpoint_interval == 0:
                self.save_checkpoint(self.checkpoint_path)

        # 11. NEW: final checkpoint
        if self.total_steps % self.checkpoint_interval != 0:
            self.save_checkpoint(self.checkpoint_path)

        print("Training loop finished!")

    # ==========================================
    # Checkpoint save / load
    # ==========================================

    def save_checkpoint(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        buffer = self.replay_buffer
        n = len(buffer)

        checkpoint = {
            "online": self.q_network.state_dict(),
            "target": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),

            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_timesteps": self.episode_timesteps,

            "replay": {
                "states": buffer.states[:n].copy(),
                "next_states": buffer.next_states[:n].copy(),
                "actions": buffer.actions[:n].copy(),
                "rewards": buffer.rewards[:n].copy(),
                "terminated": buffer.terminated[:n].copy(),
                "truncated": buffer.truncated[:n].copy(),
                "size": buffer.size,
                "position": buffer.position,
                "capacity": buffer.capacity,
            },

            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

        # Write to a temporary file first.
        # Replace the checkpoint only after saving succeeds.
        temporary_path = path.with_suffix(".tmp")

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filepath):
        # Only load checkpoints created by yourself
        # or from other trusted sources.
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
            weights_only=False
        )

        self.q_network.load_state_dict(
            checkpoint["online"]
        )

        self.target_network.load_state_dict(
            checkpoint["target"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.target_network.eval()

        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]

        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_timesteps = checkpoint["episode_timesteps"]

        # Restore Replay Buffer
        replay = checkpoint["replay"]
        buffer = self.replay_buffer

        if replay["capacity"] != buffer.capacity:
            raise ValueError("Replay Buffer capacity mismatch")

        n = replay["size"]

        for field in (
            "states",
            "next_states",
            "actions",
            "rewards",
            "terminated",
            "truncated"
        ):
            getattr(buffer, field)[:n] = replay[field]

        buffer.size = n
        buffer.position = replay["position"]

        # Restore RNG states
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])

        print(f"Checkpoint loaded: {filepath}")
