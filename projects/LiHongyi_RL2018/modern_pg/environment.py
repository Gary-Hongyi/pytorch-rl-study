"""阶段 1：现代 Atari 环境；不包含 PG / DQN 算法。"""
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from atari_core import NoopReset, MaxAndSkip, EpisodicLife

gym.register_envs(ale_py)


class FireReset(gym.Wrapper):
    """对应官方 FireResetEnv，在 reset 后发球。"""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for action in (1, 2):
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class DQNFrames(gym.Wrapper):
    """环境处理：灰度、84x84、/255、HWC 四帧；不是 DQN 实现。"""
    def __init__(self, env, clip_rewards):
        super().__init__(env)
        self.frames = deque(maxlen=4)
        self.clip_rewards = clip_rewards
        self.observation_space = gym.spaces.Box(0, 1, (84, 84, 4), np.float32)

    @staticmethod
    def process(obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        frame = self.process(obs)
        self.frames.extend([frame] * 4)
        return np.stack(self.frames, axis=-1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self.process(obs))
        info = dict(info, raw_reward=float(reward))
        if self.clip_rewards:
            reward = float(np.sign(reward))
        return np.stack(self.frames, axis=-1), reward, terminated, truncated, info


def make_env(game, *, wrapped=False, test=True, render_mode=None, max_episode_steps=27000):
    """显式设置 frame skip 和 sticky action；不宣称与旧评分等价。"""
    if game not in ("Pong", "Breakout"):
        raise ValueError("Only Pong and Breakout are supported")
    if wrapped and game != "Breakout":
        raise ValueError("The homework wrapper is only used for Breakout")
    env = gym.make(
        f"ALE/{game}-v5", frameskip=1 if wrapped else (2, 5),
        repeat_action_probability=0.0, full_action_space=False,
        render_mode=render_mode, max_episode_steps=max_episode_steps,
    )
    if wrapped:
        env = EpisodicLife(MaxAndSkip(NoopReset(env)))
        env = FireReset(env)
        env = DQNFrames(env, clip_rewards=not test)
    return env


class Environment:
    """官方 Agent 所需四返回值桥接；现代训练应直接使用 make_env。"""
    def __init__(self, env_name, args=None, atari_wrapper=False, test=False):
        mapping = {"Pong-v0": "Pong", "ALE/Pong-v5": "Pong",
                   "BreakoutNoFrameskip-v4": "Breakout", "ALE/Breakout-v5": "Breakout"}
        self.env = make_env(mapping[env_name], wrapped=atari_wrapper, test=test)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._seed = None
        self.last_reset_info = {}

    def seed(self, seed):
        self._seed = seed
        self.action_space.seed(seed)

    def reset(self):
        obs, self.last_reset_info = self.env.reset(seed=self._seed)
        self._seed = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, terminated=bool(terminated), truncated=bool(truncated))
        return obs, reward, bool(terminated or truncated), info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()

    def close(self):
        self.env.close()


def preprocess_pg_rgb(observation, image_size=(80, 80)):
    """明确灰度输出，替代官方已删除的 scipy.misc.imresize；不含策略。"""
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, tuple(reversed(image_size)), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None]
