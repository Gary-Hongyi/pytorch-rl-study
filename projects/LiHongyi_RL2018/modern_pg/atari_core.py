"""官方 Atari 包装流程的 Gymnasium 适配，仅实现环境操作。"""
import gymnasium as gym
import numpy as np


class NoopReset(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(int(self.unwrapped.np_random.integers(1, 31))):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class MaxAndSkip(gym.Wrapper):
    def step(self, action):
        frames, total = [], 0.0
        for _ in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            frames.append(obs)
            total += reward
            if terminated or truncated:
                break
        return np.maximum(frames[-1], frames[-2] if len(frames) > 1 else frames[-1]), total, terminated, truncated, info


class EpisodicLife(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_done = terminated or truncated
        lives = self.unwrapped.ale.lives()
        lost_life = 0 < lives < self.lives
        self.lives = lives
        return obs, reward, bool(terminated or lost_life), truncated, dict(info, life_lost=lost_life, real_done=self.real_done)

    def reset(self, **kwargs):
        if self.real_done or kwargs.get("seed") is not None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        self.lives = self.unwrapped.ale.lives()
        return obs, info
