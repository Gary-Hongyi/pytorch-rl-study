"""Port of original-2018/hw4/atari_wrapper.py to the Gymnasium five-value API.

Wrapper order, life boundaries, FIRE actions and terminal max-pool buffer
semantics follow that file. No dependency on either Pong experiment.
"""
from collections import deque

import cv2
import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        noops = self.override_num_noops
        if noops is None:
            noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Preserve the official buffer behavior even on early terminal steps.
        # Do not replace this with max(last two available frames).
        return self._obs_buffer.max(axis=0), total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = bool(terminated or truncated)
        lives = self.unwrapped.ale.lives()
        life_lost = 0 < lives < self.lives
        info = dict(info, life_lost=bool(life_lost), real_terminated=bool(terminated),
                    real_truncated=bool(truncated), real_done=self.was_real_done)
        self.lives = lives
        return obs, reward, bool(terminated or life_lost), truncated, info

    def reset(self, **kwargs):
        if self.was_real_done or kwargs.get("seed") is not None:
            # An explicit seed starts a new reproducible game. Ordinary life
            # resets below continue the existing game, as in the official code.
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE" and len(meanings) >= 3

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            _, info = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            # Like the official implementation, return the last action's obs.
            _, info = self.env.reset()
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)[..., None]


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (84, 84, 1), np.float32)

    def observation(self, observation):
        return np.asarray(observation).astype(np.float32) / 255.0


class ClipRewardEnv(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(np.sign(reward)), terminated, truncated, dict(info, raw_reward=float(reward))


class RawRewardInfo(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, dict(info, raw_reward=float(reward))


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (84, 84, k), np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        self.frames.extend([obs] * self.k)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(self.frames, axis=2)


def wrap_atari(env, *, test=False):
    """Apply the full official pipeline, including life boundaries at test time."""
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = RawRewardInfo(env) if test else ClipRewardEnv(env)
    return FrameStack(env, 4)
