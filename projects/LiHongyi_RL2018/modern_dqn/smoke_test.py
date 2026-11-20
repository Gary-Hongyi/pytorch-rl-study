"""Bounded HW4-2 environment checks; no model, optimizer, or training loop."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import unittest

import ale_py
import gymnasium as gym
import numpy as np
import torch

from atari_wrapper import ClipRewardEnv, EpisodicLifeEnv, MaxAndSkipEnv, wrap_atari
from environment import ACTION_MEANINGS, Environment, make_breakout_env

ROOT = Path(__file__).resolve().parents[1]
DETAILS = {}


class CompatibleRNG:
    """Same stream for old randint and modern integers in the reference check."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def integers(self, *args):
        return self.rng.integers(*args)

    randint = integers


class ScriptedAtari(gym.Env):
    """Deterministic raw frames, multi-point rewards, and life/game boundaries."""
    metadata = {}

    def __init__(self, truncate_at=None):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
        self.ale = SimpleNamespace(lives=lambda: self.lives)
        self.np_random = CompatibleRNG(42)
        self.actions = []
        self.reset_count = 0
        self.frame = 0
        self.lives = 5
        self.truncate_at = truncate_at
        self.closed = False

    def get_action_meanings(self):
        return list(ACTION_MEANINGS)

    def image(self):
        out = np.full((210, 160, 3), self.frame % 256, np.uint8)
        out[..., 1] = (self.frame * 3) % 256
        return out

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = CompatibleRNG(seed)
        self.reset_count += 1
        self.frame, self.lives = 0, 5
        return self.image(), {"lives": self.lives}

    def step(self, action):
        self.actions.append(int(action))
        self.frame += 1
        if self.frame % 13 == 0:
            self.lives = max(0, self.lives - 1)
        terminated = self.lives == 0
        truncated = self.truncate_at is not None and self.frame >= self.truncate_at
        reward = float((-3, 0, 4)[self.frame % 3])
        return self.image(), reward, terminated, truncated, {"lives": self.lives}

    def close(self):
        self.closed = True


def original_reference(raw, test):
    """Run actual untouched official wrapper code under a minimal old-Gym API.

    Same fake emulator/RNG on both sides isolates port correctness from changes
    to ALE, Gym and historical NumPy RNG implementations.
    """
    class OldWrapper:
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.spec = SimpleNamespace(id="BreakoutNoFrameskip-v4")

        @property
        def unwrapped(self):
            return self.env.unwrapped

        def reset(self, **kwargs):
            if hasattr(self, "_reset"):
                return self._reset(**kwargs)
            return self.env.reset(**kwargs)

        def step(self, action):
            if hasattr(self, "_step"):
                return self._step(action)
            return self.env.step(action)

    class OldObservation(OldWrapper):
        def reset(self, **kwargs):
            return self._observation(self.env.reset(**kwargs))

        def step(self, action):
            obs, r, done, info = self.env.step(action)
            return self._observation(obs), r, done, info

    class OldReward(OldWrapper):
        def step(self, action):
            obs, r, done, info = self.env.step(action)
            return obs, self._reward(r), done, info

    class OldRaw(OldWrapper):
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)[0]

        def step(self, action):
            obs, r, terminated, truncated, info = self.env.step(action)
            return obs, r, terminated or truncated, info

    shim = SimpleNamespace(Wrapper=OldWrapper, ObservationWrapper=OldObservation,
                           RewardWrapper=OldReward, spaces=gym.spaces,
                           make=lambda env_id: OldRaw(raw))
    source = (ROOT / "original-2018/hw4/atari_wrapper.py").read_text(encoding="utf-8")
    source = source.replace("import gym\n", "").replace("from gym import spaces\n", "")
    namespace = {"gym": shim, "spaces": gym.spaces}
    exec(compile(source, "<official-wrapper-in-memory>", "exec"), namespace)
    return namespace["make_wrap_atari"]("BreakoutNoFrameskip-v4", clip_rewards=not test)


class EnvironmentChecks(unittest.TestCase):
    def assert_state(self, env, obs):
        self.assertEqual(obs.shape, (84, 84, 4))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))

    def test_original_wrapper_reference(self):
        # Compare reset/step pixels, reward, done, raw action order and resets
        # against the actual official source in both clipping configurations.
        for test in (False, True):
            old_raw, new_raw = ScriptedAtari(), ScriptedAtari()
            old, new = original_reference(old_raw, test), wrap_atari(new_raw, test=test)
            try:
                np.testing.assert_array_equal(np.array(old.reset()), new.reset()[0])
                for i in range(40):
                    a = i % 4
                    old_obs, old_r, old_done, _ = old.step(a)
                    obs, r, terminated, truncated, _ = new.step(a)
                    np.testing.assert_array_equal(np.array(old_obs), obs)
                    self.assertEqual(old_r, r)
                    self.assertEqual(old_done, terminated or truncated)
                    if old_done:
                        np.testing.assert_array_equal(np.array(old.reset()), new.reset()[0])
                self.assertEqual(old_raw.actions, new_raw.actions)
                self.assertEqual(old_raw.reset_count, new_raw.reset_count)
            finally:
                new.close()
        DETAILS["reference"] = {"passed": True, "steps_per_mode": 40,
                                "modes": ["train", "test"], "real_2018_emulator": False}

    def test_clipping_and_frame_skip(self):
        raw = ScriptedAtari()
        env = ClipRewardEnv(MaxAndSkipEnv(raw))
        try:
            env.reset(seed=42)
            obs, reward, _, _, info = env.step(2)
            # Frames 1..4 yield 0,+4,-3,0 -> sum +1; next 4 -> +5.
            self.assertEqual(raw.actions, [2] * 4)
            _, reward, _, _, info = env.step(2)
            self.assertEqual(info["raw_reward"], 5.0)
            self.assertEqual(reward, 1.0)
            DETAILS["synthetic_clipping"] = {"raw_reward": 5.0, "returned_reward": reward}
        finally:
            env.close()

    def test_life_reset_and_truncation(self):
        raw = ScriptedAtari()
        env = EpisodicLifeEnv(raw)
        try:
            env.reset(seed=42)
            for _ in range(13):
                _, _, terminated, truncated, info = env.step(0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["life_lost"])
            self.assertFalse(info["real_done"])
            resets = raw.reset_count
            env.reset()
            self.assertEqual(raw.reset_count, resets)
        finally:
            env.close()
        raw = ScriptedAtari(truncate_at=1)
        env = EpisodicLifeEnv(raw)
        try:
            env.reset(seed=42)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["real_truncated"])
            env.reset()
            self.assertEqual(raw.reset_count, 2)
        finally:
            env.close()
        self.assertTrue(raw.closed)

    def test_real_ale_train_test_and_reproducibility(self):
        train = make_breakout_env(test=False, render_mode="rgb_array")
        test = make_breakout_env(test=True, render_mode="rgb_array")
        record = {"steps_per_mode": 256, "seed": 42}
        try:
            a, _ = train.reset()
            b, _ = test.reset()
            self.assert_state(train, a)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a[..., 0], a[..., 3])
            self.assertEqual(tuple(train.unwrapped.get_action_meanings()), ACTION_MEANINGS)
            rewards, life_ends, game_ends = set(), 0, 0
            digest = hashlib.sha256(a.tobytes())
            for _ in range(256):
                action = int(train.action_space.sample())
                self.assertEqual(action, int(test.action_space.sample()))
                a, r, term, trunc, info = train.step(action)
                b, raw_r, term_b, trunc_b, info_b = test.step(action)
                self.assert_state(train, a)
                np.testing.assert_array_equal(a, b)
                self.assertEqual((term, trunc), (term_b, trunc_b))
                self.assertIsInstance(term, bool)
                self.assertIsInstance(trunc, bool)
                self.assertEqual(r, float(np.sign(raw_r)))
                self.assertEqual(raw_r, info["raw_reward"])
                self.assertEqual(raw_r, info_b["raw_reward"])
                rewards.add(raw_r)
                digest.update(a.tobytes())
                life_ends += info["life_lost"]
                game_ends += info["real_terminated"]
                if term or trunc:
                    a, _ = train.reset()
                    b, _ = test.reset()
                    np.testing.assert_array_equal(a, b)
            before = a[..., 1:].copy()
            a, _, _, _, _ = train.step(0)
            np.testing.assert_array_equal(a[..., :3], before)
            image = train.render()
            self.assertEqual(image.shape, (210, 160, 3))
            with self.assertRaises(ValueError):
                train.step(4)
            record.update(passed=True, shape=list(a.shape), dtype=str(a.dtype),
                          range=[float(a.min()), float(a.max())], action_meanings=list(ACTION_MEANINGS),
                          observed_raw_rewards=sorted(rewards), life_ends=int(life_ends), game_ends=int(game_ends),
                          trajectory_sha256=digest.hexdigest(), render_shape=list(image.shape),
                          ale_max_frames=train.unwrapped.ale.getInt("max_num_frames_per_episode"),
                          gym_max_episode_steps=train.spec.max_episode_steps)
        finally:
            train.close()
            test.close()
        record["close_completed"] = True
        rom = Path(ale_py.roms.get_rom_path("breakout"))
        record["rom"] = {"path": str(rom), "bytes": rom.stat().st_size,
                         "sha256": hashlib.sha256(rom.read_bytes()).hexdigest()}
        DETAILS["real_ale"] = record

    def test_four_value_bridge_and_torch_input(self):
        env = Environment()
        try:
            env.seed(42)
            obs = env.reset()
            state = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
            self.assertEqual(tuple(state.shape), (1, 4, 84, 84))
            obs, reward, done, info = env.step(env.get_random_action())
            self.assertIsInstance(done, bool)
            self.assertEqual(done, info["terminated"] or info["truncated"])
            self.assertEqual(torch.sum(torch.ones(4)).item(), 4)
            DETAILS["torch_input"] = {"shape": list(state.shape), "dtype": str(state.dtype), "passed": True}
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("smoke_results.json"))
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EnvironmentChecks))
    report = {"python": platform.python_version(), "executable": sys.executable,
              "versions": {n: metadata.version(n) for n in ["torch", "gymnasium", "ale-py", "opencv-python", "numpy"]},
              "tests_run": result.testsRun, "passed": result.wasSuccessful(), "details": DETAILS,
              "failures": [str(t) + "\n" + s for t, s in result.failures + result.errors],
              "training_executed": False, "cuda_available": torch.cuda.is_available()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
