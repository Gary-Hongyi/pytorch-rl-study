"""Independent HW4-2 Breakout factory and optional official API bridge."""
import ale_py
import gymnasium as gym

from atari_wrapper import wrap_atari

gym.register_envs(ale_py)
DEFAULT_SEED = 42
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")


class SeedOnFirstReset(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._initial_seed = seed
        self.action_space.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self._initial_seed
        self._initial_seed = None
        if seed is not None:
            self.action_space.seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Breakout action: {action!r}")
        return self.env.step(action)


def make_breakout_env(*, test=False, seed=DEFAULT_SEED, render_mode=None):
    """Modern five-value interface; no shortened episode/time-limit overrides."""
    env = gym.make(
        "ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0,
        full_action_space=False, obs_type="rgb", render_mode=render_mode,
    )
    try:
        meanings = tuple(env.unwrapped.get_action_meanings())
        if meanings != ACTION_MEANINGS:
            raise RuntimeError(f"Unexpected action mapping: {meanings}")
        return SeedOnFirstReset(wrap_atari(env, test=test), seed)
    except Exception:
        env.close()
        raise


class Environment:
    """Four-value facade with the official Agent/test interface, seed 42 by default.

    Modern DQN should use make_breakout_env directly to retain both end flags.
    """
    def __init__(self, env_name="BreakoutNoFrameskip-v4", args=None, atari_wrapper=True, test=False):
        if env_name not in ("BreakoutNoFrameskip-v4", "ALE/Breakout-v5") or not atari_wrapper:
            raise ValueError("HW4-2 requires wrapped Breakout; wrapper cannot be disabled")
        self.env = make_breakout_env(test=test, seed=getattr(args, "seed", DEFAULT_SEED))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._pending_seed = None
        self.last_reset_info = {}

    def seed(self, seed):
        self._pending_seed = seed
        self.action_space.seed(seed)

    def reset(self):
        obs, self.last_reset_info = self.env.reset(seed=self._pending_seed)
        self._pending_seed = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, bool(terminated or truncated), dict(info, terminated=bool(terminated), truncated=bool(truncated))

    def get_random_action(self):
        return int(self.action_space.sample())

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def close(self):
        self.env.close()
