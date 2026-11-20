
import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity=10_000,
        observation_shape=(84, 84, 4)
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape

        # 1. 存储 State 和 Next State
        self.states = np.empty(
            (capacity, *observation_shape),
            dtype=np.uint8
        )

        self.next_states = np.empty(
            (capacity, *observation_shape),
            dtype=np.uint8
        )

        # 2. 存储 Action、Reward 和结束标志
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)

        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)

        # 3. 环形缓冲区管理
        self.position = 0
        self.size = 0

    def _encode_observation(self, observation):
        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        if observation.shape != self.observation_shape:
            raise ValueError(
                f"Expected {self.observation_shape}, "
                f"got {observation.shape}"
            )

        return np.rint(
            np.clip(observation, 0.0, 1.0) * 255
        ).astype(np.uint8)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated
    ):
        # 转换图像存储格式
        self.states[self.position] = (
            self._encode_observation(state)
        )

        self.next_states[self.position] = (
            self._encode_observation(next_state)
        )

        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)

        self.terminated[self.position] = bool(terminated)
        self.truncated[self.position] = bool(truncated)

        # 移动写入位置，满了之后覆盖最旧经验
        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity
        )

    def sample(self, batch_size=32):
        if self.size < batch_size: # 这里的self.size 代表的是当前已有的经验值 
            raise ValueError(
                f"Not enough experiences: "
                f"{self.size}/{batch_size}"
            )

        # 从已有经验中随机抽取，不重复
        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=False
        )

        # 恢复图像数据
        states = (
            self.states[indices].astype(np.float32) / 255.0
        )

        next_states = (
            self.next_states[indices].astype(np.float32) / 255.0
        )

        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            next_states,
            self.terminated[indices],
            self.truncated[indices]
        )

    def __len__(self):
        return self.size
