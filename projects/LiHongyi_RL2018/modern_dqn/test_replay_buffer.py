
import numpy as np

from replay_buffer import ReplayBuffer


def main():
    # 使用小容量测试，不实际分配 10,000 条经验
    buffer = ReplayBuffer(capacity=5)

    # 添加 5 条模拟经验
    for i in range(5):
        state = np.full(
            (84, 84, 4),
            i / 255.0,
            dtype=np.float32
        )

        next_state = np.full(
            (84, 84, 4),
            (i + 1) / 255.0,
            dtype=np.float32
        )

        buffer.add(
            state=state,
            action=i % 4,
            reward=float(i),
            next_state=next_state,
            terminated=(i == 4),
            truncated=False
        )

    print("Buffer size:", len(buffer))

    # 随机采样 3 条
    batch = buffer.sample(batch_size=3)

    states, actions, rewards, next_states, terminated, truncated = batch

    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    print("Rewards shape:", rewards.shape)
    print("Next states shape:", next_states.shape)

    print("States dtype:", states.dtype)
    print("States range:", states.min(), states.max())

    assert len(buffer) == 5
    assert states.shape == (3, 84, 84, 4)
    assert next_states.shape == (3, 84, 84, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert states.dtype == np.float32
    assert next_states.dtype == np.float32

    # 测试环形覆盖
    extra_state = np.zeros(
        (84, 84, 4),
        dtype=np.float32
    )

    buffer.add(
        extra_state, 0, 0.0,
        extra_state, False, False
    )

    assert len(buffer) == 5

    print("Ring buffer test passed!")
    print("Replay Buffer test passed!")


if __name__ == "__main__":
    main()
