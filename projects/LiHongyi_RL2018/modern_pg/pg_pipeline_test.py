
import gymnasium as gym
import ale_py
import torch
import numpy as np

# 这里把modern里面写的网络和预处理的模块导入进来使用
from pg_preprocess import preprocess_frame, compute_frame_difference
from pg_policy import PongActor


# 创建 Pong 环境
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")

# 创建 Actor
actor = PongActor()

# 【改动1】内部动作编号 → Pong 环境动作编号
ACTION_MAP = [2, 3]

# 检查 Pong 的实际动作定义
action_meanings = env.unwrapped.get_action_meanings()
print("Pong action meanings:", action_meanings)

assert action_meanings[2:4] == ["RIGHT", "LEFT"]


observation, info = env.reset()
previous_frame = None

try:
    for step in range(10):

        # 1. 图像预处理
        current_frame = preprocess_frame(observation)

        # 2. 计算帧差
        frame_diff = compute_frame_difference(
            current_frame,
            previous_frame
        )

        # 3. 转换成 PyTorch Tensor
        state = torch.from_numpy(frame_diff)

        # 4. 输入 Actor，得到 Logits
        with torch.no_grad():
            logits = actor(state)

        # 【改动2】根据 Actor 的概率分布采样动作
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()

        # 【改动3】将内部动作映射为 Pong 动作
        action = ACTION_MAP[action_index.item()]

        # 检查动作是否合法
        assert env.action_space.contains(action)

        print(
            f"Step {step}:",
            "Frame:", frame_diff.shape,
            "Changed pixels:", np.count_nonzero(frame_diff),
            "Logits:", logits.tolist(),
            "Actor index:", action_index.item(),
            "Pong action:", action
        )

        # 5. 更新上一帧
        previous_frame = current_frame

        # 【改动4】环境执行 Actor 的动作，而不是随机动作
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            previous_frame = None

finally:
    env.close()
