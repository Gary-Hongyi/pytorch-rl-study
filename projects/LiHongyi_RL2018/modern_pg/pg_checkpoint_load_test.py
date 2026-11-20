
from pathlib import Path
import torch
from pg_policy import PongActor

# 1. 找到刚才保存的文件
path = Path(__file__).resolve().parent / "checkpoints" / "pong_pg_latest.pt"

# 2. 从硬盘读取 Checkpoint
checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

# 3. 创建一个新的 Actor
actor = PongActor()

# 4. 把保存的参数加载到新 Actor 中
actor.load_state_dict(checkpoint["model"])

print("模型参数恢复成功！")
print("已训练局数:", checkpoint["episodes_completed"])
print("每局 Reward:", checkpoint["episode_rewards"])
