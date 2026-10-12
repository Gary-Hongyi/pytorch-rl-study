# Policy Gradient / REINFORCE

理解整条轨迹如何形成策略梯度更新，再把更新放进重复训练流程。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [pg_cartpole.py](experiments/pg_cartpole.py) | 4→32→2 actor、Categorical 采样、折扣回报、单回合一次更新 |
| 2 | [pg_cartpole_multiple_episode.py](experiments/pg_cartpole_multiple_episode.py) | 重复 1,000 个回合，每回合更新并展示奖励曲线 |

`multiple_episode` 表示重复回合训练，代码不是收集多个回合后统一更新。两个文件保留 actor 和 `compute_returns` 的重复定义，便于独立对照。

单回合脚本中的 `states`、`actions` 记录没有用于最终 loss，且缺少显式 `env.close()`；这些细节保持原样，作为后续复盘内容。当前未发现保存的 RL 曲线、模型或训练日志，不能据此宣称已经稳定解决 CartPole。
