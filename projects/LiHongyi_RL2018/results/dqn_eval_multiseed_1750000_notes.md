# Breakout 1,750,000 Steps 多测试 Seed 评估

## 协议与数据

固定两个已训练 Checkpoint，仅改变 evaluation seed。训练 Seed 42 沿用历史训练记录，未重新训练。两份 Checkpoint 内部 total_steps 均已核实为 1,750,000。所有 20 组结果均为本轮重新评估，未混用旧均值。

使用项目 Environment(BreakoutNoFrameskip-v4, atari_wrapper=True, test=True)，epsilon=0.025，未裁剪奖励。每组完成 100 个包装 Episode，每次评估前重新设置 Python、NumPy、Torch 和环境种子。当前 EpisodicLifeEnv 在丢命时也终止一个 Episode，此口径与历史评估脚本一致，100 个包装 Episode 不等同于 100 次完整多条命游戏。

专用脚本复用原 Agent 的 make_action、观察转换与 init_game_setting。两个算法的推理方法、CNN 和环境封装已确认相同，仅载入各自 Online 权重，未实例化训练优化器或 Replay。CPU 单线程执行以减少小型 CNN 的线程调度开销，torch.inference_mode() 下推理。执行前后模型权重与原始 Checkpoint 的哈希一致。

## 逐 Seed 结果

| Evaluation seed | DQN Mean100 | Double DQN Mean100 | Double−DQN |
| ---: | ---: | ---: | ---: |
| 11037 | 38.24 | 42.93 | +4.69 |
| 2026 | 39.63 | 38.88 | -0.75 |
| 2027 | 40.54 | 35.20 | -5.34 |
| 10007 | 41.36 | 42.44 | +1.08 |
| 27182 | 39.24 | 34.04 | -5.20 |
| 31415 | 37.03 | 38.88 | +1.85 |
| 4242 | 39.20 | 42.09 | +2.89 |
| 7777 | 36.68 | 38.00 | +1.32 |
| 8888 | 36.72 | 46.17 | +9.45 |
| 12345 | 42.06 | 35.42 | -6.64 |

## 跨 Seed 汇总

| 指标 | DQN Baseline | Double DQN |
| --- | ---: | ---: |
| Mean100 的跨 seed 平均 | 39.0700 | 39.4050 |
| Mean100 的跨 seed 标准差（ddof=0） | 1.8108 | 3.7347 |
| Mean100 最小值 | 36.68 | 34.04 |
| Mean100 最大值 | 42.06 | 46.17 |
| Mean100 > 40 的 seed 数 | 3 | 4 |

Double DQN 领先 6 个 seeds，DQN 领先 4 个，持平 0 个。配对差值均值为 +0.3350，总体标准差为 4.7441，中位数为 +1.2000，范围为 [-6.64, +9.45]。

summary.csv 的 std_reward 表示单个模型、单个 seed 下 100 局 Reward 的总体标准差。上表及配对差值标准差则描述 10 个 seed 级别均值或差值的离散程度，两种统计层级分别解释。reward_gt_40 表示该组 Mean100 是否严格大于 40。

## 汇报意义与边界

本实验研究固定训练结果的 evaluation seed sensitivity。图表展示测试随机种子如何影响 Mean100 和同 seed 下的算法排名，可用于补充单测试 seed 结论的适用范围。

这是每种算法各一份训练 Seed 42 Checkpoint 的测试，未估计多 training seeds 的训练稳定性。跨测试 seed 的离散程度包含有限 100 局采样和测试轨迹随机性的影响，不能据此证明算法普遍更稳定或建立统计显著性。相同 seed 统一随机数初始化，但两策略动作会导致环境轨迹分化，并非逐局完全相同的测试样本。

与旧表核对时，本轮 Baseline 在 Seed 11037 的 Mean100 为 38.24，旧归档为 31.32。旧记录未包含逐局数据及 Checkpoint 哈希，差异来源待核实；不能直接将两次结果当作同一批样本。本轮使用已核实步数及哈希的两个文件重新完成全部 seed，统计不混入旧结果。CPU 线程数等数值运行条件已在上文记录，未将其认定为差异原因。

## 文件与核验

- [逐局 Reward 明细](dqn_eval_multiseed_1750000_episodes.csv)，2,000 行。
- [模型与 Seed 汇总](dqn_eval_multiseed_1750000_summary.csv)，20 行。
- [配对比较表](dqn_eval_multiseed_1750000_pairwise.csv)，10 行。
- [Mean100 比较图](../figures/dqn_eval_multiseed_mean100_1750000.png)。
- [配对差值图](../figures/dqn_eval_multiseed_difference_1750000.png)。
- [Seed 级均值分布图](../figures/dqn_eval_multiseed_distribution_1750000.png)。

所有统计从逐局 CSV 独立复算，检查 10 个指定 seeds、每组 1–100 连续编号、20 组汇总、配对差值及阈值标志一致性。统计图为英文、300 DPI。CSV 中保留 Checkpoint 绝对路径、SHA-256、训练步数和测试 epsilon。

- DQN Baseline Checkpoint `projects\LiHongyi_RL2018\modern_dqn\checkpoints\dqn_baseline_cpu_01_latest_1750000.pt`，SHA-256 `1a4899e3c1a27b2db71823b60f649135f4b44159b645041dc6c18baf109e1828`。
- Double DQN Checkpoint `projects\LiHongyi_RL2018\modern_double_dqn\checkpoints\double_dqn_cpu_01_latest_1750000.pt`，SHA-256 `44358f250f68ba43dd2c6de34e9cc6e38c166e93c01408bd17d5559a4132e324`。

绘图环境 Python 3.11.17，NumPy 2.4.6，Matplotlib 3.11.2。
