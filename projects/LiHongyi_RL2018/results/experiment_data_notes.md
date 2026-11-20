# 实验数据与图表说明

## Breakout 独立评估

- `figures/dqn_eval_mean100_comparison.png` 读取 `results/dqn_evaluation_records.csv` 中测试 Seed 11037 的记录。横轴为真实训练步数，纵轴为每组 100 局未裁剪 Episode Reward 的算术平均值 Mean100。
- `figures/dqn_eval_seed_robustness.png` 读取同一 CSV 中 1,750,000 训练步的记录，横轴为测试 Seed 11037 和 2026，柱颜色区分算法，纵轴为独立评估 Mean100。
- CSV 的 13 条记录全部来自用户确认的历史评估结果。统一训练 Seed 42、测试 epsilon 0.025、每组 100 Episodes、未裁剪奖励。未从训练日志推算评估成绩。
- Reward 40 是目标参考线。Double DQN 在 2,000,000 步无数据，CSV 未添加该条记录，折线止于 1,750,000 步。
- 只有各组均值，没有逐局评估奖励、标准差或多训练种子重复实验。图中未添加置信区间、显著性检验，也不据此判断算法必然更优。两组测试种子仅作描述性比较。
- 重现命令为 `python -B plot_dqn_evaluations.py`，可从任意工作目录调用脚本。

## Pong 训练记录盘点

| 算法 | 实际来源 | 已完成局数 | 累计环境步数 | 最后 Avg30 |
| --- | --- | --- | --- | --- |
| Vanilla PG | `modern_pg/checkpoints/pong_pg_latest.pt` | 9000 | 34578828 | -3.3333 |
| Actor-Critic，MC value baseline | `modern_pg_improved/checkpoints/pong_pg_latest.pt` | 10000 | 45211537 | 2.2667 |

- 两个实验均未找到轻量 CSV 训练日志。仅通过 `torch.load(..., map_location="cpu", weights_only=True)` 读取上述两个本地 Checkpoint，未实例化模型或导入环境模块。
- 两者均保存 `episode_rewards`、`episode_timesteps`、`episodes_completed` 和 `total_steps`。Reward 与步数长度等于完成局数，累计步数严格递增且最终值等于 `total_steps`。训练代码确认每次 `env.step()` 累加一次步数，在单局结束时记录累计值。
- `figures/pong_vanilla_pg_training.png` 和 `figures/pong_actor_critic_training.png` 分别使用上述来源。横轴均为从 1 开始的 Episode Number，纵轴为 Episode Reward。浅色线为每局原始奖励，实色线为包括当前局的最近 30 局均值，从第 30 局开始，前 29 局无均值记录。
- 从 Checkpoint 提取的逐局记录分别归档至 `results/pong_vanilla_pg_training_records.csv` 和 `results/pong_actor_critic_training_records.csv`。包含 Episode、原始奖励、累计环境步数和 Avg30，未改写 Checkpoint。
- 现有 Vanilla Checkpoint 还包括 `pong_pg_early_25ep.pt`、`pong_pg_latest_30.pt`。改进版另有 `pong_pg_6000.pt`、`pong_pg_9000.pt`、`pong_pg_10000.pt`。本次未加载这些快照，文件名中的数字未作为验证过的局数。
- 历史图已完成第二轮归档。Vanilla 当前训练历史图位于 `modern_pg/figures/pong_vanilla_pg_avg30_9000episodes.png`，Actor-Critic 的 6,000、9,000、10,000 局图位于 `modern_pg_improved/figures/`。25 局早期独立实验位于 `archive/legacy_figures/modern_pg/early_experiments/`，三张旧 Pong 比较图位于 `archive/legacy_figures/pong_comparisons/`，旧名重复副本位于 `archive/legacy_figures/duplicates/`。完整路径、来源与移动记录见 [图片资产索引](figure_inventory.md)。

## Pong 配置与可比性限制

- 两份现有训练入口均设置 Python、NumPy、Torch 和环境 Seed 42，调用 `Environment("Pong-v0", atari_wrapper=False, test=True)`。两份 Agent 均使用动作 2/3、帧差、gamma 0.99、得失分时重置 Return、Adam 学习率 1e-4、每局一次策略更新。Vanilla 标准化 Return，改进版使用 Critic 和标准化 Advantage，属于带 MC 价值基线的 Actor-Critic。
- `modern_pg/environment.py` 当前映射至 `ALE/Pong-v5`，frame skip 为 (2, 5)，sticky action 概率 0，最大 27000 步，PG 路径无奖励裁剪。
- 改进版训练入口将 `../modern` 加入模块搜索路径以导入 `environment`，但该目录目前不存在。Checkpoint 也未保存环境版本、封装参数、运行配置或续训种子状态，因此当前环境源文件不能确认为两组历史运行的共同环境。
- 严格对照实验的环境可比性待核实。已有描述性对照图 `figures/pong_pg_vs_actor_critic_descriptive.png` 使用两份真实 CSV，图内明确标注历史环境配置未完全核实。两张独立训练图保留各自完整记录。
- 重现命令为 `python -B plot_pong_training_results.py`，使用现有本地 Checkpoint 重新提取记录并绘图。

## 指标区别与输出

Episode Reward 是训练中单局环境返回奖励的总和。Avg30 是同一次训练最近 30 局的滑动平均，各窗口重叠。独立评估 Mean100 是固定训练 Checkpoint 下另行测试 100 局的均值。训练时策略持续更新，训练曲线与独立评估点不能混为同一统计指标。

所有新图均为英文标题和坐标轴、300 DPI PNG。使用现有 `mlds2018-hw4` Python 环境中的 NumPy、PyTorch、Matplotlib 绘制。新增脚本只读取历史记录并生成归档文件。
