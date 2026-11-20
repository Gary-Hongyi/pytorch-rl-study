# 图片资产索引

## 目录管理说明

| 目录 | 职责 | 当前图片数 |
| --- | --- | ---: |
| `figures/` | 指定的六张正式精选汇报图 | 6 |
| `modern_pg/figures/` | 当前 Vanilla PG 训练历史图 | 1 |
| `modern_pg_improved/figures/` | Actor-Critic 6,000、9,000、10,000 局历史图 | 3 |
| `modern_dqn/figures/` | DQN Baseline 各训练节点与完整训练图 | 6 |
| `modern_double_dqn/figures/` | Double DQN 各训练节点与完整训练图 | 5 |
| `archive/legacy_figures/` | 旧版跨算法图、重复副本、早期来源不完整图和调试图 | 17 |

正式汇报从根目录 `figures/` 取图。算法历史训练阶段从各算法的 `figures/` 取图。旧版本与重复内容集中归档，仅用于追溯。项目根目录及四个算法目录本层不直接保存 PNG。绘图脚本保留原位置。

第二轮开始时有 38 个 PNG 文件、29 份不同内容。本轮移动 17 个文件，整理后仍为 38 个文件、29 份不同内容，图像内容保持一致。重复文件保留于归档中，不占用正式图库和算法图库。不同节点共享早期曲线属于正常的训练历史，按节点保留。

## 正式精选汇报图

| 图片 | 用途与口径 |
| --- | --- |
| [pong_vanilla_pg_training.png](../figures/pong_vanilla_pg_training.png) | Vanilla PG 9,000 局原始 Reward 与 Avg30，300 DPI |
| [pong_actor_critic_training.png](../figures/pong_actor_critic_training.png) | AC 10,000 局原始 Reward 与 Avg30，300 DPI |
| [pong_pg_vs_actor_critic_descriptive.png](../figures/pong_pg_vs_actor_critic_descriptive.png) | 描述性学习过程对照，图内保留环境配置未完全核实的说明，300 DPI |
| [dqn_eval_mean100_comparison.png](../figures/dqn_eval_mean100_comparison.png) | Breakout 各节点独立评估 Mean100，300 DPI |
| [dqn_eval_seed_robustness.png](../figures/dqn_eval_seed_robustness.png) | Breakout 1.75M 节点的两个测试种子验证，300 DPI |
| [breakout_dqn_vs_double_dqn_avg30_1750000steps.png](../figures/breakout_dqn_vs_double_dqn_avg30_1750000steps.png) | Breakout 共同训练范围 Avg30，历史原图为 160 DPI |

## 数据来源代码
表格中的来源编号指向以下真实数据及绘图逻辑。临时复画核验文件保存于系统临时目录，不计入项目图片资产。逐像素一致说明历史图可以从这些数值与样式复现，历史运行时环境的可比性仍需独立核实。

- **P0** 来自 [25 局早期 Checkpoint](../modern_pg/checkpoints/pong_pg_early_25ep.pt) 的 `episode_rewards`。图中单局序列与该记录核对一致，显示原始 Reward 和 5 局均值，Checkpoint 缺少累计 Steps。当前原绘图程序缺失，该记录与 P1 的前 25 局不同。
- **P1** 使用 [Vanilla CSV](pong_vanilla_pg_training_records.csv)，其上游为 [Vanilla 最新 Checkpoint](../modern_pg/checkpoints/pong_pg_latest.pt)。历史 Avg30 图按 [旧绘图脚本](../modern_pg/pg_plot_reward.py) 的样式复现，逐像素一致。最终原始 Reward + Avg30 图由 [归档绘图脚本](../plot_pong_training_results.py) 生成。
- **P2** 使用 [Actor-Critic CSV](pong_actor_critic_training_records.csv)，其上游为 [AC 最新 Checkpoint](../modern_pg_improved/checkpoints/pong_pg_latest.pt)。6,000、9,000、10,000 局快照的 Reward 和累计步数均与 CSV 对应前缀完全一致。历史图按 [旧绘图脚本](../modern_pg_improved/pg_plot_reward.py) 的样式逐像素复现。最终图由 [归档绘图脚本](../plot_pong_training_results.py) 生成。
- **PC** 使用 P1、P2 及 [历史比较脚本](../compare_pg.py) 的 Avg30 和共同累计步数逻辑。上一轮使用实际 Vanilla CSV 和 AC 快照前缀在临时目录复现，两个不同版本均逐像素一致。本轮将脚本中的失效数据路径 `modern/` 修正为 `modern_pg/`，输出改至旧版归档目录。
- **B0** 使用 [DQN smoke CSV](../modern_dqn/logs/dqn_smoke.csv) 和单算法绘图样式复现，逐像素一致。
- **B1** 使用 [Baseline 训练 CSV](../modern_dqn/logs/dqn_baseline_cpu_01.csv) 中 `total_steps`、`reward`，截取指定节点前的完整 Episode，按 [Baseline 绘图脚本](../modern_dqn/plot_dqn.py) 复现，6 个节点均逐像素一致。
- **B2** 使用 [Double DQN 训练 CSV](../modern_double_dqn/logs/double_dqn_cpu_01.csv) 和 [Double DQN 绘图脚本](../modern_double_dqn/plot_dqn.py)，5 个节点均逐像素复现。
- **BC** 使用 B1、B2 和 [训练比较脚本](../compare_dqn.py)。使用完整 Baseline 日志及对应 Double DQN 节点的前缀，按共同累计 Steps 范围截取，5 张均逐像素复现。
- **E1** 使用 [独立评估 CSV](dqn_evaluation_records.csv) 和 [评估绘图脚本](../plot_dqn_evaluations.py)，为用户确认的 13 条历史均值记录。
- **PN** 使用 P1、P2 和上一轮新增的 [描述性比较脚本](../plot_pong_descriptive_comparison.py)，仅重算最近 30 局均值，不填补、外推或插值结果。

## 全部图片最终位置

路径相对于项目根目录。历史位置列记录本轮开始时的位置，属于迁移记录。归档中的重复文件保留了旧文件名，算法图库中保留明确命名的代表文件。

| 最终相对路径 | 所属算法 | 图表类型与坐标 | 训练节点 | 当前用途 | 本轮迁移前位置 | 来源与限制 |
| --- | --- | --- | --- | --- | --- | --- |
| [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1000000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1000000.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 1,000,000 Steps | 旧版比较图追溯 | `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1000000.png` | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1250000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1250000.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 1,250,000 Steps | 旧版比较图追溯 | `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1250000.png` | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1500000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1500000.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 1,500,000 Steps | 旧版比较图追溯 | `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1500000.png` | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1750000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1750000.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 1,750,000 Steps | 旧版比较图追溯 | `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1750000.png` | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_500000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_500000.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 500,000 Steps | 旧版比较图追溯 | `modern_double_dqn/figures/dqn_vs_double_dqn_curve_500000.png` | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [archive/legacy_figures/debug/modern_dqn/dqn_learning_curve.png](../archive/legacy_figures/debug/modern_dqn/dqn_learning_curve.png) | DQN smoke | 冒烟训练；Steps；原始 Reward | 2 Episodes，结束于 23 / 44 Steps | 调试历史 | `modern_dqn/figures/dqn_learning_curve.png` | B0；160 DPI；奖励均为 0，与 smoke CSV 复画逐像素一致；不用于正式结果 |
| [archive/legacy_figures/duplicates/breakout/breakout_double_dqn_training_1750000steps.png](../archive/legacy_figures/duplicates/breakout/breakout_double_dqn_training_1750000steps.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 1,750,000 Steps | 重复副本追溯 | `figures/breakout_double_dqn_training_1750000steps.png` | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [archive/legacy_figures/duplicates/breakout/breakout_dqn_training_2000000steps.png](../archive/legacy_figures/duplicates/breakout/breakout_dqn_training_2000000steps.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 2,000,000 Steps | 重复副本追溯 | `figures/breakout_dqn_training_2000000steps.png` | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [archive/legacy_figures/duplicates/modern_pg/pong_pg_avg30_vs_steps_9000episode.png](../archive/legacy_figures/duplicates/modern_pg/pong_pg_avg30_vs_steps_9000episode.png) | Vanilla PG | 训练；累计 Steps；Avg30 | 9,000 Episodes / 34,578,828 Steps | 重复副本追溯 | `modern_pg/pong_pg_avg30_vs_steps_9000episode.png` | P1；150 DPI；由 CSV 按旧脚本样式复画后逐像素一致 |
| [archive/legacy_figures/duplicates/modern_pg/pong_pg_rewards.png](../archive/legacy_figures/duplicates/modern_pg/pong_pg_rewards.png) | Vanilla PG 早期实验 | 训练；Episode；原始 Reward + Avg5 | 25 Episodes | 重复副本追溯 | `modern_pg/pong_pg_rewards.png` | P0；150 DPI；早期 Checkpoint 无累计步数，与当前 9,000 局轨迹开头不同；原绘图脚本缺失 |
| [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_avg30_vs_steps.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_avg30_vs_steps.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 10,000 Episodes / 45,211,537 Steps | 重复副本追溯 | `modern_pg_improved/pong_pg_avg30_vs_steps.png` | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |
| [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_6000.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_6000.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 6,000 Episodes / 18,767,308 Steps | 重复副本追溯 | `modern_pg_improved/pong_pg_improved_6000.png` | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |
| [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_9000.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_9000.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 9,000 Episodes / 38,298,580 Steps | 重复副本追溯 | `modern_pg_improved/pong_pg_improved_9000.png` | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |
| [archive/legacy_figures/modern_pg/early_experiments/pong_vanilla_pg_early_25episodes_reward_avg5.png](../archive/legacy_figures/modern_pg/early_experiments/pong_vanilla_pg_early_25episodes_reward_avg5.png) | Vanilla PG 早期实验 | 训练；Episode；原始 Reward + Avg5 | 25 Episodes | 早期实验追溯 | `modern_pg/figures/pong_vanilla_pg_early_25episodes_reward_avg5.png` | P0；150 DPI；早期 Checkpoint 无累计步数，与当前 9,000 局轨迹开头不同；原绘图脚本缺失 |
| [archive/legacy_figures/pong_comparisons/pong_pg_comparison.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison.png) | Vanilla PG / Actor-Critic | 历史描述性对比；累计 Steps；Avg30 | 与 comparison_9000 完全重复 | 旧版比较图追溯 | `pong_pg_comparison.png` | PC；150 DPI；数据可精确追溯，历史环境可比性待核实 |
| [archive/legacy_figures/pong_comparisons/pong_pg_comparison_6000.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison_6000.png) | Vanilla PG / Actor-Critic | 历史描述性对比；累计 Steps；Avg30 | AC 6,000 局；PG 截至 6,335 局 | 旧版比较图追溯 | `pong_pg_comparison_6000.png` | PC；150 DPI；数据可精确追溯，历史环境可比性待核实 |
| [archive/legacy_figures/pong_comparisons/pong_pg_comparison_9000.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison_9000.png) | Vanilla PG / Actor-Critic | 历史描述性对比；累计 Steps；Avg30 | AC 9,000 局快照；PG 9,000 / AC 截至 8,461 局 | 旧版比较图追溯 | `pong_pg_comparison_9000.png` | PC；150 DPI；数据可精确追溯，历史环境可比性待核实 |
| [figures/breakout_dqn_vs_double_dqn_avg30_1750000steps.png](../figures/breakout_dqn_vs_double_dqn_avg30_1750000steps.png) | DQN / Double DQN | 训练对比；共同累计 Steps；Avg30 | Double DQN 节点 1,750,000 Steps | 正式精选 | 原位保留 | BC；160 DPI；完整 Baseline 日志与 Double DQN 节点前缀按共同范围复画，逐像素一致 |
| [figures/dqn_eval_mean100_comparison.png](../figures/dqn_eval_mean100_comparison.png) | DQN / Double DQN | 独立评估；Steps；Mean100 | 500k–2M；Double DQN 止于 1.75M | 正式精选 | 原位保留 | E1；300 DPI；测试 Seed 11037；逐局评估记录缺失 |
| [figures/dqn_eval_seed_robustness.png](../figures/dqn_eval_seed_robustness.png) | DQN / Double DQN | 独立评估；测试 Seed；Mean100 | 1.75M Steps | 正式精选 | 原位保留 | E1；300 DPI；仅两个测试种子，不能推断稳定性 |
| [figures/pong_actor_critic_training.png](../figures/pong_actor_critic_training.png) | Actor-Critic | 训练；Episode；原始 Reward + Avg30 | 10,000 Episodes | 正式精选 | 原位保留 | P2；300 DPI；历史环境版本待核实 |
| [figures/pong_pg_vs_actor_critic_descriptive.png](../figures/pong_pg_vs_actor_critic_descriptive.png) | Vanilla PG / Actor-Critic | 描述性训练对比；Episode 与累计 Steps；Avg30 | PG 9,000 / AC 10,000 Episodes | 正式精选 | 原位保留 | PN；300 DPI；图内注明历史环境配置未完全核实、预算不同 |
| [figures/pong_vanilla_pg_training.png](../figures/pong_vanilla_pg_training.png) | Vanilla PG | 训练；Episode；原始 Reward + Avg30 | 9,000 Episodes | 正式精选 | 原位保留 | P1；300 DPI；历史环境版本待核实 |
| [modern_double_dqn/figures/double_dqn_cpu_01_curve_1000000.png](../modern_double_dqn/figures/double_dqn_cpu_01_curve_1000000.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 1,000,000 Steps | 算法历史节点 | 原位保留 | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_double_dqn/figures/double_dqn_cpu_01_curve_1250000.png](../modern_double_dqn/figures/double_dqn_cpu_01_curve_1250000.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 1,250,000 Steps | 算法历史节点 | 原位保留 | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_double_dqn/figures/double_dqn_cpu_01_curve_1500000.png](../modern_double_dqn/figures/double_dqn_cpu_01_curve_1500000.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 1,500,000 Steps | 算法历史节点 | 原位保留 | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_double_dqn/figures/double_dqn_cpu_01_curve_1750000.png](../modern_double_dqn/figures/double_dqn_cpu_01_curve_1750000.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 1,750,000 Steps | 算法历史节点 | 原位保留 | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_double_dqn/figures/double_dqn_cpu_01_curve_500000.png](../modern_double_dqn/figures/double_dqn_cpu_01_curve_500000.png) | Double DQN | 训练；累计 Steps；原始 Reward + Avg30 | 500,000 Steps | 算法历史节点 | 原位保留 | B2；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_100000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_100000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 100,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_1000000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_1000000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 1,000,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_1250000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_1250000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 1,250,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_1750000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_1750000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 1,750,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_2000000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_2000000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 2,000,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_dqn/figures/dqn_baseline_cpu_01_curve_500000.png](../modern_dqn/figures/dqn_baseline_cpu_01_curve_500000.png) | DQN Baseline | 训练；累计 Steps；原始 Reward + Avg30 | 500,000 Steps | 算法历史节点 | 原位保留 | B1；160 DPI；用日志截至节点的记录复画，逐像素一致 |
| [modern_pg/figures/pong_vanilla_pg_avg30_9000episodes.png](../modern_pg/figures/pong_vanilla_pg_avg30_9000episodes.png) | Vanilla PG | 训练；累计 Steps；Avg30 | 9,000 Episodes / 34,578,828 Steps | 算法历史节点 | 原位保留 | P1；150 DPI；由 CSV 按旧脚本样式复画后逐像素一致 |
| [modern_pg_improved/figures/pong_actor_critic_avg30_10000episodes.png](../modern_pg_improved/figures/pong_actor_critic_avg30_10000episodes.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 10,000 Episodes / 45,211,537 Steps | 算法历史节点 | 原位保留 | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |
| [modern_pg_improved/figures/pong_actor_critic_avg30_6000episodes.png](../modern_pg_improved/figures/pong_actor_critic_avg30_6000episodes.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 6,000 Episodes / 18,767,308 Steps | 算法历史节点 | 原位保留 | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |
| [modern_pg_improved/figures/pong_actor_critic_avg30_9000episodes.png](../modern_pg_improved/figures/pong_actor_critic_avg30_9000episodes.png) | Actor-Critic | 训练；累计 Steps；Avg30 | 9,000 Episodes / 38,298,580 Steps | 算法历史节点 | 原位保留 | P2；150 DPI；与 CSV 复画逐像素一致；旧图标题笼统写 Policy Gradient，归档名注明 AC |

## 本轮移动记录

| 原位置 | 最终位置 |
| --- | --- |
| `pong_pg_comparison.png` | [archive/legacy_figures/pong_comparisons/pong_pg_comparison.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison.png) |
| `pong_pg_comparison_6000.png` | [archive/legacy_figures/pong_comparisons/pong_pg_comparison_6000.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison_6000.png) |
| `pong_pg_comparison_9000.png` | [archive/legacy_figures/pong_comparisons/pong_pg_comparison_9000.png](../archive/legacy_figures/pong_comparisons/pong_pg_comparison_9000.png) |
| `modern_pg/pong_pg_rewards.png` | [archive/legacy_figures/duplicates/modern_pg/pong_pg_rewards.png](../archive/legacy_figures/duplicates/modern_pg/pong_pg_rewards.png) |
| `modern_pg/pong_pg_avg30_vs_steps_9000episode.png` | [archive/legacy_figures/duplicates/modern_pg/pong_pg_avg30_vs_steps_9000episode.png](../archive/legacy_figures/duplicates/modern_pg/pong_pg_avg30_vs_steps_9000episode.png) |
| `modern_pg_improved/pong_pg_avg30_vs_steps.png` | [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_avg30_vs_steps.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_avg30_vs_steps.png) |
| `modern_pg_improved/pong_pg_improved_6000.png` | [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_6000.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_6000.png) |
| `modern_pg_improved/pong_pg_improved_9000.png` | [archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_9000.png](../archive/legacy_figures/duplicates/modern_pg_improved/pong_pg_improved_9000.png) |
| `figures/breakout_dqn_training_2000000steps.png` | [archive/legacy_figures/duplicates/breakout/breakout_dqn_training_2000000steps.png](../archive/legacy_figures/duplicates/breakout/breakout_dqn_training_2000000steps.png) |
| `figures/breakout_double_dqn_training_1750000steps.png` | [archive/legacy_figures/duplicates/breakout/breakout_double_dqn_training_1750000steps.png](../archive/legacy_figures/duplicates/breakout/breakout_double_dqn_training_1750000steps.png) |
| `modern_pg/figures/pong_vanilla_pg_early_25episodes_reward_avg5.png` | [archive/legacy_figures/modern_pg/early_experiments/pong_vanilla_pg_early_25episodes_reward_avg5.png](../archive/legacy_figures/modern_pg/early_experiments/pong_vanilla_pg_early_25episodes_reward_avg5.png) |
| `modern_dqn/figures/dqn_learning_curve.png` | [archive/legacy_figures/debug/modern_dqn/dqn_learning_curve.png](../archive/legacy_figures/debug/modern_dqn/dqn_learning_curve.png) |
| `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1000000.png` | [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1000000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1000000.png) |
| `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1250000.png` | [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1250000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1250000.png) |
| `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1500000.png` | [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1500000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1500000.png) |
| `modern_double_dqn/figures/dqn_vs_double_dqn_curve_1750000.png` | [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1750000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_1750000.png) |
| `modern_double_dqn/figures/dqn_vs_double_dqn_curve_500000.png` | [archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_500000.png](../archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_500000.png) |

## 归档分类与重复说明

- `archive/legacy_figures/pong_comparisons/` 保存根目录原有三张 Pong 比较图。其中 `pong_pg_comparison.png` 与 `pong_pg_comparison_9000.png` 内容相同，SHA-256 为 `b85e99802eaabc71e7e525bdfa070a12d654a76c419ab79a370f40093441860d`。
- `archive/legacy_figures/breakout_comparisons/` 保存五个节点的旧 Avg30 比较图。1.75M 版本与正式精选图库的对应图相同，其余版本保留训练过程追溯价值。
- `archive/legacy_figures/duplicates/modern_pg/` 和 `duplicates/modern_pg_improved/` 保存原算法目录散落的旧名副本，明确命名的正式历史图保留在对应算法图库。
- `archive/legacy_figures/duplicates/breakout/` 保存上一轮复制到根图库的两张单算法完整曲线副本。对应有效图位于 [DQN 2M 历史图](../modern_dqn/figures/dqn_baseline_cpu_01_curve_2000000.png) 和 [Double DQN 1.75M 历史图](../modern_double_dqn/figures/double_dqn_cpu_01_curve_1750000.png)。
- `archive/legacy_figures/modern_pg/early_experiments/` 保存 25 局独立早期 PG 图。该图使用 Avg5，Checkpoint 缺少累计步数，原绘图脚本待核实，不能拼接入当前 9,000 局轨迹。其旧名副本仍作为重复资产归档。
- `archive/legacy_figures/debug/modern_dqn/` 保存仅 44 Steps 的 smoke 图。
- 正式精选目录和各算法图库内部经哈希核对均无重复内容。归档保留全部历史文件，未删除重复副本。

## 绘图脚本输出路径管理

| 脚本 | 本轮后的输出路径 |
| --- | --- |
| [compare_pg.py](../compare_pg.py) | `archive/legacy_figures/pong_comparisons/pong_pg_comparison_regenerated.png` |
| [compare_dqn.py](../compare_dqn.py) | `archive/legacy_figures/breakout_comparisons/dqn_vs_double_dqn_curve_regenerated.png` |
| [Vanilla pg_plot_reward.py](../modern_pg/pg_plot_reward.py) | `modern_pg/figures/pong_vanilla_pg_avg30_regenerated.png` |
| [AC pg_plot_reward.py](../modern_pg_improved/pg_plot_reward.py) | `modern_pg_improved/figures/pong_actor_critic_avg30_regenerated.png` |

以上四个脚本会创建所需父目录，目标图片已存在时停止，防止覆盖历史输出。`compare_pg.py` 的 Vanilla 输入路径已修正至现有 `modern_pg/checkpoints/pong_pg_latest.pt`。这些改动仅涉及绘图路径与输出保护。

其余绘图脚本的输出位置已符合目录职责，维持原有行为。`plot_pong_training_results.py` 与 `plot_dqn_evaluations.py` 可更新对应最终图，`plot_pong_descriptive_comparison.py` 遇到已有同名图片会停止。两份单算法 DQN 绘图脚本继续输出各算法 `figures/`。本轮仅静态检查脚本，未运行绘图、训练或评估。

## Pong 历史比较图的准确范围
| 历史图 | 共同累计 Steps 上限 | PG 实际展示局数 | AC 实际展示局数 |
| --- | ---: | ---: | ---: |
| comparison_6000 | 18,767,308 | 6,335 | 6,000 |
| comparison_9000 与 comparison | 34,578,828 | 9,000 | 8,461 |

`comparison_6000` 使用 AC 6,000 局快照，Vanilla 采用现有完整历史并截至共同步数。`comparison_9000` 使用 AC 9,000 局快照，但共同预算由 Vanilla 的 34,578,828 Steps 决定，AC 最后一条可展示记录为第 8,461 局的 34,575,040 Steps。故文件名中的局数不能解释为两条曲线都展示相同数量 Episodes。历史脚本末尾还用 `np.interp` 打印共同步数处的值，本轮和上一轮核验均未执行该打印逻辑，未采用这些插值数值。

两组记录的奖励都由逐局环境返回奖励求和，Avg30 数学定义相同。当前代码显示相同动作映射与 MC 回报流程，Checkpoint 的 Actor 权重维度也一致。改进版依赖已不存在的 `modern/` 环境与预处理模块，历史版本、封装参数及实际导入路径待核实，因此这些图片归类为描述性训练对照。数据与像素可追溯不等于严格实验可比性已成立。

## 指标口径与待核实事项
- Episode Reward 是训练中单个包装 Episode 的奖励总和，Avg5 和 Avg30 分别是最近 5 局与 30 局均值，Mean100 为固定 Checkpoint 独立测试 100 个 Episode 的均值。训练与评估指标分别保留。
- 当前 Breakout 代码在测试时也以丢命形成 Episode 边界，图中的独立评估应按该包装流程解释，完整多条命游戏评分口径待核实。
- Pong 两组历史环境、预处理源文件、软件版本及续训配置待人工核实。源代码模块缺失属于可比性证据缺口。
- 25 局早期 PG 图的原始绘图脚本与实验配置待核实。其奖励可追溯至早期 Checkpoint，累计步数未保存。
- 旧图中的 Official Baseline / Official Target 文案保留原样，是否对应当前环境下适用的课程评分标准待核实。
- 评估 CSV 保存用户确认的均值，逐局评估原始记录与跨训练种子稳定性待核实。本索引不以测试种子图推断统计显著性。

## 文件保护与验证

本轮已验证全部 38 张图片在最终位置可解码打开，移动前后 SHA-256 保持一致。全部图片的数量与内容集合一致。项目根目录 PNG 数量为零，根目录 `figures/` 恰好包含六张指定图片。

检查本项目 Markdown 图片链接、资产索引引用及其他本轮受影响的路径，更新了数据说明中的旧图片位置。四个调整后的绘图脚本通过 Python AST 语法检查及路径检查。训练算法、网络结构、超参数、Checkpoint、CSV 与训练日志均保持原文件内容。
