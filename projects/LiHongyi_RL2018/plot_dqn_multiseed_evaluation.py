"""Validate completed multi-seed results, create three figures and concise notes."""
import csv
import platform
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [11037, 2026, 2027, 10007, 27182, 31415, 4242, 7777, 8888, 12345]
ALGORITHMS = ["DQN Baseline", "Double DQN"]
COLORS = ["#0F4D92", "#B64342"]


def read_csv(name):
    with (ROOT / "results" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = read_csv("dqn_eval_multiseed_1750000_summary.csv")
    episodes = read_csv("dqn_eval_multiseed_1750000_episodes.csv")
    pairs = read_csv("dqn_eval_multiseed_1750000_pairwise.csv")
    if len(summary) != 20 or len(episodes) != 2000 or len(pairs) != 10:
        raise ValueError("Expected 20 summary rows, 2000 episodes and 10 pairs")
    expected = {(a, s) for a in ALGORITHMS for s in SEEDS}
    index = {(row["algorithm"], int(row["eval_seed"])): row for row in summary}
    if set(index) != expected or [int(row["eval_seed"]) for row in pairs] != SEEDS:
        raise ValueError("Unexpected algorithms or evaluation seeds")
    for algorithm, seed in expected:
        group = [row for row in episodes if row["algorithm"] == algorithm and int(row["eval_seed"]) == seed]
        if [int(row["episode"]) for row in group] != list(range(1, 101)):
            raise ValueError(f"Invalid episode indices: {algorithm}, {seed}")
        values = [float(row["episode_reward"]) for row in group]
        row = index[(algorithm, seed)]
        calculations = {"mean_reward": statistics.mean(values), "std_reward": statistics.pstdev(values),
                        "median_reward": statistics.median(values), "min_reward": min(values),
                        "max_reward": max(values)}
        for key, actual in calculations.items():
            if not np.isclose(actual, float(row[key]), rtol=0, atol=1e-12):
                raise ValueError(f"Summary mismatch: {algorithm}, {seed}, {key}")
        if int(row["episodes"]) != 100 or row["reward_gt_40"] != str(calculations["mean_reward"] > 40):
            raise ValueError("Episode count / threshold mismatch")
        for detail in group:
            for field in ["checkpoint_name", "checkpoint_path", "checkpoint_sha256", "training_steps", "test_epsilon"]:
                if detail[field] != row[field]:
                    raise ValueError(f"Episode provenance mismatch: {field}")
    base = np.array([float(index[(ALGORITHMS[0], s)]["mean_reward"]) for s in SEEDS])
    double = np.array([float(index[(ALGORITHMS[1], s)]["mean_reward"]) for s in SEEDS])
    differences = double - base
    for row, b, d, delta in zip(pairs, base, double, differences):
        if not np.allclose([float(row[k]) for k in ["dqn_mean_reward", "double_dqn_mean_reward", "difference_double_minus_dqn"]],
                           [b, d, delta], rtol=0, atol=1e-12):
            raise ValueError("Pairwise table mismatch")

    figure_names = ["dqn_eval_multiseed_mean100_1750000.png", "dqn_eval_multiseed_difference_1750000.png",
                    "dqn_eval_multiseed_distribution_1750000.png"]
    note = ROOT / "results/dqn_eval_multiseed_1750000_notes.md"
    for path in [*(ROOT / "figures" / name for name in figure_names), note]:
        if path.exists():
            raise FileExistsError(path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    footer = "1,750,000 training steps | Training seed 42 | Test epsilon 0.025 | 100 wrapped episodes per seed"

    def save(fig, name):
        fig.text(0.5, 0.02, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.065, 1, 1), pad=1.5)
        path = ROOT / "figures" / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=300, facecolor="white")
        plt.close(fig)

    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for shift, values, color, label in zip([-0.18, 0.18], [base, double], COLORS, ALGORITHMS):
        bars = ax.bar(x + shift, values, width=0.36, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(40, color="#767676", linestyle="--", label="Reference reward = 40")
    ax.set_xticks(x, [str(s) for s in SEEDS])
    ax.set(title="Breakout independent evaluation across 10 test seeds", xlabel="Evaluation seed",
           ylabel="Mean100 unclipped reward", ylim=(0, max(base.max(), double.max(), 40) * 1.24))
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    save(fig, figure_names[0])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(x, differences, color=[COLORS[1] if v >= 0 else COLORS[0] for v in differences])
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in differences], padding=4, fontsize=10)
    ax.axhline(0, color="#4D4D4D", linewidth=1)
    ax.set_xticks(x, [str(s) for s in SEEDS])
    ax.set(title="Breakout paired differences by evaluation seed", xlabel="Evaluation seed",
           ylabel="Mean100 difference (Double DQN - DQN)")
    ax.margins(y=0.22)
    ax.grid(axis="y", alpha=0.15)
    save(fig, figure_names[1])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    boxes = ax.boxplot([base, double], positions=[1, 2], widths=0.35, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set(facecolor=color, alpha=0.22)
    offsets = np.linspace(-0.09, 0.09, 10)
    for i, (b, d) in enumerate(zip(base, double)):
        ax.plot([1 + offsets[i], 2 + offsets[i]], [b, d], color="#999999", alpha=0.45, linewidth=0.8)
    ax.scatter(1 + offsets, base, color=COLORS[0], s=35, zorder=3)
    ax.scatter(2 + offsets, double, color=COLORS[1], s=35, zorder=3)
    ax.axhline(40, color="#767676", linestyle="--", linewidth=1)
    ax.set_xticks([1, 2], ALGORITHMS)
    ax.set(title="Breakout distribution of seed-level Mean100 rewards", ylabel="Mean100 unclipped reward")
    ax.text(0.5, 0.02, "Each dot is one evaluation seed; lines connect the same seed", transform=ax.transAxes,
            ha="center", fontsize=9)
    ax.margins(y=0.2)
    save(fig, figure_names[2])

    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    lines = ["# Breakout 1,750,000 Steps 多测试 Seed 评估", "", "## 协议与数据", "",
             "固定两个已训练 Checkpoint，仅改变 evaluation seed。训练 Seed 42 沿用历史训练记录，未重新训练。两份 Checkpoint 内部 total_steps 均已核实为 1,750,000。所有 20 组结果均为本轮重新评估，未混用旧均值。", "",
             "使用项目 Environment(BreakoutNoFrameskip-v4, atari_wrapper=True, test=True)，epsilon=0.025，未裁剪奖励。每组完成 100 个包装 Episode，每次评估前重新设置 Python、NumPy、Torch 和环境种子。当前 EpisodicLifeEnv 在丢命时也终止一个 Episode，此口径与历史评估脚本一致，100 个包装 Episode 不等同于 100 次完整多条命游戏。", "",
             "专用脚本复用原 Agent 的 make_action、观察转换与 init_game_setting。两个算法的推理方法、CNN 和环境封装已确认相同，仅载入各自 Online 权重，未实例化训练优化器或 Replay。CPU 单线程执行以减少小型 CNN 的线程调度开销，torch.inference_mode() 下推理。执行前后模型权重与原始 Checkpoint 的哈希一致。", "",
             "## 逐 Seed 结果", "", "| Evaluation seed | DQN Mean100 | Double DQN Mean100 | Double−DQN |",
             "| ---: | ---: | ---: | ---: |"]
    for seed, b, d, delta in zip(SEEDS, base, double, differences):
        lines.append(f"| {seed} | {b:.2f} | {d:.2f} | {delta:+.2f} |")
    lines += ["", "## 跨 Seed 汇总", "", "| 指标 | DQN Baseline | Double DQN |",
              "| --- | ---: | ---: |",
              f"| Mean100 的跨 seed 平均 | {base.mean():.4f} | {double.mean():.4f} |",
              f"| Mean100 的跨 seed 标准差（ddof=0） | {base.std():.4f} | {double.std():.4f} |",
              f"| Mean100 最小值 | {base.min():.2f} | {double.min():.2f} |",
              f"| Mean100 最大值 | {base.max():.2f} | {double.max():.2f} |",
              f"| Mean100 > 40 的 seed 数 | {int((base > 40).sum())} | {int((double > 40).sum())} |", "",
              f"Double DQN 领先 {wins} 个 seeds，DQN 领先 {losses} 个，持平 {ties} 个。配对差值均值为 {differences.mean():+.4f}，总体标准差为 {differences.std(ddof=0):.4f}，中位数为 {np.median(differences):+.4f}，范围为 [{differences.min():+.2f}, {differences.max():+.2f}]。", "",
              "summary.csv 的 std_reward 表示单个模型、单个 seed 下 100 局 Reward 的总体标准差。上表及配对差值标准差则描述 10 个 seed 级别均值或差值的离散程度，两种统计层级分别解释。reward_gt_40 表示该组 Mean100 是否严格大于 40。", "",
              "## 汇报意义与边界", "",
              "本实验研究固定训练结果的 evaluation seed sensitivity。图表展示测试随机种子如何影响 Mean100 和同 seed 下的算法排名，可用于补充单测试 seed 结论的适用范围。", "",
              "这是每种算法各一份训练 Seed 42 Checkpoint 的测试，未估计多 training seeds 的训练稳定性。跨测试 seed 的离散程度包含有限 100 局采样和测试轨迹随机性的影响，不能据此证明算法普遍更稳定或建立统计显著性。相同 seed 统一随机数初始化，但两策略动作会导致环境轨迹分化，并非逐局完全相同的测试样本。", "",
              f"与旧表核对时，本轮 Baseline 在 Seed 11037 的 Mean100 为 {base[SEEDS.index(11037)]:.2f}，旧归档为 31.32。旧记录未包含逐局数据及 Checkpoint 哈希，差异来源待核实；不能直接将两次结果当作同一批样本。本轮使用已核实步数及哈希的两个文件重新完成全部 seed，统计不混入旧结果。CPU 线程数等数值运行条件已在上文记录，未将其认定为差异原因。", "",
              "## 文件与核验", "",
              "- [逐局 Reward 明细](dqn_eval_multiseed_1750000_episodes.csv)，2,000 行。",
              "- [模型与 Seed 汇总](dqn_eval_multiseed_1750000_summary.csv)，20 行。",
              "- [配对比较表](dqn_eval_multiseed_1750000_pairwise.csv)，10 行。",
              "- [Mean100 比较图](../figures/" + figure_names[0] + ")。",
              "- [配对差值图](../figures/" + figure_names[1] + ")。",
              "- [Seed 级均值分布图](../figures/" + figure_names[2] + ")。", "",
              "所有统计从逐局 CSV 独立复算，检查 10 个指定 seeds、每组 1–100 连续编号、20 组汇总、配对差值及阈值标志一致性。统计图为英文、300 DPI。CSV 中保留 Checkpoint 绝对路径、SHA-256、训练步数和测试 epsilon。", ""]
    for algorithm in ALGORITHMS:
        row = index[(algorithm, SEEDS[0])]
        lines += [f"- {algorithm} Checkpoint `{row['checkpoint_path']}`，SHA-256 `{row['checkpoint_sha256']}`。"]
    lines += ["", f"绘图环境 Python {platform.python_version()}，NumPy {np.__version__}，Matplotlib {matplotlib.__version__}。", ""]
    with note.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"VALIDATED: 20 groups / 2000 episodes / 10 pairs; Double wins={wins}, DQN wins={losses}, ties={ties}")
    print(f"Mean100 across seeds: DQN={base.mean():.4f}, Double={double.mean():.4f}, paired difference={differences.mean():+.4f}")


if __name__ == "__main__":
    main()
