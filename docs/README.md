# 学习时间线

日期来自 2026-10-13 迁移前读取的本地文件创建/修改时间，并结合源码判断。文件可能被复制或后续编辑；仓库当时没有任何 Git 提交，因此以下时间是线索而非严格证明。

| 时间线索（2026 年） | 学习内容 | 对应材料 |
| --- | --- | --- |
| 此前，日期未记录 | 线性代数、概率统计、梯度、优化与 ML 基础 | 个人提供的背景，未发现独立文件 |
| 09-01 | PyTorch 环境、张量、形状、设备 | [张量与设备](../01_pytorch_basics/01_tensors_and_devices/README.md) |
| 09-03 | 梯度累积、手动更新、单步 SGD、循环优化 | [自动求导与优化](../01_pytorch_basics/02_autograd_and_optimization/README.md) |
| 09-05，部分文件后续修改 | 线性层权重、手动赋值、训练变体 | [线性层](../01_pytorch_basics/03_linear_layers/README.md) |
| 09-09 至 09-18，MLP 脚本之后仍有修改 | MLP 回归、损失曲线、Dataset / DataLoader | [MLP](../02_deep_learning/01_mlp_and_data_loading/README.md) |
| 09-18 | 三分类、交叉熵、预测类别与准确率 | [分类](../02_deep_learning/02_classification/README.md) |
| 09-20 至 09-21 | MNIST、训练/评估、模型保存和加载 | [MNIST MLP](../02_deep_learning/03_mnist_mlp/README.md) |
| 09-22 至 09-24 | CNN 形状跟踪、训练、卷积核与特征图 | [CNN](../02_deep_learning/04_cnn/README.md) |
| 09-28 | Gymnasium 环境交互 | [Gymnasium](../03_reinforcement_learning/01_gymnasium_basics/README.md) |
| 10-11 至 10-12 | 单回合策略梯度更新到 1,000 回合训练循环 | [Policy Gradient](../03_reinforcement_learning/02_policy_gradient/README.md) |
| 未来 | 李宏毅 RL 2018 Policy Gradient 与 DQN 作业 | [项目规划](../projects/README.md) |

## 阅读时值得比较的变化

1. 手动修改参数 → 使用优化器 → 重复优化循环。
2. 手工赋值的线性层 → 多层网络 → 数据加载器。
3. 玩具数据 → MNIST → 保存参数并重新加载评估。
4. CNN 中间形状 → 卷积核与特征图的可视化理解。
5. 随机动作交互 → 单回合 REINFORCE 更新 → 重复回合训练。

模型定义、训练循环和回报计算中的重复代码保留在各练习内，方便独立阅读。今后项目需要复用时，可另行提取公共实现，保留这些历史实验作为来源。

[迁移前路径、时间与 SHA-256](migration/README.md)
