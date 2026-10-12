# PyTorch 与强化学习学习档案

这是我的个人机器学习与强化学习学习作品集，记录从基础概念、小型实验到完整训练流程的学习过程。

**学习主线：PyTorch 基础 → 深度学习实验 → MNIST / CNN → 强化学习 → 未来 Policy Gradient 与 DQN 项目。**

此前学习过线性代数、概率统计、梯度与优化；这些背景来自个人学习经历，目前仓库没有对应的独立笔记，因此不建立空目录。

## 从哪里开始

| 阶段 | 内容与入口 | 当前状态 |
| --- | --- | --- |
| 1 | [PyTorch 基础](01_pytorch_basics/README.md)：张量、设备、自动求导、优化、线性层 | 基础练习 |
| 2 | [深度学习](02_deep_learning/README.md)：MLP、数据加载、分类、MNIST、CNN | 训练与理解实验 |
| 3 | [强化学习](03_reinforcement_learning/README.md)：Gymnasium、CartPole、REINFORCE | 从单回合更新到重复训练 |
| 4 | [项目](projects/README.md)：李宏毅 RL 2018 Policy Gradient 作业、DQN 作业 | 计划中 |

建议按表中顺序阅读。快速了解学习成果，可以查看 MNIST 训练与加载、CNN 特征可视化，以及两个 REINFORCE 脚本的变化。

- [学习时间线](docs/README.md)
- [完整迁移记录与原始文件校验](docs/migration/README.md)
- [历史 MLP 曲线](02_deep_learning/01_mlp_and_data_loading/artifacts/README.md)
- [历史 MNIST 模型](02_deep_learning/03_mnist_mlp/artifacts/README.md)

## 如何理解这些文件

`experiments/` 保存学习时的原始代码，包括重复实现、打印调试和中间尝试；原文件名与中文注释均予以保留。文件名中的 `test` 通常表示练习，并不代表自动化测试。

`projects/` 用于后续独立项目。目前没有将任何实验标为经过验证的最终实现，也没有宣称已达到特定准确率或 CartPole 分数。现存程序配置不等同于可核验的历史训练记录。

## 环境与运行

从源码可见的直接依赖是 Python、PyTorch、torchvision、Gymnasium、Matplotlib。本次未运行训练、导入验证或核对依赖版本，因此没有生成带有猜测版本的环境文件。

原本地 VS Code 配置使用 Conda 环境 `torch-rl-study`，并包含本机绝对路径；换一台电脑时需要自行选择解释器。

建议在**仓库根目录**运行脚本，例如：

```powershell
python 01_pytorch_basics/01_tensors_and_devices/experiments/tensor_test.py
python 02_deep_learning/04_cnn/experiments/cnn_test1.py
python 03_reinforcement_learning/01_gymnasium_basics/experiments/first_cartpole.py
```

三个没有扩展名的非空 Python 练习也可用 `python 路径` 运行。原文件名中的拼写和扩展名均未调整。

注意：

- `test_torch.py` 和 `tensor_operation.py` 直接使用 CUDA，没有 CPU 回退。
- MNIST 程序使用 `./data`，解释为**当前工作目录**下的 data，而非脚本所在目录。
- MNIST 训练仍把 `mnist_model.pth` 保存到当前工作目录；加载脚本也从当前工作目录寻找此名称。
- 原有两个模型已移至主题下的 artifacts，因此迁移后从根目录直接运行加载脚本，会因根目录没有模型而失败，除非先产生或另行放置所需模型。详情见 [模型说明](02_deep_learning/03_mnist_mlp/artifacts/README.md)。
- MLP 脚本仍将 PNG 写入当前工作目录；历史 PNG 不会自动随重跑更新。
- 在实验目录启动 MNIST 脚本可能下载出另一份 data。训练、下载和输出均可能产生新文件。

## 保留原则

本次只迁移文件并添加 README：没有修改 Python 代码，没有删除实验，没有合并模型，没有新增数学空目录，没有执行 Git 提交。详细时间线根据文件时间与源码推断，不能替代历史版本记录。
