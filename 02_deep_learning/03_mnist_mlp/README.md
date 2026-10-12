# MNIST MLP

把神经网络训练迁移到图像数据，并学习模型保存与重新加载。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [mnist_test.py](experiments/mnist_test.py) | 784→128→10 MLP，训练、测试、单张预测可视化和保存参数 |
| 2 | [mnist_test_load.py](experiments/mnist_test_load.py) | 重新创建同结构模型，加载参数并在测试集评估 |

两个脚本保留各自的模型定义。[两个历史 checkpoint 的来源与路径限制](artifacts/README.md)。MNIST 使用当前工作目录下的 `./data`；建议训练时从仓库根目录启动。加载脚本目前仍寻找当前工作目录下的 `mnist_model.pth`，不会自动选择迁移后的历史模型。
