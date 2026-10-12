# CNN

先追踪网络各层形状，再训练并观察卷积特征。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [cnn_test1.py](experiments/cnn_test1.py) | 打印卷积、池化、flatten 和全连接层的中间形状 |
| 2 | [cnn_mnist_train.py](experiments/cnn_mnist_train.py) | MNIST CNN 训练、测试、第一层卷积核与两层特征图可视化 |

形状打印是理解网络的重要记录，保持不变。训练脚本使用当前工作目录下的 `./data`，图像通过 `plt.show()` 展示；未发现它保存的 CNN checkpoint 或特征图文件，本次也未新增实验输出。
