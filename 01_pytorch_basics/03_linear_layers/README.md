# 线性层

从检查参数到手动赋值，再到两个训练变体。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [weight_bias.py](experiments/weight_bias.py) | 检查 nn.Linear 的权重、偏置及输出维度 |
| 2 | [layer_hand.py](experiments/layer_hand.py) | 手动指定参数，查看反向传播前后的梯度 |
| 3 | [nn_test](experiments/nn_test) | 固定输入与目标下的线性层训练 |
| 4 | [nn_test2](experiments/nn_test2) | 改变初始化、输入、目标、学习率与打印方式 |

两个 `nn_test` 文件代表不同实验配置，均保留原名。打印与手动赋值有助于理解计算过程，没有清理或重构。
