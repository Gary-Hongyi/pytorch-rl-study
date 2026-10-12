# MLP 与数据加载

从完整张量训练过渡到自定义 Dataset 与小批量训练。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [mlp_test.py](experiments/mlp_test.py) | 4→8→2 MLP、全批量回归、MSE、Adam、损失曲线 |
| 2 | [mlp_addDataset.py](experiments/mlp_addDataset.py) | 自定义 Dataset、DataLoader、batch_size=2 与最终预测 |

[历史图片说明](artifacts/README.md)。`mlp_test.py` 现有代码每轮使用整个三样本张量，尽管保存文件名含 `multiple_batches`。代码和文件名均保留，不据此声称它使用了 DataLoader。从根目录运行会把 PNG 写到根目录，而不是历史 artifacts 文件夹。
