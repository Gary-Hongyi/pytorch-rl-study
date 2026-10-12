# 历史 MLP 图片

| 文件 | 观察与来源 |
| --- | --- |
| [mlp_training_curve.png](figures/mlp_training_curve.png) | 2026-09-10 文件时间；图中约 200 轮，损失下降 |
| [mlp_training_multiple_batches1.png](figures/mlp_training_multiple_batches1.png) | 2026-09-10 文件时间；图中约 1,000 轮，损失下降 |

两张图片来自原 `Pytorch/` 目录，保留原名和原始字节。文件时间不等于已验证的训练时间。

当前 `mlp_test.py` 配置为 1,000 轮，保存第二个名称，且使用全批量训练。但脚本后来有修改，因此无法确认现有图片对应的确切源码版本、随机种子或训练配置。第一张图对应的旧版本未在 Git 中保存。保留这些不确定性，不补写虚构的复现信息。

重跑脚本会将输出写到当前工作目录，可能覆盖该目录中的同名 PNG；不会自动定位这里的历史图片。
