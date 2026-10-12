# 自动求导与优化

建议对照阅读四个脚本，理解梯度计算与参数更新的关系。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [auto_grad1.py](experiments/auto_grad1.py) | 两次 backward 的梯度累积与清零 |
| 2 | [update_w.py](experiments/update_w.py) | 在 no_grad 中手动更新参数 |
| 3 | [opitimizer.py](experiments/opitimizer.py) | 使用 SGD 执行一次参数更新 |
| 4 | [autograd_test](experiments/autograd_test) | 重复执行 zero_grad、backward、step |

`opitimizer.py` 保留原始拼写；`autograd_test` 保留无扩展名形式，可直接使用 `python` 执行。它们是不同学习阶段，不作为重复文件合并。
