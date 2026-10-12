# 张量与设备

从环境检查开始，认识张量属性、运算与形状变化。

| 建议阅读顺序 | 文件 | 学习内容 |
| --- | --- | --- |
| 1 | [test_torch.py](experiments/test_torch.py) | PyTorch 版本、CUDA 可用性与 GPU 张量 |
| 2 | [tensor_test.py](experiments/tensor_test.py) | 张量创建、shape、dtype、device |
| 3 | [tensor_operation.py](experiments/tensor_operation.py) | 随机张量、逐元素运算与设备迁移 |
| 4 | [tensor_shape.py](experiments/tensor_shape.py) | reshape、unsqueeze 与维度观察 |

前两个涉及 GPU 操作的脚本是 `test_torch.py` 和 `tensor_operation.py`，它们直接调用 CUDA，运行需要可用的 CUDA 环境。其他练习可从仓库根目录用 Python 执行。
