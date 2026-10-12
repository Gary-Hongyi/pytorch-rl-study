# 2026-10-13 迁移记录

先创建目录并移动文件，随后核对全部原文件的大小与 SHA-256，最后新增 README。未修改任何 Python 代码，未执行训练，未重命名任何原文件，未删除文件，未新增 Git 提交。

- 迁移前共有 36 个文件（不计 .git 内部文件）。
- 移动 27 个：22 个非空学习脚本、1 个空占位文件、2 张图片、2 个模型。
- 原位保留 9 个：8 个 MNIST 数据文件和 .vscode/settings.json。
- 全部 36 个原文件通过大小与 SHA-256 一致性检查。
- Git 在迁移前没有提交、没有已跟踪文件。不能依靠既有 Git 历史还原；本表记录原路径和文件时间。
- 原 Pytorch/、RL/ 及其子目录的空目录留在本地，未执行目录删除；Git 不跟踪空目录。

## 移动清单

路径相对仓库根目录。每个文件保持原始 basename。

| 原路径 | 新路径 |
| --- | --- |
| `Pytorch/test_torch.py` | [01_pytorch_basics/01_tensors_and_devices/experiments/test_torch.py](../../01_pytorch_basics/01_tensors_and_devices/experiments/test_torch.py) |
| `Pytorch/tensor_test.py` | [01_pytorch_basics/01_tensors_and_devices/experiments/tensor_test.py](../../01_pytorch_basics/01_tensors_and_devices/experiments/tensor_test.py) |
| `Pytorch/tensor_operation.py` | [01_pytorch_basics/01_tensors_and_devices/experiments/tensor_operation.py](../../01_pytorch_basics/01_tensors_and_devices/experiments/tensor_operation.py) |
| `Pytorch/tensor_shape.py` | [01_pytorch_basics/01_tensors_and_devices/experiments/tensor_shape.py](../../01_pytorch_basics/01_tensors_and_devices/experiments/tensor_shape.py) |
| `Pytorch/auto_grad1.py` | [01_pytorch_basics/02_autograd_and_optimization/experiments/auto_grad1.py](../../01_pytorch_basics/02_autograd_and_optimization/experiments/auto_grad1.py) |
| `Pytorch/update_w.py` | [01_pytorch_basics/02_autograd_and_optimization/experiments/update_w.py](../../01_pytorch_basics/02_autograd_and_optimization/experiments/update_w.py) |
| `Pytorch/opitimizer.py` | [01_pytorch_basics/02_autograd_and_optimization/experiments/opitimizer.py](../../01_pytorch_basics/02_autograd_and_optimization/experiments/opitimizer.py) |
| `Pytorch/autograd_test` | [01_pytorch_basics/02_autograd_and_optimization/experiments/autograd_test](../../01_pytorch_basics/02_autograd_and_optimization/experiments/autograd_test) |
| `Pytorch/weight_bias.py` | [01_pytorch_basics/03_linear_layers/experiments/weight_bias.py](../../01_pytorch_basics/03_linear_layers/experiments/weight_bias.py) |
| `Pytorch/layer_hand.py` | [01_pytorch_basics/03_linear_layers/experiments/layer_hand.py](../../01_pytorch_basics/03_linear_layers/experiments/layer_hand.py) |
| `Pytorch/nn_test` | [01_pytorch_basics/03_linear_layers/experiments/nn_test](../../01_pytorch_basics/03_linear_layers/experiments/nn_test) |
| `Pytorch/nn_test2` | [01_pytorch_basics/03_linear_layers/experiments/nn_test2](../../01_pytorch_basics/03_linear_layers/experiments/nn_test2) |
| `Pytorch/mlp_test.py` | [02_deep_learning/01_mlp_and_data_loading/experiments/mlp_test.py](../../02_deep_learning/01_mlp_and_data_loading/experiments/mlp_test.py) |
| `Pytorch/mlp_addDataset.py` | [02_deep_learning/01_mlp_and_data_loading/experiments/mlp_addDataset.py](../../02_deep_learning/01_mlp_and_data_loading/experiments/mlp_addDataset.py) |
| `Pytorch/classfication_test.py` | [02_deep_learning/02_classification/experiments/classfication_test.py](../../02_deep_learning/02_classification/experiments/classfication_test.py) |
| `Pytorch/mnist_test.py` | [02_deep_learning/03_mnist_mlp/experiments/mnist_test.py](../../02_deep_learning/03_mnist_mlp/experiments/mnist_test.py) |
| `Pytorch/mnist_test_load.py` | [02_deep_learning/03_mnist_mlp/experiments/mnist_test_load.py](../../02_deep_learning/03_mnist_mlp/experiments/mnist_test_load.py) |
| `Pytorch/cnn_test1.py` | [02_deep_learning/04_cnn/experiments/cnn_test1.py](../../02_deep_learning/04_cnn/experiments/cnn_test1.py) |
| `Pytorch/cnn_mnist_train.py` | [02_deep_learning/04_cnn/experiments/cnn_mnist_train.py](../../02_deep_learning/04_cnn/experiments/cnn_mnist_train.py) |
| `RL/1.Gymnasium Basics/first_cartpole.py` | [03_reinforcement_learning/01_gymnasium_basics/experiments/first_cartpole.py](../../03_reinforcement_learning/01_gymnasium_basics/experiments/first_cartpole.py) |
| `RL/RL2018/pg_cartpole.py` | [03_reinforcement_learning/02_policy_gradient/experiments/pg_cartpole.py](../../03_reinforcement_learning/02_policy_gradient/experiments/pg_cartpole.py) |
| `RL/RL2018/pg_cartpole_multiple_episode.py` | [03_reinforcement_learning/02_policy_gradient/experiments/pg_cartpole_multiple_episode.py](../../03_reinforcement_learning/02_policy_gradient/experiments/pg_cartpole_multiple_episode.py) |
| `Pytorch/mlp_training_curve.png` | [02_deep_learning/01_mlp_and_data_loading/artifacts/figures/mlp_training_curve.png](../../02_deep_learning/01_mlp_and_data_loading/artifacts/figures/mlp_training_curve.png) |
| `Pytorch/mlp_training_multiple_batches1.png` | [02_deep_learning/01_mlp_and_data_loading/artifacts/figures/mlp_training_multiple_batches1.png](../../02_deep_learning/01_mlp_and_data_loading/artifacts/figures/mlp_training_multiple_batches1.png) |
| `Pytorch/mnist_model.pth` | [02_deep_learning/03_mnist_mlp/artifacts/checkpoints/from_pytorch_dir/mnist_model.pth](../../02_deep_learning/03_mnist_mlp/artifacts/checkpoints/from_pytorch_dir/mnist_model.pth) |
| `mnist_model.pth` | [02_deep_learning/03_mnist_mlp/artifacts/checkpoints/from_repo_root/mnist_model.pth](../../02_deep_learning/03_mnist_mlp/artifacts/checkpoints/from_repo_root/mnist_model.pth) |
| `Pytorch/mlp_test1` | [archive/placeholders/mlp_test1](../../archive/placeholders/mlp_test1) |

## 迁移前文件记录

时间为本地文件系统提供的时间，不是经过验证的实验运行日期。原路径列可与移动清单对应；不在移动清单中的文件保持原位置。校验记录用于后续检查来源和识别变化。

| 原路径 | 字节数 | 创建时间 | 修改时间 | SHA-256 |
| --- | ---: | --- | --- | --- |
| `mnist_model.pth` | 409605 | 2026-10-11 10:11:15 | 2026-10-11 10:11:15 | `8BB6D17F3DD20914CD65D1CAC849B0F69C8F63CDF473FB92CE1656500839C5D5` |
| `.vscode/settings.json` | 738 | 2026-09-01 14:59:03 | 2026-09-05 10:45:57 | `373FE9A304476A0DA149E81BB6331B572DC8C24110CEB9F9F6D6AD5D28C16052` |
| `data/MNIST/raw/t10k-images-idx3-ubyte` | 7840016 | 2026-09-20 21:22:28 | 2026-09-20 21:22:28 | `0FA7898D509279E482958E8CE81C8E77DB3F2F8254E26661CEB7762C4D494CE7` |
| `data/MNIST/raw/t10k-images-idx3-ubyte.gz` | 1648877 | 2026-09-20 21:22:27 | 2026-09-20 21:22:28 | `8D422C7B0A1C1C79245A5BCF07FE86E33EEAFEE792B84584AEC276F5A2DBC4E6` |
| `data/MNIST/raw/t10k-labels-idx1-ubyte` | 10008 | 2026-09-20 21:22:29 | 2026-09-20 21:22:29 | `FF7BCFD416DE33731A308C3F266CC351222C34898ECBEAF847F06E48F7EC33F2` |
| `data/MNIST/raw/t10k-labels-idx1-ubyte.gz` | 4542 | 2026-09-20 21:22:29 | 2026-09-20 21:22:29 | `F7AE60F92E00EC6DEBD23A6088C31DBD2371ECA3FFA0DEFAEFB259924204AEC6` |
| `data/MNIST/raw/train-images-idx3-ubyte` | 47040016 | 2026-09-20 21:22:22 | 2026-09-20 21:22:23 | `BA891046E6505D7AADCBBE25680A0738AD16AEC93BDE7F9B65E87A2FC25776DB` |
| `data/MNIST/raw/train-images-idx3-ubyte.gz` | 9912422 | 2026-09-20 21:22:20 | 2026-09-20 21:22:22 | `440FCABF73CC546FA21475E81EA370265605F56BE210A4024D2CA8F203523609` |
| `data/MNIST/raw/train-labels-idx1-ubyte` | 60008 | 2026-09-20 21:22:25 | 2026-09-20 21:22:25 | `65A50CBBF4E906D70832878AD85CCDA5333A97F0F4C3DD2EF09A8A9EEF7101C5` |
| `data/MNIST/raw/train-labels-idx1-ubyte.gz` | 28881 | 2026-09-20 21:22:25 | 2026-09-20 21:22:25 | `3552534A0A558BBED6AED32B30C495CCA23D567EC52CAC8BE1A0730E8010255C` |
| `Pytorch/auto_grad1.py` | 201 | 2026-09-03 10:15:39 | 2026-09-03 10:35:29 | `CDC138C40334BEAD9FC48C85EC40E063A4B9FAEB6D0DC0B4B65957C59D887475` |
| `Pytorch/autograd_test` | 282 | 2026-09-03 11:26:58 | 2026-09-03 11:36:01 | `ABF13A88D3D89F7359B2E6D82CBCE758655595C978FF4B996DD888AB437EE5E3` |
| `Pytorch/classfication_test.py` | 1257 | 2026-09-18 19:53:59 | 2026-09-18 20:34:35 | `F54BCE19929F57C04FAAFC4C2E09A768F055C954F664DEEA57DB6AEF5F61C22E` |
| `Pytorch/cnn_mnist_train.py` | 5214 | 2026-09-24 10:44:38 | 2026-09-24 11:33:23 | `7590D2CEA25CA40F1582F659E3E5DD0F41056D16E3BB19D8A30C42501F807262` |
| `Pytorch/cnn_test1.py` | 1362 | 2026-09-22 16:54:49 | 2026-09-22 16:57:52 | `8593F9F1F56231723578A075305BD5D6060BCF2C662C881AEEEF0D32FE0A8B25` |
| `Pytorch/layer_hand.py` | 503 | 2026-09-05 11:16:20 | 2026-09-05 11:24:21 | `F35A14CDD1A33F7F8BE9FA78B652BF3CC6494904347FA2DBB46D716638C1DB39` |
| `Pytorch/mlp_addDataset.py` | 2056 | 2026-09-17 09:52:37 | 2026-09-18 22:54:21 | `813C14ECB57AC8A479D17E8D5279A9A3B8DD5122BD92B584112D58B9830311E2` |
| `Pytorch/mlp_test.py` | 1185 | 2026-09-09 20:34:23 | 2026-09-21 16:07:29 | `42DCDD22595590E7D63698905F67A3304AFA2CF0674DEBDD34B15A53A6237EF6` |
| `Pytorch/mlp_test1` | 0 | 2026-09-09 20:33:47 | 2026-09-09 20:33:47 | `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855` |
| `Pytorch/mlp_training_curve.png` | 34057 | 2026-09-10 11:02:20 | 2026-09-10 11:02:20 | `BC65B201C31F313123165A7B152ADBF32C239D4F8D9C7DBB929A5F1ADB7D3D46` |
| `Pytorch/mlp_training_multiple_batches1.png` | 29907 | 2026-09-10 11:32:24 | 2026-09-10 11:32:24 | `D7A47D2A4AE0D0CA9108860B09229044D630B558D414191BA949CE3EF23F3689` |
| `Pytorch/mnist_model.pth` | 409605 | 2026-09-21 10:47:50 | 2026-09-21 10:47:50 | `56D86628FC4A2EE6805D16BCD701C7D52429178F888C8D07906E6C7A9599E332` |
| `Pytorch/mnist_test_load.py` | 1534 | 2026-09-20 21:02:32 | 2026-09-21 10:51:39 | `1BC18BAB4A492EF186824A40CF32068F17D8C1F67BE0DBA58AF6C6676D0123F3` |
| `Pytorch/mnist_test.py` | 3009 | 2026-09-21 10:52:41 | 2026-09-21 10:53:14 | `6BD5D56AC9949F67C99C7227E9C699C9B940C61030FDFD0F6E67C210C75CCCD1` |
| `Pytorch/nn_test` | 516 | 2026-09-05 14:21:53 | 2026-09-18 22:54:21 | `A6221538C24031567F119964DBC6F2B31013613279B367D19F5F21F29CBABF49` |
| `Pytorch/nn_test2` | 585 | 2026-09-05 16:20:09 | 2026-09-05 16:40:29 | `0941CC3E777F89CD53FB6247F90969CC78DB6B2F8441F3D75C9542B347C4B767` |
| `Pytorch/opitimizer.py` | 196 | 2026-09-03 11:14:58 | 2026-09-03 11:15:06 | `9D23178A242A648F28D3911AF9C9861E0091EEB37DDBF25AAA759BB929C85D7F` |
| `Pytorch/tensor_operation.py` | 212 | 2026-09-01 16:09:03 | 2026-09-01 16:16:55 | `36BFF3A08DEC836FFD1963865158AC0BE4353B46A64244EC5650F05C91DD6CA6` |
| `Pytorch/tensor_shape.py` | 345 | 2026-09-01 16:23:29 | 2026-09-01 17:42:28 | `566CF5ACBB4D61BF280A29183F1E1370931FEFC6FC371B4E8D40390FC9F137D2` |
| `Pytorch/tensor_test.py` | 127 | 2026-09-01 16:00:55 | 2026-09-01 16:01:57 | `CD486EEE2E79D46B3955A06EF0102FCFBABA1C64A16242CF4793ACC638DC7F6A` |
| `Pytorch/test_torch.py` | 124 | 2026-09-01 15:01:27 | 2026-09-01 15:02:04 | `C452C7DAD01CCB7A6C91871E254D3C26BC4B37CE5AA32BD1AE9171E66A606AAF` |
| `Pytorch/update_w.py` | 177 | 2026-09-03 10:45:25 | 2026-09-03 10:46:08 | `E1CAFA7D1CC61BBBBDDD000BE7F5BCA3FC1C7A5736A4116A1A3B48A62F0FE198` |
| `Pytorch/weight_bias.py` | 244 | 2026-09-05 10:30:59 | 2026-09-05 11:09:48 | `1D2149788EF8E6F998CB183F46F4D749B768212342002E056F10DB1F588C6EE7` |
| `RL/1.Gymnasium Basics/first_cartpole.py` | 722 | 2026-09-28 20:18:40 | 2026-09-28 20:34:27 | `AC68DC2360B8ED63430C7CFB54ABBE4889E646F570FC502C6A10D210D254A5C8` |
| `RL/RL2018/pg_cartpole_multiple_episode.py` | 3060 | 2026-10-12 19:54:06 | 2026-10-12 20:17:00 | `CD99F6D4CF8439BB32D9866A1789D1DD0A763621FD877F6EAD3DD552AC998776` |
| `RL/RL2018/pg_cartpole.py` | 1884 | 2026-10-11 10:06:13 | 2026-10-12 19:42:51 | `8CCF23B30E42E3E54C2C5E8CAE9EAE0655AC26B72B0A1BE36128F14D061C8CDC` |

## 已知影响

移动不会改变 Python 相对路径的解析规则：路径仍相对于启动程序时的工作目录。MNIST 加载脚本不会自动找到已归档的模型，训练模型和 MLP 图片也不会自动保存到 artifacts。详见 [主 README](../../README.md) 与 [模型说明](../../02_deep_learning/03_mnist_mlp/artifacts/README.md)。

本次没有选择默认 checkpoint、修复代码、生成依赖文件或添加忽略规则。这些可以作为明确范围的后续改进。
