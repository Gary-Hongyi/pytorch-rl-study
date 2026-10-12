# 历史 MNIST 模型

两个文件均为 409,605 字节，但 SHA-256 不同，因此分别保留，未选择默认模型、加载比较参数或合并。

| 文件 | 原位置 | 迁移前文件时间 |
| --- | --- | --- |
| [mnist_model.pth](checkpoints/from_pytorch_dir/mnist_model.pth) | Pytorch/mnist_model.pth | 2026-09-21 |
| [mnist_model.pth](checkpoints/from_repo_root/mnist_model.pth) | 仓库根目录 mnist_model.pth | 2026-10-11 |

时间来自文件系统；两个文件的具体训练运行、随机种子、准确率及参数是否一致尚未验证。模型与现有 MNIST MLP 脚本的关联来自名称与代码上下文，不表示已逐个验证可加载。

## 路径限制

Python 代码未修改：

- 训练脚本仍以裸文件名 `mnist_model.pth` 保存到当前工作目录。
- 加载脚本仍从当前工作目录读取此名称，不会搜索本目录。
- 因原根目录模型已经归档，迁移后从根目录直接执行加载脚本会找不到文件，除非先训练生成或另行放置一个模型。
- 直接切换到这里的 checkpoint 子目录运行加载脚本，会使 `./data` 也指向该子目录，可能下载另一份 MNIST 数据。

后续若要修复，应先明确选择哪个历史 checkpoint，再单独批准路径调整；本次没有替用户选择。重跑训练时建议从仓库根目录启动，新模型会与这两个历史模型分开存放。
