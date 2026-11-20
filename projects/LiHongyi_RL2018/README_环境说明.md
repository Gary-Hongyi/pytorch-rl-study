> 历史阶段记录：本文描述初期环境准备。当前算法已实现，运行限制与结果见仓库根目录 README；文中的 modern/ 等旧路径已不适用于当前结构。

# 李宏毅 2018 MLDS HW4：阶段 1 环境说明

本目录仅准备环境与兼容接口，没有实现 Policy Gradient / DQN，没有调用 Agent.train()，没有模型训练或成绩验收。

## 目录与原版来源

- `original-2018/`：官方仓库完整 Git clone，包含上游自带的 Python 3.6 缓存文件。不要在这里编辑。
- `original-2018-source.zip`：`git archive HEAD` 导出的全部上游已跟踪文件，方便脱离嵌套 Git 仓库保存与解压。
- `original-sha256.json`：全部 16 个上游已跟踪文件的 SHA256。
- `modern/`：本次新增的环境兼容代码、复现脚本和测试报告。

官方来源：https://github.com/mldsta/mlds-2018-hw4

固定上游提交：`c41312185c4923876e55cc60fa3d8ff702cb04e6`。没有执行 Git commit / push。现有 PyTorch、CartPole 等学习实验未改动。

## 环境与安装

新建 Conda 环境 `mlds2018-hw4`，解释器为：

`<conda-env>\python.exe`

与 `torch-rl-study` 独立，安装命令均指向新环境解释器。VS Code 可对本次运行手动选择这个解释器；未修改项目全局编辑器设置。

核心版本和实际检测结果以 `modern/smoke_results.json` 为准，完整 Python 依赖以 `modern/requirements-lock.txt` 为准，Conda 精确包 URL 记录在 `modern/conda-explicit.txt`。

| 组件 | 用途 |
|---|---|
| Python 3.11.17 | 独立运行时 |
| PyTorch 2.11.0 CPU | 后续网络与自动微分，本阶段仅做 CPU 张量运算检查 |
| Gymnasium 1.4.0 | reset/step、空间定义与 TimeLimit |
| ALE 0.12.1 | Atari 模拟器以及随包提供的 Pong/Breakout ROM |
| NumPy 2.4.6 | 图像数组 |
| OpenCV 5.0.0.93 | RGB 转灰度和缩放 |
| SciPy 1.17.1 | 让未修改的官方 PG 模板正常 import；不恢复旧 imresize |

ALE 从 0.9 起随 PyPI 包提供 ROM，本次无需 AutoROM 或额外 ROM 下载。安装机制见 [ALE release notes](https://ale.farama.org/release_notes/index.html)。

### 日常复测（PowerShell）

```powershell
cd 'projects\LiHongyi_RL2018'
conda activate mlds2018-hw4
python -B modern/smoke_test.py --steps 1000 --output modern/smoke_results-recheck.json
```

`-B` 禁止产生 Python 缓存，避免导入官方模板时向原版目录写入文件。报告名可自行换成新名字。

不依赖 Conda 激活也可以：

```powershell
& '<conda-env>\python.exe' -B modern/smoke_test.py --output modern/smoke_results-recheck.json
```

### 在另一台机器复现

安装 Windows x64 Conda 后，在本目录运行 `./modern/setup.ps1`。脚本遇到已有同名环境会停止，避免覆盖。它创建 Python 3.11.17，并从锁定文件安装依赖，然后运行 Smoke Test。阶段 1 使用 CPU wheel，不依赖 GPU 驱动或本机 CUDA Toolkit。

机器有 RTX 5060 系列 GPU，但官方 CUDA 12.8 wheel（2.75 GB）下载长时间无进展，额外连接检查出现 connection reset，因此本阶段改装 CPU 版。环境 Smoke Test 不需要 GPU；这不影响两个游戏的环境验证。阶段 2 开始训练前可再规划 GPU 安装，本阶段没有验证 GPU 加速。

初次部署时用户配置的 Conda 镜像元数据获取较慢，中断期间造成不完整缓存；最终使用独立缓存 `<conda-package-cache>` 和官方 main channel 成功部署。未删除或覆盖原有 Conda 缓存，也未修改 Conda 全局配置。

## 官方代码怎样连接起来

原版 `main.py` 读取参数，建立 `Environment`，把它传给 `Agent_PG` 或 `Agent_DQN`；训练入口随后调用 `agent.train()`。本阶段不执行这个入口。

原版 `test.py` 的循环是：`env.reset()` 获取初始 state → `agent.init_game_setting()` 清理单局状态 → `agent.make_action(state, test=True)` 输出动作 → `env.step(action)` 返回下一张图、reward、done、info → 更新 state，直到结束。

模板中的 `make_action()` 本来就返回随机动作，因此可以检查这些接口是否连通，无需填写任何 `YOUR CODE HERE`。`smoke_test.py` 在内存中让官方模块的 `environment` import 指向现代适配层，读取原版 Agent 和 test 函数；每个 Agent 只跑一局，最多 64 次交互。未复制或修改 Agent 模板，也未调用 train()。这个短循环的均值没有算法成绩意义。

现代环境循环直接使用 `make_env()`：`obs, info = reset(seed=...)`，`obs, reward, terminated, truncated, info = step(action)`。episode 在 `terminated or truncated` 时停止。将来训练计算目标时，要按算法分别处理真正终止与时间截断，不能简单把二者的学习语义混为一谈。

## 旧版与现代兼容

| 原版 2018 | 现代接口 | 本次方案 |
|---|---|---|
| `import gym` | `import gymnasium as gym`，注册 ALE | 仅现代目录使用新包 |
| `Pong-v0` / `BreakoutNoFrameskip-v4` | `ALE/Pong-v5` / `ALE/Breakout-v5` | 显式映射，并配置帧跳与 sticky action |
| `reset()` 返回 obs | 返回 `(obs, info)` | 桥接层返回 obs，保存 reset info |
| `step()` 返回四项、done | 五项，terminated 与 truncated 分开 | 桥接层合并 done，原始标志保存在 info；现代接口保留五项 |
| `env.seed(seed)` | `reset(seed=seed)` | seed 暂存至下次 reset；动作空间单独设 seed |
| `_step/_reset/_observation/_reward` | 公共 `step/reset/observation/reward` | 独立重写环境包装流程 |
| `np_random.randint` | Generator 的 `integers` | 现代 noop 包装器使用 integers |
| `scipy.misc.imresize` | 已删除 | `preprocess_pg_rgb` 用 OpenCV，明确输出灰度 `(80,80,1)` |
| PyTorch 0.3 时代的 Variable、`.data[0]`、volatile | Tensor 原生自动微分、`.item()`、`torch.no_grad()` | 仅说明迁移点；官方模板尚无网络实现，不需要修改算法 |

官方 PG 预处理函数的注释声称灰度输出，但原函数未明确做 RGB→灰度，不能盲信其注释。桥接测试不会调用它；现代替代函数单独验证形状。原版环境注释中的 Pong 奖励说明也不应代替实际游戏规则与实测。

## 图像与游戏参数

Pong 不使用 DQN 包装器，观察为 `(210,160,3)` uint8 RGB。显式设置 frameskip=(2,5)、repeat_action_probability=0、最小动作集。

Breakout 原始环境也可单独测试。作业包装路径设置底层 frameskip=1，然后由 MaxAndSkip 重复动作 4 帧并对末两帧取最大值，避免重复跳帧。reset 时做随机 NOOP；掉命视为包装器 episode 结束，真 game over 才完整 reset；FIRE 启动游戏；最后灰度、84×84、float32 /255、最近四帧沿最后轴堆叠。因此输入为 `(84,84,4)` float32，范围 [0,1]。测试配置不裁剪 reward；训练配置仅测试包装器的 sign 裁剪，并不训练模型。`info.raw_reward` 留存裁剪前奖励；`life_lost/real_done` 区分掉命与真正 game over。

将来给 PyTorch CNN 输入前需将 HWC 变为 CHW，再添加 batch 维度。本阶段未创建 CNN。

现代 ALE、随机数生成器、时间限制和部分终止帧处理与 2018 实现存在差异；现代结果不能直接当作满足官方基线。本次仅确认环境及接口可用。

## 测试与下一步

2026-10-13 实测：7 项环境/桥接检查全部通过，PyTorch CPU 张量运算通过，`pip check` 输出 `No broken requirements found`。原版 Git 工作区干净，16 个文件的 SHA256 校验全部一致。

| 配置 | reset / 1000 步随机交互 | Observation | 实测 reward | terminated 次数 |
|---|---|---|---|---|
| Pong 原始 RGB | 通过 | (210,160,3) uint8 | -1, 0, 1 | 0 |
| Breakout 原始 RGB | 通过 | (210,160,3) uint8 | 0, 1 | 4 |
| Breakout 作业包装，测试配置 | 通过 | (84,84,4) float32 | 0, 1 | 17（含掉命） |
| Breakout 作业包装，训练配置，仅随机交互 | 通过 | (84,84,4) float32 | 0, 1 | 17（含掉命） |

Pong 在 1000 步内未自然结束完整一局，返回的两个标志类型正常；额外一帧 TimeLimit 检查确认 truncated=True。原版模板桥接各最多 64 步，仅检查调用流程。以上不构成完整 episode 评分、策略效果或学习能力验证。所有配置的 RGB 数组渲染通过，形状 (210,160,3)。CPU 版的 cuda_available=False 为预期结果。

`modern/smoke_results.json` 记录四种配置各 1000 步的 observation、动作含义、reward 集合、结束次数与 RGB render 结果；还包括原版两个模板的短循环，以及 max_episode_steps=1 强制触发 truncated 的检查。没有调用 human 窗口；rgb_array 渲染验证不代表窗口显示已验证。

进入 HW4-1 前，先读 `modern/environment.py` 的 `make_env/reset/step`，再读原版 `agent_pg.py` 的四个待实现接口。重点理解：RGB 图像与预处理、单帧和帧差、动作编号与游戏动作、episode 边界、轨迹中 state/action/reward 的对应关系。随后再把已学过的 Return、log probability、Policy Gradient loss 接到这些接口上。

阶段 1 到此为止；阶段 2 算法实现需要你另行确认。
