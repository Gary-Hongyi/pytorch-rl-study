> 历史阶段记录：本文描述初期环境准备。当前算法已实现，运行限制与结果见仓库根目录 README；文中的 modern/ 等旧路径已不适用于当前结构。

# HW4-2 Breakout：现代环境准备

本阶段只准备环境。没有 Q-Network、Replay Buffer、TD Loss、Target Network 或训练循环；`agent_dqn.py` 是后续学习入口。没有改动 `original-2018/`、`modern/`、`modern_pg_improved/`。

## 复测命令

在项目根目录 PowerShell 运行：

```powershell
& '<conda-env>\python.exe' -B projects/LiHongyi_RL2018/modern_dqn/smoke_test.py
& '<conda-env>\python.exe' -m pip check
```

或 `conda activate mlds2018-hw4` 后，用 `python -B` 执行同一个脚本。`-B` 避免产生缓存。本目录可独立运行，不依赖 PG 实验的导入路径、网络、Checkpoint 或结果。

现有环境：Python 3.11.17、PyTorch 2.11.0+cpu、Gymnasium 1.4.0、ALE 0.12.1、NumPy 2.4.6、OpenCV 5.0.0.93。此次没有安装、升级或重装依赖。完整现状快照为 `requirements-lock.txt`；其中可能含 PG 绘图依赖，Breakout 环境模块本身只依赖 ALE/Gymnasium/NumPy/OpenCV。PyTorch 在 Smoke Test 中仅用于输入布局转换和 CPU 张量检查。

## 原版文件核对

实际读取了 `original-2018/hw4/README.md`、`agent_dir/agent_dqn.py`、`environment.py`、`atari_wrapper.py`、`main.py`、`test.py`、`argument.py`、`requirement.txt`，并查看了基础 Agent。

| 原始文件 | 已提供的内容 / 待实现内容 |
|---|---|
| README / requirement.txt | 旧 Gym、Atari、OpenCV 的安装及命令；未锁定版本，也未提供完整 PyTorch 依赖 |
| main.py | DQN 使用 `BreakoutNoFrameskip-v4`，打开 Atari Wrapper，构造 Agent 后调用 train；测试则调用 test 函数 |
| agent_dqn.py | 四个接口模板；train 为 pass，make_action 返回随机动作，无网络、replay、TD 更新或 target 同步 |
| environment.py | 封装动作合法性检查、reset、四项 step、seed；根据 test 决定是否裁剪奖励 |
| atari_wrapper.py | 提供完整图像处理和环境包装链，是本次移植的行为依据 |
| test.py | seed=11037；调用 make_action(state, test=True)，累计未裁剪奖励；DQN 外部指定 100 episodes |
| argument.py | 空的参数扩展函数，注释中的 lr=0.01 只是通用示例，不是 DQN 指定学习率 |

用户给出的 DQN/epsilon-greedy/replay/target、参考参数和 >40 标准记录在 `reference_config.json`，这些并非空模板已经实现的能力。gamma 和 epsilon 衰减总步数尚未给定，不自行填默认值。online 每 4 agent steps 更新与一个 action 重复 4 emulator frames 是两个不同层面的计数。

## 包装器顺序与状态

底层 ALE：`ALE/Breakout-v5`，显式 `frameskip=1`、`repeat_action_probability=0.0`、最小离散动作集、RGB Observation。避免 ALE v5 默认 sticky action / 跳帧与官方 NoFrameskip 用法冲突。

随后按原版顺序包裹：

1. NoopReset：完整 reset 后随机执行 1..30 次 NOOP。
2. MaxAndSkip：动作重复 4 原始帧、奖励求和、末两帧 max pooling。
3. EpisodicLife：掉命产生 episode 边界；非真正 game over 的 reset 只执行 NOOP 继续游戏。
4. FireReset：reset 后依次执行动作 1、2（此时各经过 MaxAndSkip）。
5. WarpFrame：RGB→灰度，以 OpenCV INTER_AREA 缩放为 84×84，添加单通道维度。
6. ScaledFloatFrame：float32 /255。
7. 训练配置 ClipReward：对四帧奖励之和取 sign；测试配置不裁剪。info.raw_reward 保留裁剪前累计值。
8. FrameStack：最近四帧沿最后轴堆叠；reset 初始重复同一帧四次。

输入最终为 `(84,84,4)`、float32、范围 [0,1]，没有裁剪画面计分板，也没有引入新的动作子集。后续 PyTorch CNN 输入只需布局转换为 `(batch,4,84,84)`，不能再 /255 一次。

动作实测严格检查为：0=NOOP、1=FIRE、2=RIGHT、3=LEFT。合法空间为 Discrete(4)，动作 4 会被拒绝。

## API、seed 与 Episode 语义

`make_breakout_env(test=False, seed=42)` 返回现代五返回值环境：

```python
from environment import make_breakout_env

env = make_breakout_env(test=False)
try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
finally:
    env.close()
```

此片段用于在本目录的脚本中导入。默认仅首次 reset 应用 seed=42，后续普通 reset 不重复播种。动作空间也单独播种。显式 `reset(seed=42)` 则强制开始一个可复现新游戏，即使当前处于掉命边界；这是现代 API 扩展。神经网络初始化和训练采样等随机性待后续算法阶段另行设置，此次只验证环境/动作采样复现。

`terminated` 包括 EpisodicLife 人工掉命边界；info 中额外保留 `life_lost`、`real_terminated`、`real_truncated`、`real_done`。`truncated` 保持独立；将来 replay / TD target 需明确区分这两种标志。本次没有实现任何 TD mask。

`Environment` 类提供原版四返回值桥接：done=terminated or truncated，两标志保存在 info；保留原版构造函数和 get_random_action 等接口。它不允许禁用 Wrapper。

**官方测试模式仍然使用 EpisodicLife。**原版 `test.py` 按 Wrapper 的 done 计数，因此 100 episodes 不应被擅自改成 100 个完整多命游戏。现代兼容层保持这一点；本次没有执行 100 episodes 评分，也没有模型成绩。

默认工程 seed=42 按本次要求设置，与原版 test.py 固定的 11037 有区别；将来复现官方测试条件可调用 bridge.seed(11037) 或 make_breakout_env(test=True, seed=11037)，不要修改原始 test.py。

## 兼容差异与确认范围

| 2018 实现 | 本目录处理 | 影响 |
|---|---|---|
| Gym `_reset/_step/_observation/_reward` | Gymnasium 公共方法，reset 两项、step 五项 | 接口移植，未取消任何 Wrapper |
| RNG randint | Generator.integers | 采样范围相同；不保证旧 NumPy 同 seed 轨迹一致 |
| BreakoutNoFrameskip-v4 | ALE/Breakout-v5 加显式参数 | 实测 4 个合法动作；底层模拟器版本不等同于历史版本 |
| Warp/Scale 的 Box 声明不准确 | Box 明确 float32、[0,1]、HWC4 | 修正元数据，实际状态数值沿用原版 |
| FrameStack 返回 LazyFrames，Environment 随即 np.array | 直接返回 NumPy 堆叠 | 状态值/布局一致；不提供 LazyFrames 存储优化，后续 replay 内存设计另议 |
| MaxAndSkip 提前终止时可能复用旧缓冲帧 | 原样保留更新第 3/4 帧的规则 | 不擅自修复会改变终止图像的旧行为 |
| FireReset 在 action=2 结束后可能返回 reset 前的图像 | 保留该规则 | 不声称能修复历史代码所有问题 |
| 原版 Gym/ALE 依赖未锁版本，未列时间限制 | 使用已安装 ALE 的 108000 raw-frame 上限；不添加 27000-step TimeLimit | 历史时间限制无法由本仓库确认，不能声称完全等价 |

未直接复用 `modern/environment.py`：其 max skip 在提前终止时取可用末两帧，而原版缓冲更新规则不同；它还显式施加 27000 底层 step 限制。本目录从官方源码独立移植，避免连带改变 PG，也避免带入这些差异。只复用现有 Conda 依赖。

## 验证内容与证据

`smoke_test.py` 包含五组检查：

- 原版参考对比：只在内存中给原版源码提供旧 Gym shim，对同一个可控模拟器输入比较训练/测试配置各 40 步的图像、奖励、done、原始动作和 reset 次数。
- 可控奖励测试：4 原始帧奖励和为 +5 时，训练返回 +1，info 保留 +5；确认每动作重复四帧。
- 可控掉命与截断测试：掉命不完整 reset；truncated 独立且使下一次完整 reset；close 正常转发。
- 真实 ALE 测试：训练/测试环境 seed=42，各 256 步随机交互，状态逐元素相同、动作采样相同、奖励满足裁剪关系；检查初始帧复制、堆叠移动、非法动作拒绝、RGB render 与 close。
- 官方 API 桥接 / PyTorch 输入：确认四返回值、状态转成 `(1,4,84,84)`，无网络或反向传播。

其中原版源码运行时的 NumPy 2 `LazyFrames.__array__` 废弃警告来自原版只接受 dtype 参数的定义，不影响比较结果；现代环境直接返回数组，无此接口问题。参考比较验证包装器移植，不验证旧 ALE 模拟器。真实随机轨迹若只出现 0/1 奖励，不足以单独证明 clipping，所以另用 +5 奖励专门验证。

实际测试输出与版本记录在 `smoke_results.json`；受保护文件的前后校验记录在 `verification.json`。这些检查均为短交互，没有训练。

## 下一步

先打开 `agent_dqn.py`。从理解 Q-Network 的四个动作输出和输入布局开始，再逐步实现 Replay Buffer、epsilon-greedy、TD Loss 和 Target Network。`train()` 与 `make_action()` 当前会明确抛出 NotImplementedError，防止把未实现逻辑误当成可训练的 Agent。
