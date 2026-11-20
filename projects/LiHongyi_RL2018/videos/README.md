# Atari Agent 演示视频

## Breakout Double DQN

- 视频为 `breakout_double_dqn_latest.mp4`，时长 60 秒，1280×720，60 FPS，H.264 High / avc1，yuv420p，MP4 faststart，无音轨。
- 模型来自 `../modern_double_dqn/checkpoints/double_dqn_cpu_01_latest_1750000.pt`，内部 `total_steps=1750000`。严格加载 `online` 到原 `QNetwork(action_dim=4)`。
- Checkpoint SHA-256 为 `44358f250f68ba43dd2c6de34e9cc6e38c166e93c01408bd17d5559a4132e324`，录制前后文件哈希及推理网络参数哈希一致。
- Evaluation seed 固定为 11037，模型载入后设置 Python、NumPy、Torch 和环境种子。CPU 单线程，`eval()`、`requires_grad_(False)`、`torch.inference_mode()`。
- 环境使用现有 `modern_double_dqn/environment.py` 的 `make_breakout_env(test=True, seed=42, render_mode="rgb_array")`，配合原 `Environment` 四返回值接口。显式 `env.seed(11037)` 覆盖首次 reset 的种子。原兼容名为 `BreakoutNoFrameskip-v4`，实际映射为 `ALE/Breakout-v5`，底层 frameskip=1，sticky action=0，最小动作集。
- 完整沿用 NoopReset、MaxAndSkip(4)、EpisodicLife、FireReset、灰度 84×84、float32 /255、四帧 HWC 堆叠。模型输入经原方法转换为 BCHW，动作使用原 `make_action(test=True)`，epsilon=0.025，动作编号为 NOOP / FIRE / RIGHT / LEFT。
- 录制器插入现有包装链最底层，逐次调用原底层 `step()`，然后读取原始 RGB `render()`。检查 ALE frame counter 每次恰好增加 1，逐模拟器帧采集，按 NTSC 游戏的 60 帧/秒编码。ALE Python 环境类的通用 `metadata.render_fps=30` 是另一层渲染默认值，本次采样依据底层模拟器帧数。一个四帧决策对应视频四帧。
- 210×160 原始游戏画面按最近邻放大至 630×480，等比例居中放置于 16∶9 画布，保留原生分数和生命显示。上方标注模型、训练节点、seed 和 epsilon。
- 视频选取从首次 reset 的协议 NOOP / FIRE 开始的连续 3600 个模拟器帧，包含一次掉命及原包装器的继续游戏流程，录制区间内未遇到真实 game over。reset 内的动作由原有包装器执行。最后一个四帧决策完成后共观察 3603 帧，视频截止于前 3600 帧。
- 录制前用相同 seed 检查 64 次决策，有无 RGB 渲染与采集的两条轨迹中，动作、奖励、结束标志和模型输入状态哈希逐条一致。
- OpenCV 完整解码 3600 帧，检查分辨率、FPS、游戏区域非黑以及画面变化。FFmpeg 完整解码通过，抽查 10、30、50 秒画面，球、挡板、砖块和原生分数正常。
- 演示为固定 seed 下的一次连续推理轨迹，独立评估统计保存在原 `results/` 文件中。

复现命令在项目根目录执行，脚本对已有输出采取报错保护。

```powershell
& '<conda-env>\python.exe' -B demo/record_agent_demo.py --seed 11037 --seconds 60
```

编码使用本次补装的必要视频组件 `imageio-ffmpeg 0.8.1`，其附带 FFmpeg 9.0.2，编码器为 libx264。其余推理、ALE、OpenCV、Pillow 依赖均使用既有环境。

## Pong Actor-Critic，兼容环境演示

> 演示使用重建的兼容推理环境，历史训练环境尚未完全核实；视频仅展示当前模型行为，不作为严格可比的独立评估结果。

- 视频为 `pong_actor_critic_latest.mp4`，同一游戏 Episode 开始后的连续 60 秒，1280×720，60 FPS，共 3600 帧。H.264 High / avc1，yuv420p，MP4 faststart，无音轨。Evaluation seed 固定为 7，沿用现存改进版 `pg_evaluate.py` 的 seed 选择。
- 加载 `../modern_pg_improved/checkpoints/pong_pg_latest.pt` 的 `model`。该文件与 `pong_pg_10000.pt` 字节一致，内部 `episodes_completed=10000`、`total_steps=45211537`，同时包含训练好的 `critic_model`。两份文件 SHA-256 均为 `a2617a925f676caaf7f6ca0962d925b73eb2e6e26deeb8de2dd34b4be0152786`。
- Actor 复用 `../modern_pg/pg_policy.py` 的 `PongActor`，严格恢复全部参数，结构为 Flatten(6400) → Linear(6400,256) → ReLU → Linear(256,2)。Checkpoint 中 Critic 的输出形状为 `(1,256)`，对应独立价值网络。执行游戏动作使用已训练 Actor。
- 推理方法直接来自 `../modern_pg_improved/pg_agent.py` 的 `Agent_PG.make_action()` 和 `init_game_setting()`。录制入口仅设置这两个方法需要的环境、Actor、上一帧和动作映射字段。Actor 以 `eval()`、`requires_grad_(False)` 和 `torch.inference_mode()` 执行。
- 预处理直接调用 `../modern_pg/pg_preprocess.py`。原始 210×160 RGB 经 `cv2.COLOR_RGB2GRAY`、`cv2.resize(gray, (80,80))` 的默认双线性插值、float32 /255，得到范围 [0,1] 的单帧，再计算当前帧减上一帧，输入范围 [-1,1]，形状为 `(80,80)`。Episode 首次输入为全零帧差，Actor 内部展平为 6400。录制时逐次验证输入形状、dtype、有限值和数值范围。
- Actor 输出两个 logits，通过 `torch.distributions.Categorical(logits=logits).sample()` 采样，索引 0/1 映射到合法 Atari 动作 2/3。实际 Pong 最小动作集为 `[NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]`，每次动作均验证为 2 或 3。
- 兼容环境复用 `../modern_pg/environment.py` 的 `make_env("Pong", wrapped=False, test=True, render_mode="rgb_array", max_episode_steps=27000)`，配合原 `Environment` 四返回值接口。实际为 `ALE/Pong-v5`，随机 frameskip=(2,5)，当前 ALE/Gymnasium 实现使用上界排他的采样，实际每次决策重复 2、3 或 4 个模拟器帧。sticky action 概率为 0，最小动作集，原始 RGB 观察，奖励为模拟器原始得失分。Gymnasium TimeLimit 为 27000 次外层 `step()`，游戏 mode/difficulty 使用该工厂的默认值。
- 游戏以 `reset(seed=7)` 开始。每个动作均来自 Actor，得失分后继续当前 Episode 并保留上一观察帧。Episode 在 ALE 终止或时间限制触发时结束，届时录制停止。得失分时重置折扣 Return 的代码属于训练回报计算，本次执行原推理方法。
- 使用同一个环境进行状态推理和 RGB 渲染。录制器对该环境现有 ALE 对象加透传代理，在原 `AtariEnv.step()` 内每次既有 `ale.act()` 返回后调用该环境的 `render()`，验证帧计数每次增加 1。原随机跳帧采样与动作重复循环由 Gymnasium 原实现执行，视频按 NTSC 游戏的 60 个模拟器帧/秒编码。
- 原始 210×160 RGB 游戏区域按最近邻等比例放大至 630×480，居中置于 16∶9 画布，保留比分、双方挡板和小球。顶部标明模型、训练节点和 seed，底部标明重建兼容环境。视频截取首 3600 帧，最后一次完整决策完成后实际观察 3601 帧。该片段内 Episode 持续进行。
- 录制前，用同一 seed 比较普通兼容环境与 RGB 采集环境的 64 次决策，动作、奖励、终止标志、原始观察以及预处理后帧的哈希逐条一致。正式片段有 1219 次 Actor 决策，动作 2 共 536 次，动作 3 共 683 次，实际随机跳帧 2/3/4 的决策次数分别为 427/421/371。
- OpenCV 完整解码 3600 帧，检查游戏区域非黑、分辨率、FPS 和画面变化，3587 次相邻帧出现画面变化。FFmpeg 完整解码通过，抽查 10、30、50 秒画面中的挡板、小球与比分。录制前后 Actor 参数与两份原始 Checkpoint 哈希均通过一致性检查。

### 实现来源与待核实项

- 改进版训练、评估入口均引用已缺失的 `../modern`。当前主仓库的全部 Git 历史中未找到该路径的实现，项目目录目前尚未进入已跟踪历史。
- `../original-2018-source.zip` 中 `hw4/environment.py` 和 `hw4/agent_dir/agent_pg.py` 与 `../original-2018/` 对应文件逐字节一致。该归档保存官方模板，PG 预处理仍使用已删除的 `scipy.misc.imresize`，Actor 和策略采样为待实现模板。兼容演示采用当前项目已实现的 `pg_policy.py` 与 `pg_preprocess.py`。
- 未核实的历史配置包括旧 `modern/` 的实际源文件版本、当时 Gym/ALE/ROM 版本、Frame Skip 与 sticky action、具体预处理版本、Episode 时间限制及续训随机数状态。当前演示参数如上逐项记录，Checkpoint 的输入输出结构与现有改进版推理方法已核实。
- 当前环境版本为 Python 3.11.17、PyTorch 2.11.0+cpu、Gymnasium 1.4.0、ALE 0.12.1、OpenCV 5.0.0.93。原训练数据和既有评估分析中的历史环境限制继续适用。

复现命令在项目根目录执行，脚本对已有 Pong 输出采取报错保护。

```powershell
& '<conda-env>\python.exe' -B demo/record_pong_actor_critic.py --seed 7 --seconds 60
```
