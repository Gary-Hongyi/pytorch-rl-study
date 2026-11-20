
import torch

from pg_policy import PongActor
from pg_preprocess import preprocess_frame, compute_frame_difference
from pathlib import Path


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0

    for reward in reversed(rewards):

        # Pong 特有：得分或失分时，重置累计回报
        if reward != 0:
            G = 0.0

        G = reward + gamma * G
        returns.append(G)

    returns.reverse()
    return returns


class Agent_PG:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.actor = PongActor()
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4
        )
        self.previous_frame = None
        self.action_map = [2, 3]

    # Step 5.2：每局开始时重置
    def init_game_setting(self):
        self.previous_frame = None

    # Step 5.3：根据观察值选择动作
    def make_action(self, observation, test=True):

        # 1. 图像预处理
        current_frame = preprocess_frame(observation)

        # 2. 计算帧差
        frame_diff = compute_frame_difference(
            current_frame, self.previous_frame
        )

        # 3. 保存当前帧，供下一次使用
        self.previous_frame = current_frame

        # 4. 转成 Tensor
        state = torch.from_numpy(frame_diff)

        # 5. Actor 决策
        with torch.no_grad():
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)
            action_index = dist.sample()

        # 6. 转换成 Pong 环境动作
        action = self.action_map[action_index.item()]

        return action

    def sample_training_action(self, observation):

        current_frame = preprocess_frame(observation)

        frame_diff = compute_frame_difference(
            current_frame,
            self.previous_frame
        )

        self.previous_frame = current_frame

        state = torch.from_numpy(frame_diff)

        # 训练时保留梯度,和测试中不同，和上面一个方法形成对比
        logits = self.actor(state)

        dist = torch.distributions.Categorical(logits=logits)

        action_index = dist.sample()

        log_prob = dist.log_prob(action_index)

        action = self.action_map[action_index.item()]

        return action, log_prob


    def train(self, num_episodes=5):

        # 记录每一局的 Reward
        episode_rewards = []

        # 记录每局结束时的累计环境步数
        episode_timesteps = []
        total_steps = 0

        # 新增：记录之前已经训练了多少局
        start_episode = 0

        # 1. 创建保存目录
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 2. 设置文件路径
        checkpoint_path = checkpoint_dir / "pong_pg_latest.pt"

        # ===== 自动检测并加载 Checkpoint =====

        if checkpoint_path.exists():

            # 从硬盘读取存档
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            # 旧版本没有记录 Time Steps，不能直接恢复新实验
            if "total_steps" not in checkpoint:
                raise RuntimeError(
                    "当前 Checkpoint 是旧格式。"
                    "请先将 pong_pg_latest.pt 重命名备份，"
                    "再重新运行。"
                )

            # 恢复 Actor 的网络参数
            self.actor.load_state_dict(checkpoint["model"])

            # 恢复 Adam 优化器的内部状态
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            # 恢复之前的训练局数和 Reward 记录
            start_episode = checkpoint["episodes_completed"]
            episode_rewards = list(checkpoint["episode_rewards"])

            # 恢复累计环境步数
            episode_timesteps = list(checkpoint["episode_timesteps"])
            total_steps = int(checkpoint["total_steps"])

            print(f"已恢复存档，之前训练了 {start_episode} 局")

        else:
            print("未发现 Checkpoint，从头开始训练")

        # 本次训练结束后的累计局数
        end_episode = start_episode + num_episodes

        # 外层循环：训练多局
        for episode in range(start_episode, end_episode):

            observation = self.env.reset()
            self.init_game_setting()

            done = False
            rewards = []
            log_probs = []

            # 内层循环：玩完一局
            while not done:

                action, log_prob = self.sample_training_action(observation)

                observation, reward, done, info = self.env.step(action)

                # 每次调用 env.step()，累计环境步数加一
                total_steps += 1

                rewards.append(reward)
                log_probs.append(log_prob)

            # 一局结束，计算 Return
            returns = torch.tensor(
                compute_returns(rewards),
                dtype=torch.float32
            )

            # 标准化 Return，降低训练信号的波动
            returns = (returns - returns.mean()) / (
                returns.std(unbiased=False) + 1e-8
            )

            log_probs = torch.stack(log_probs)

            # Policy Gradient Loss
            loss = -(log_probs * returns).sum()

            # 更新一次 Actor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录训练结果
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)

            # 记录这一局结束时的累计环境步数
            episode_timesteps.append(total_steps)

            print(
                f"Episode {episode + 1}/{end_episode}",
                f"Reward: {episode_reward:.1f}",
                f"Loss: {loss.item():.2f}"
            )

            # 计算最近 30 局的平均 Reward
            if len(episode_rewards) >= 30:

                avg30 = sum(episode_rewards[-30:]) / 30

                print(
                    f"Time Steps: {total_steps}",
                    f"Avg30 Reward: {avg30:.2f}"
                )

            else:
                print(
                    f"Time Steps: {total_steps}",
                    "Avg30: waiting for 30 episodes"
                )

            # ===== 每完成一个 Episode，保存 Checkpoint =====

            torch.save(
                {
                    "model": self.actor.state_dict(),  # 保留这个更新的参数
                    "optimizer": self.optimizer.state_dict(),  # 保留Adam优化器的内部状态
                    "episodes_completed": episode + 1,
                    "episode_rewards": episode_rewards,
                    "episode_timesteps": episode_timesteps,
                    "total_steps": total_steps,
                },
                checkpoint_path
            )

            print("Checkpoint saved:", checkpoint_path)

        return episode_rewards
