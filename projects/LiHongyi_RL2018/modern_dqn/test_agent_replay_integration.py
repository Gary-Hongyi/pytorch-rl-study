
import torch

from environment import make_breakout_env
from agent_dqn import Agent_DQN


# 测试专用：始终输出固定的四个 Q 值
class FixedTargetNetwork(torch.nn.Module):
    def forward(self, states):
        q_values = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]],
            dtype=torch.float32,
            device=states.device
        )
        return q_values.expand(states.shape[0], -1)


def main():
    env = make_breakout_env(test=False, seed=42)

    try:
        agent = Agent_DQN(env, args=None)
        state, _ = env.reset()

        # ===== Test 1: 真实经验 + Batch 转换 =====
        for _ in range(agent.batch_size):
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            agent.store_transition(
                state, action, reward,
                next_state, terminated, truncated
            )

            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        batch = agent._sample_batch_to_tensors()

        states, actions, rewards, next_states, terminateds, truncateds = batch

        assert len(agent.replay_buffer) == 32
        assert states.shape == (32, 4, 84, 84)
        assert actions.shape == (32, 1)
        assert next_states.shape == (32, 4, 84, 84)
        assert states.dtype == torch.float32

        print("Replay Batch test passed!")

        # ===== Test 2: Online Network 当前预测 =====
        current_q = agent._compute_current_q_values(
            states, actions
        )

        assert current_q.shape == (32, 1)
        assert current_q.requires_grad

        print("Current Q test passed!")

        
        # ===== Test: Online Network Parameter Update =====

        # 记录更新前的参数
        params_before = [
            p.detach().clone()
            for p in agent.q_network.parameters()
        ]

        # 使用 Replay Buffer 更新一次网络
        loss = agent._learn_from_replay()

        # 检查是否有参数发生变化
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                params_before,
                agent.q_network.parameters()
            )
        )

        print("Training Loss:", loss)
        print("Online parameters changed:", changed)

        assert 0 <= loss < float("inf")
        assert changed

        print("Online Network update test passed!")

        # 测试 Target Network 同步
        agent._sync_target_network()

        for online_param, target_param in zip(
            agent.q_network.parameters(),
            agent.target_network.parameters()
        ):
            assert torch.equal(online_param, target_param)
            assert not target_param.requires_grad

        print("Target Network sync test passed!")

        # ===== Test 3: TD Target 公式 =====
        # 仅替换这个测试 Agent 的 Target Network
        agent.target_network = FixedTargetNetwork().to(agent.device)
        agent.gamma = 0.99

        test_rewards = torch.tensor(
            [1.0, -1.0], device=agent.device
        )

        test_terminateds = torch.tensor(
            [False, True], device=agent.device
        )

        test_next_states = torch.zeros(
            (2, 4, 84, 84), device=agent.device
        )

        td_targets = agent._compute_td_targets(
            test_rewards,
            test_next_states,
            test_terminateds
        )

        expected = torch.tensor(
            [4.96, -1.0], device=agent.device
        )

        print("TD Targets:", td_targets)

        assert torch.allclose(td_targets, expected, atol=1e-5)
        assert not td_targets.requires_grad

        print("TD Target test passed!")
        print("All tests passed!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
