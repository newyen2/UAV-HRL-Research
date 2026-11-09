import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from networks import (
    GaussianActor, GaussianReturnCritic, QCritic, SharedDeviceQNetwork, StateValueCritic,
)
from utils import HRL_LOWER_SETTING, HRL_SETTING, HRL_UPPER_SETTING, TRAIN_SETTING


class UpperAgent:
    """One UAV's independent device-and-current-position DQN."""

    def __init__(self):
        self.state_size = HRL_UPPER_SETTING.state_size
        self.action_size = HRL_UPPER_SETTING.action_size
        self.hidden_size = HRL_UPPER_SETTING.hidden_size
        self.device = TRAIN_SETTING.device

        self.network = SharedDeviceQNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_net = SharedDeviceQNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(
            params=self.network.parameters(), lr=HRL_UPPER_SETTING.lr
        )
        self.average_reward_rate = 0.0

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            )
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            was_training = self.network.training
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state_tensor)
            self.network.train(was_training)
            action = np.argmax(action_values.cpu().numpy(), axis=1)
        else:
            action = np.asarray(
                [random.randrange(self.action_size)], dtype=np.int64
            )
        return action

    def _build_targets(self, experiences):
        states, actions, rewards, next_states, durations, completed = experiences
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1).values.unsqueeze(1)
            same_action_q_values = next_q_values.gather(1, actions)
            # An unfinished fragment continues its current target; the current-position action always
            # completes after its single primitive step and may choose again.
            continuation_values = torch.where(
                completed.bool(), max_next_q_values, same_action_q_values
            )
            if HRL_UPPER_SETTING.reward_mode == "average_reward_smdp":
                td_targets = (
                    rewards
                    - self.average_reward_rate * durations
                    + continuation_values
                )
            else:
                td_targets = rewards + (
                    HRL_UPPER_SETTING.gamma ** durations
                ) * continuation_values
        return states, actions, td_targets

    def learn(self, experiences):
        states, actions, td_targets = self._build_targets(experiences)
        q_values = self.network(states)
        predicted_q_values = q_values.gather(1, actions)
        if HRL_UPPER_SETTING.reward_mode == "average_reward_smdp":
            # Update rho from the pre-optimizer TD error, matching HRL's rule.
            td_errors = (td_targets - predicted_q_values).detach()
            self.average_reward_rate += (
                HRL_UPPER_SETTING.average_reward_lr * float(td_errors.mean())
            )

        loss = 0.5 * F.mse_loss(predicted_q_values, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()
        return float(loss.detach())

    def soft_update(self):
        with torch.no_grad():
            if self.device == "cuda":
                local_values = [
                    parameter.data for parameter in self.network.parameters()
                ]
                target_values = [
                    parameter.data for parameter in self.target_net.parameters()
                ]
                local_updates = torch._foreach_mul(
                    local_values, HRL_UPPER_SETTING.tau
                )
                target_updates = torch._foreach_mul(
                    target_values, 1.0 - HRL_UPPER_SETTING.tau
                )
                updated_values = torch._foreach_add(
                    local_updates, target_updates
                )
                torch._foreach_copy_(target_values, updated_values)
            else:
                for target_param, local_param in zip(
                    self.target_net.parameters(), self.network.parameters()
                ):
                    target_param.data.copy_(
                        HRL_UPPER_SETTING.tau * local_param.data
                        + (1.0 - HRL_UPPER_SETTING.tau) * target_param.data
                    )




class LowerAgent:
    # Select one lower-level algorithm from the startup setting.
    def __init__(self):
        self.state_size = HRL_LOWER_SETTING.state_size
        self.action_size = HRL_LOWER_SETTING.action_size
        self.hidden_size = HRL_LOWER_SETTING.hidden_size
        self.device = TRAIN_SETTING.device

        self.actor = GaussianActor(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=self.hidden_size,
        ).to(self.device)

        if HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t"):
            self.target_actor = GaussianActor(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)

        if HRL_LOWER_SETTING.algorithm == "a2c":
            self.critic = StateValueCritic(
                state_size=self.state_size,
                layer_size=self.hidden_size,
            ).to(self.device)
        elif HRL_LOWER_SETTING.algorithm == "sac":
            self.critic_1 = QCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.critic_2 = QCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.target_critic_1 = QCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.target_critic_2 = QCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
        elif HRL_LOWER_SETTING.algorithm == "dsac":
            self.critic = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.target_critic = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
        else:
            self.critic_1 = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.critic_2 = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.target_critic_1 = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)
            self.target_critic_2 = GaussianReturnCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                layer_size=self.hidden_size,
            ).to(self.device)

        if HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t"):
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_actor.requires_grad_(False)

        if HRL_LOWER_SETTING.algorithm == "dsac":
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_critic.requires_grad_(False)
        elif HRL_LOWER_SETTING.algorithm in ("sac", "dsac_t"):
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            self.target_critic_1.requires_grad_(False)
            self.target_critic_2.requires_grad_(False)

        if HRL_LOWER_SETTING.algorithm in ("a2c", "dsac"):
            self.critic_parameters = list(self.critic.parameters())
        elif HRL_LOWER_SETTING.algorithm in ("sac", "dsac_t"):
            self.critic_parameters = (
                list(self.critic_1.parameters()) + list(self.critic_2.parameters())
            )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=HRL_LOWER_SETTING.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic_parameters, lr=HRL_LOWER_SETTING.critic_lr
        )
        self.update_count = 0
        if HRL_LOWER_SETTING.algorithm == "a2c":
            self._last_action_sample = None
            self._a2c_rollout = []
        if HRL_LOWER_SETTING.algorithm == "dsac_t":
            self.mean_std1 = None
            self.mean_std2 = None

    def _to_tensor_batch(self, values):
        tensor = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        # 一維 state／action 向量視為單筆資料，補上 batch 維度。
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _prepare_experiences(self, experiences):
        states, actions, rewards, next_states, option_ends, teacher_actions = experiences
        states = self._to_tensor_batch(states)
        actions = self._to_tensor_batch(actions)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        next_states = self._to_tensor_batch(next_states)
        option_ends = torch.as_tensor(
            option_ends, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        teacher_actions = self._to_tensor_batch(teacher_actions)
        batch_size = states.shape[0]
        fields = (actions, rewards, next_states, option_ends, teacher_actions)
        if any(field.shape[0] != batch_size for field in fields):
            raise ValueError("HRL lower experience fields must have matching batch sizes")
        if states.shape[1] != self.state_size or next_states.shape[1] != self.state_size:
            raise ValueError("HRL lower states do not match the configured state size")
        if actions.shape[1] != self.action_size or teacher_actions.shape[1] != self.action_size:
            raise ValueError("HRL lower actions do not match the configured action size")
        return states, actions, rewards, next_states, option_ends, teacher_actions

    @staticmethod
    def _action_distribution(actor, states):
        mean, raw_log_standard_deviation = actor(states)
        log_standard_deviation = raw_log_standard_deviation.clamp(
            HRL_LOWER_SETTING.log_std_min,
            HRL_LOWER_SETTING.log_std_max,
        )
        return mean, Normal(mean, log_standard_deviation.exp())

    @staticmethod
    def _tanh_log_probability(distribution, latent_actions):
        # 穩定計算 log(1 - tanh(x)^2)，包括接近飽和的動作。
        correction = 2.0 * (
            np.log(2.0) - latent_actions - F.softplus(-2.0 * latent_actions)
        )
        return (distribution.log_prob(latent_actions) - correction).sum(
            dim=-1, keepdim=True
        )

    def _sample_tanh_action(self, actor, states, reparameterize=True):
        _, distribution = self._action_distribution(actor, states)
        if reparameterize:
            latent_actions = distribution.rsample()
        else:
            latent_actions = distribution.sample()
        actions = torch.tanh(latent_actions)
        log_probabilities = self._tanh_log_probability(
            distribution, latent_actions
        )
        return actions, log_probabilities, latent_actions

    @staticmethod
    def _mean_tanh_action(actor, states):
        mean, _ = actor(states)
        return torch.tanh(mean)

    def get_action(self, states, deterministic=False):
        state_tensor = self._to_tensor_batch(states)
        with torch.no_grad():
            if deterministic:
                if HRL_LOWER_SETTING.algorithm == "a2c":
                    self._last_action_sample = None
                    self._a2c_rollout.clear()
                actions = self._mean_tanh_action(self.actor, state_tensor)
            else:
                if HRL_LOWER_SETTING.algorithm == "a2c":
                    if self._last_action_sample is not None:
                        # A new action before learn means the prior transition was skipped.
                        self._a2c_rollout.clear()
                    actions, _, latent_actions = self._sample_tanh_action(
                        self.actor, state_tensor, reparameterize=False
                    )
                    self._last_action_sample = (
                        state_tensor.detach().clone(),
                        actions.detach().clone(),
                        latent_actions.detach().clone(),
                    )
                else:
                    actions, _, _ = self._sample_tanh_action(self.actor, state_tensor)
        return actions.cpu().numpy()

    def Pretrain(self, states, teacher_actions):
        if HRL_LOWER_SETTING.algorithm == "a2c":
            self._last_action_sample = None
            self._a2c_rollout.clear()
        state_tensor = self._to_tensor_batch(states)

        teacher_action_tensor = self._to_tensor_batch(teacher_actions)

        for _ in range(HRL_SETTING.pretrain_batches):
            batch_indices = torch.randint(
                len(state_tensor),
                (HRL_LOWER_SETTING.pretrain_batch_size,),
                device=self.device,
            )
            predicted_actions = self._mean_tanh_action(
                self.actor, state_tensor[batch_indices]
            )
            loss = F.mse_loss(
                predicted_actions, teacher_action_tensor[batch_indices]
            )
            self.actor_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

        if (HRL_SETTING.pretrain_batches > 0
                and HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t")):
            # 離線初始化後讓 TD target 使用已預訓練的政策，再開始正常軟更新。
            self.target_actor.load_state_dict(self.actor.state_dict())

    # 依演算法建立 critic target，並以 option end 遮罩停止 bootstrap。
    def _critic_target(self, rewards, next_states, option_ends):
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        next_states = self._to_tensor_batch(next_states)
        option_ends = torch.as_tensor(
            option_ends, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        with torch.no_grad():
            if HRL_LOWER_SETTING.algorithm == "sac":
                next_actions, next_log_probabilities, _ = self._sample_tanh_action(
                    self.actor, next_states
                )
                next_q1 = self.target_critic_1(next_states, next_actions)
                next_q2 = self.target_critic_2(next_states, next_actions)
                clipped_next_q = torch.minimum(next_q1, next_q2)
                soft_next_value = (
                    clipped_next_q
                    - HRL_LOWER_SETTING.alpha * next_log_probabilities
                )
                # option_ends 代表服務選項結束時停止 bootstrap。
                return rewards + HRL_LOWER_SETTING.gamma * (1.0 - option_ends) * soft_next_value
            elif HRL_LOWER_SETTING.algorithm == "dsac":
                next_actions, next_log_probabilities, _ = self._sample_tanh_action(
                    self.target_actor, next_states
                )
                target_mean, raw_target_std = self.target_critic(
                    next_states, next_actions
                )
                target_std = F.softplus(raw_target_std) + HRL_LOWER_SETTING.std_min
                noise = torch.randn_like(target_mean).clamp(-3.0, 3.0)
                sampled_return = target_mean + target_std * noise
                soft_return = sampled_return - (
                    HRL_LOWER_SETTING.alpha * next_log_probabilities
                )
                return rewards + HRL_LOWER_SETTING.gamma * (
                    1.0 - option_ends
                ) * soft_return
        raise ValueError("critic_target is only available for SAC and DSAC")

    # DSAC-T 以同一個較低均值 critic 建立期望與抽樣 target。
    def _critic_target_dsact(self, rewards, next_states, option_ends):
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        next_states = self._to_tensor_batch(next_states)
        option_ends = torch.as_tensor(
            option_ends, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_probabilities, _ = self._sample_tanh_action(
                self.target_actor, next_states
            )
            target_mean_1, raw_target_std_1 = self.target_critic_1(
                next_states, next_actions
            )
            target_mean_2, raw_target_std_2 = self.target_critic_2(
                next_states, next_actions
            )
            target_std_1 = F.softplus(raw_target_std_1) + HRL_LOWER_SETTING.std_min
            target_std_2 = F.softplus(raw_target_std_2) + HRL_LOWER_SETTING.std_min
            target_mean = torch.minimum(target_mean_1, target_mean_2)
            target_std = torch.where(target_mean_1 <= target_mean_2, target_std_1, target_std_2)

            # 期望與抽樣 target 都使用同一個被選中的 critic 分布。
            soft_mean_return = target_mean - (
                HRL_LOWER_SETTING.alpha * next_log_probabilities
            )
            expected_target = rewards + HRL_LOWER_SETTING.gamma * (
                1.0 - option_ends
            ) * soft_mean_return

            noise = torch.randn_like(target_mean).clamp(-3.0, 3.0)
            sampled_return = target_mean + target_std * noise
            soft_return = sampled_return - (
                HRL_LOWER_SETTING.alpha * next_log_probabilities
            )
            sampled_target = rewards + HRL_LOWER_SETTING.gamma * (
                1.0 - option_ends
            ) * soft_return
        return expected_target, sampled_target

    # DSAC 將均值誤差與標準差誤差分別依抽樣 target 計算。
    @staticmethod
    def _critic_loss_dsac(predicted_mean, predicted_std, sampled_target):
        detached_std = predicted_std.detach()
        # 截斷只限制標準差誤差，均值誤差仍使用原始 target。
        bounded_target = predicted_mean.detach() + (
            sampled_target.detach() - predicted_mean.detach()
        ).clamp(-HRL_LOWER_SETTING.td_bound, HRL_LOWER_SETTING.td_bound)
        mean_coefficient = -(sampled_target - predicted_mean).detach() / (
            detached_std.pow(2) + 0.1
        )
        std_coefficient = (
            detached_std.pow(2)
            - (predicted_mean.detach() - bounded_target).pow(2)
        ) / (detached_std.pow(3) + 0.1)
        return (
            mean_coefficient * predicted_mean
            + std_coefficient * predicted_std
        ).mean()

    # DSAC-T 的均值以期望 target 訓練，標準差使用截斷後的抽樣 target。
    @staticmethod
    def _critic_loss_dsact(
        predicted_mean, predicted_std, expected_target, sampled_target,
        running_mean_std
    ):
        # detached 倍率修正標準差的梯度尺度，避免尺度本身反向傳播。
        detached_std = predicted_std.detach()
        multiplier = (
            running_mean_std.pow(2) / (detached_std.pow(2) + 0.1)
        ).clamp(0.1, 10.0)
        td_bound = 3.0 * running_mean_std
        bounded_target = predicted_mean.detach() + (
            sampled_target.detach() - predicted_mean.detach()
        ).clamp(-td_bound, td_bound)
        mean_loss = F.huber_loss(
            predicted_mean, expected_target.detach(), delta=50.0, reduction="none"
        )
        bounded_huber_loss = F.huber_loss(
            predicted_mean.detach(), bounded_target, delta=50.0, reduction="none"
        )
        std_loss = predicted_std * (
            detached_std.pow(2) - bounded_huber_loss
        ) / (detached_std + 0.1)
        return (multiplier * (mean_loss + std_loss)).mean()

    # 驗證 action cache 並收集 rollout；episode 尾端以 critic value bootstrap。
    def _prepare_a2c_rollout(self, batch, rollout_end):
        states, actions, rewards, next_states, option_ends, teacher_actions = batch
        last_action_sample = self._last_action_sample
        if last_action_sample is None:
            cache_matches = False
        else:
            cached_states, cached_actions, latent_actions = last_action_sample
            cache_matches = torch.equal(states, cached_states) and torch.equal(
                actions, cached_actions
            )
        if not cache_matches:
            self._last_action_sample = None
            self._a2c_rollout.clear()
            raise ValueError(
                "A2C learn requires the exact state/action batch from its "
                "most recent stochastic get_action call"
            )

        if self._a2c_rollout and states.shape[0] != self._a2c_rollout[0][0].shape[0]:
            self._last_action_sample = None
            self._a2c_rollout.clear()
            raise ValueError("A2C rollout batch size must remain constant")

        self._last_action_sample = None
        # 保存獨立 tensor，避免呼叫端修改已收集資料。
        # 每筆依序為 state、action、reward、next_state、option_end、teacher、latent。
        self._a2c_rollout.append(tuple(
            tensor.detach().clone()
            for tensor in (
                states,
                actions,
                rewards,
                next_states,
                option_ends,
                teacher_actions,
                latent_actions,
            )
        ))

        if (
            len(self._a2c_rollout) < HRL_LOWER_SETTING.a2c_steps
            and not rollout_end
        ):
            return None

        rollout = list(self._a2c_rollout)
        self._a2c_rollout.clear()

        with torch.no_grad():
            running_returns = self.critic(rollout[-1][3])
            returns_by_step = [None] * len(rollout)
            # 各 UAV 獨立由尾端 value 向前累加；完成遮罩切斷新目標的後續回報。
            for step_index in range(len(rollout) - 1, -1, -1):
                step_rewards = rollout[step_index][2]
                step_option_ends = rollout[step_index][4]
                running_returns = step_rewards + HRL_LOWER_SETTING.gamma * (
                    1.0 - step_option_ends
                ) * running_returns
                returns_by_step[step_index] = running_returns

        # 依時間再依 UAV 展平，讓狀態、示範、latent 與 return 保持對齊。
        states = torch.cat([step[0] for step in rollout], dim=0)
        teacher_actions = torch.cat([step[5] for step in rollout], dim=0)
        latent_actions = torch.cat([step[6] for step in rollout], dim=0)
        td_targets = torch.cat(returns_by_step, dim=0)
        return states, teacher_actions, latent_actions, td_targets

    def learn(self, experiences, *, rollout_end=False, joint_episodes=0):
        # 輸入驗證失敗時清除 A2C 暫存資料，避免無效經驗污染後續 rollout。
        try:
            batch = self._prepare_experiences(experiences)
        except Exception:
            if HRL_LOWER_SETTING.algorithm == "a2c":
                self._last_action_sample = None
                self._a2c_rollout.clear()
            raise

        states, actions, rewards, next_states, option_ends, teacher_actions = batch
        if HRL_LOWER_SETTING.algorithm == "a2c":
            prepared = self._prepare_a2c_rollout(batch, rollout_end)
            if prepared is None:
                return None
            states, teacher_actions, latent_actions, td_targets = prepared

        imitation_weight = HRL_LOWER_SETTING.imitation_weight
        if HRL_LOWER_SETTING.imitation_decay_episodes > 0:
            decay_fraction = min(
                max(joint_episodes, 0)
                / HRL_LOWER_SETTING.imitation_decay_episodes,
                1.0,
            )
            imitation_weight += (
                HRL_LOWER_SETTING.min_imitation_weight - imitation_weight
            ) * decay_fraction

        if HRL_LOWER_SETTING.algorithm == "a2c":
            state_values = self.critic(states)
            advantages = td_targets - state_values
            critic_loss = 0.5 * F.mse_loss(state_values, td_targets)
        elif HRL_LOWER_SETTING.algorithm == "sac":
            td_targets = self._critic_target(rewards, next_states, option_ends)
            q1_values = self.critic_1(states, actions)
            q2_values = self.critic_2(states, actions)
            critic_loss = F.mse_loss(q1_values, td_targets) + F.mse_loss(
                q2_values, td_targets
            )
        elif HRL_LOWER_SETTING.algorithm == "dsac":
            td_targets = self._critic_target(rewards, next_states, option_ends)
            predicted_mean, raw_predicted_std = self.critic(states, actions)
            predicted_std = F.softplus(raw_predicted_std) + HRL_LOWER_SETTING.std_min
            critic_loss = self._critic_loss_dsac(
                predicted_mean, predicted_std, td_targets
            )
        elif HRL_LOWER_SETTING.algorithm == "dsac_t":
            expected_target, sampled_target = self._critic_target_dsact(
                rewards, next_states, option_ends
            )
            means_1, raw_stds_1 = self.critic_1(states, actions)
            means_2, raw_stds_2 = self.critic_2(states, actions)
            stds_1 = F.softplus(raw_stds_1) + HRL_LOWER_SETTING.std_min
            stds_2 = F.softplus(raw_stds_2) + HRL_LOWER_SETTING.std_min
            # 尺度統計脫離梯度圖，EMA 平滑批次波動並讓每個 critic 保有獨立尺度。
            current_mean_std1 = float(stds_1.detach().mean())
            if self.mean_std1 is None:
                self.mean_std1 = current_mean_std1
            else:
                self.mean_std1 = (
                    (1.0 - HRL_LOWER_SETTING.std_rate) * self.mean_std1
                    + HRL_LOWER_SETTING.std_rate * current_mean_std1
                )
            current_mean_std2 = float(stds_2.detach().mean())
            if self.mean_std2 is None:
                self.mean_std2 = current_mean_std2
            else:
                self.mean_std2 = (
                    (1.0 - HRL_LOWER_SETTING.std_rate) * self.mean_std2
                    + HRL_LOWER_SETTING.std_rate * current_mean_std2
                )
            running_std_1 = torch.as_tensor(
                self.mean_std1, dtype=torch.float32, device=self.device
            ).detach()
            running_std_2 = torch.as_tensor(
                self.mean_std2, dtype=torch.float32, device=self.device
            ).detach()
            critic_loss_1 = self._critic_loss_dsact(
                means_1, stds_1, expected_target, sampled_target, running_std_1
            )
            critic_loss_2 = self._critic_loss_dsact(
                means_2, stds_2, expected_target, sampled_target, running_std_2
            )
            critic_loss = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_parameters, 1.0)
        self.critic_optimizer.step()
        if HRL_LOWER_SETTING.algorithm == "sac":
            self.critic_optimizer.zero_grad(set_to_none=True)
        # update_count 統計已完成的 critic 更新，所有演算法都在此遞增。
        self.update_count += 1

        actor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        imitation_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t"):
            update_actor = (
                self.update_count % HRL_LOWER_SETTING.policy_delay == 0
            )
        else:
            update_actor = True

        if update_actor:
            # 凍結 critic 參數梯度，但保留 Q 對輸入 action 的梯度供 actor 更新。
            for parameter in self.critic_parameters:
                parameter.requires_grad_(False)

            if HRL_LOWER_SETTING.algorithm == "a2c":
                _, distribution = self._action_distribution(self.actor, states)
                selected_log_probabilities = self._tanh_log_probability(
                    distribution, latent_actions
                )
                actor_loss = -(
                    selected_log_probabilities * advantages.detach()
                ).mean()
            else:
                policy_actions, log_probabilities, _ = self._sample_tanh_action(
                    self.actor, states
                )
                if HRL_LOWER_SETTING.algorithm == "sac":
                    policy_q1 = self.critic_1(states, policy_actions)
                    policy_q2 = self.critic_2(states, policy_actions)
                    policy_value = torch.minimum(policy_q1, policy_q2)
                elif HRL_LOWER_SETTING.algorithm == "dsac":
                    policy_value, _ = self.critic(states, policy_actions)
                else:
                    policy_mean_1 = self.critic_1(states, policy_actions)[0]
                    policy_mean_2 = self.critic_2(states, policy_actions)[0]
                    policy_value = torch.minimum(policy_mean_1, policy_mean_2)
                actor_loss = (
                    HRL_LOWER_SETTING.alpha * log_probabilities - policy_value
                ).mean()

            imitation_loss = F.mse_loss(
                self._mean_tanh_action(self.actor, states), teacher_actions
            )
            combined_actor_loss = (
                actor_loss
                + imitation_weight * imitation_loss
            )
            self.actor_optimizer.zero_grad()
            combined_actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            for parameter in self.critic_parameters:
                parameter.requires_grad_(True)

            if HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t"):
                self.soft_update(self.actor, self.target_actor)
            if HRL_LOWER_SETTING.algorithm == "dsac":
                self.soft_update(self.critic, self.target_critic)
            elif HRL_LOWER_SETTING.algorithm in ("sac", "dsac_t"):
                self.soft_update(self.critic_1, self.target_critic_1)
                self.soft_update(self.critic_2, self.target_critic_2)

        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "imitation_loss": float(imitation_loss.detach()),
            "imitation_weight": imitation_weight,
        }

    # 以設定的 tau 原地同步 target weights，供各演算法共用。
    def soft_update(self, local_model, target_model):
        local_params = list(local_model.parameters())
        target_params = list(target_model.parameters())
        with torch.no_grad():
            torch._foreach_mul_(target_params, 1.0 - HRL_LOWER_SETTING.tau)
            torch._foreach_add_(
                target_params, local_params, alpha=HRL_LOWER_SETTING.tau
            )
