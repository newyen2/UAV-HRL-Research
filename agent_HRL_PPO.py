"""Agents for the dedicated HRL-PPO/A2PO training method."""

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from networks import HRLPPOActor, SharedDeviceQNetwork, StateValueCritic
from utils import (
    ENV_SETTING,
    HRL_PPO_LOWER_SETTING,
    HRL_PPO_UPPER_SETTING,
    TRAIN_SETTING,
)


class UpperAgent:
    """Independent duration-aware DQN for one UAV's target choice."""

    def __init__(self):
        self.state_size = HRL_PPO_UPPER_SETTING.state_size
        self.action_size = HRL_PPO_UPPER_SETTING.action_size
        self.hidden_size = HRL_PPO_UPPER_SETTING.hidden_size
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
            self.network.parameters(), lr=HRL_PPO_UPPER_SETTING.lr
        )
        self.average_reward_rate = 0.0

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim != 2 or state_tensor.shape != (1, self.state_size):
            raise ValueError(
                f"Expected one HRL-PPO upper state of size {self.state_size}, "
                f"got {tuple(state_tensor.shape)}"
            )

        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            action = int(self.network(state_tensor).argmax(dim=1).item())
        self.network.train(was_training)
        return action

    def _build_targets(self, experiences):
        states, actions, rewards, next_states, durations, completed = experiences
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1, 1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(durations, dtype=torch.float32, device=self.device).reshape(-1, 1)
        completed = torch.as_tensor(completed, dtype=torch.bool, device=self.device).reshape(-1, 1)
        batch_size = states.shape[0]
        if states.shape != (batch_size, self.state_size):
            raise ValueError("HRL-PPO upper states do not match the configured size")
        if next_states.shape != states.shape:
            raise ValueError("HRL-PPO upper next states must match the state batch")
        if any(
            field.shape[0] != batch_size
            for field in (actions, rewards, durations, completed)
        ):
            raise ValueError("HRL-PPO upper experience fields must have matching batch sizes")

        if HRL_PPO_UPPER_SETTING.reward_mode not in (
            "aoi_risk_count", "option_mean_environment", "average_reward_smdp"
        ):
            raise ValueError(
                f"Unknown HRL-PPO upper reward mode: {HRL_PPO_UPPER_SETTING.reward_mode}"
            )

        with torch.no_grad():
            # Completed options may select a new target. An unfinished fragment
            # must continue the target already represented by its action field.
            target_values = self.target_net(next_states)
            max_next_values = target_values.max(dim=1, keepdim=True).values
            same_action_values = target_values.gather(1, actions)
            continuation_values = torch.where(
                completed, max_next_values, same_action_values
            )
            if HRL_PPO_UPPER_SETTING.reward_mode == "average_reward_smdp":
                td_targets = (
                    rewards
                    - self.average_reward_rate * durations
                    + continuation_values
                )
            else:
                td_targets = rewards + (
                    HRL_PPO_UPPER_SETTING.gamma ** durations
                ) * continuation_values
        return states, actions, td_targets

    def learn(self, experiences):
        states, actions, td_targets = self._build_targets(experiences)
        predicted_values = self.network(states).gather(1, actions)
        if HRL_PPO_UPPER_SETTING.reward_mode == "average_reward_smdp":
            # Match HRL's rho update using the pre-optimizer TD error.
            td_errors = (td_targets - predicted_values).detach()
            self.average_reward_rate += (
                HRL_PPO_UPPER_SETTING.average_reward_lr
                * float(td_errors.mean().cpu())
            )
        loss = 0.5 * F.mse_loss(predicted_values, td_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite HRL-PPO upper DQN loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = clip_grad_norm_(self.network.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite HRL-PPO upper DQN gradient")
        self.optimizer.step()
        self.soft_update()
        return float(loss.detach().cpu())

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
                    local_values, HRL_PPO_UPPER_SETTING.tau
                )
                target_updates = torch._foreach_mul(
                    target_values, 1.0 - HRL_PPO_UPPER_SETTING.tau
                )
                updated_values = torch._foreach_add(local_updates, target_updates)
                torch._foreach_copy_(target_values, updated_values)
            else:
                for target_param, local_param in zip(
                    self.target_net.parameters(), self.network.parameters()
                ):
                    target_param.data.copy_(
                        HRL_PPO_UPPER_SETTING.tau * local_param.data
                        + (1.0 - HRL_PPO_UPPER_SETTING.tau) * target_param.data
                    )


class LowerAgent:
    """One UAV's tanh-Gaussian PPO actor, with no rollout state cache."""

    def __init__(self):
        self.state_size = HRL_PPO_LOWER_SETTING.state_size
        self.action_size = HRL_PPO_LOWER_SETTING.action_size
        self.hidden_size = HRL_PPO_LOWER_SETTING.hidden_size
        self.device = TRAIN_SETTING.device

        self.actor = HRLPPOActor(
            state_size=self.state_size,
            action_size=self.action_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        # Keep the free parameter inside its configured clamp so std gradients
        # remain available, including when callers choose a narrower interval.
        with torch.no_grad():
            self.actor.log_std.clamp_(HRL_PPO_LOWER_SETTING.log_std_min,
                                      HRL_PPO_LOWER_SETTING.log_std_max)
        self.optimizer = optim.Adam(
            self.actor.parameters(), lr=HRL_PPO_LOWER_SETTING.actor_lr
        )

    def _state_batch(self, states):
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.state_size:
            raise ValueError(
                f"Expected HRL-PPO lower states shaped (batch, {self.state_size}), "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    def _latent_batch(self, latent_actions, batch_size):
        tensor = torch.as_tensor(
            latent_actions, dtype=torch.float32, device=self.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (batch_size, self.action_size):
            raise ValueError(
                "HRL-PPO latent actions must match the configured batch and action sizes"
            )
        return tensor

    def _distribution(self, states):
        means = self.actor(states)
        log_standard_deviations = self.actor.log_std.clamp(
            HRL_PPO_LOWER_SETTING.log_std_min,
            HRL_PPO_LOWER_SETTING.log_std_max,
        ).expand_as(means)
        return Normal(means, log_standard_deviations.exp())

    @staticmethod
    def _tanh_log_probability(distribution, latent_actions):
        # log|d tanh(z)/dz| in a stable form, including saturated actions.
        log_jacobian = 2.0 * (
            np.log(2.0) - latent_actions - F.softplus(-2.0 * latent_actions)
        )
        return (
            distribution.log_prob(latent_actions) - log_jacobian
        ).sum(dim=-1)

    def log_prob(self, states, latent_actions):
        state_tensor = self._state_batch(states)
        latent_tensor = self._latent_batch(latent_actions, state_tensor.shape[0])
        log_probabilities = self._tanh_log_probability(
            self._distribution(state_tensor), latent_tensor
        )
        if not torch.isfinite(log_probabilities).all():
            raise FloatingPointError("Non-finite HRL-PPO lower log probability")
        return log_probabilities

    def get_action(self, state, deterministic=False):
        state_tensor = self._state_batch(state)
        if state_tensor.shape[0] != 1:
            raise ValueError("LowerAgent.get_action accepts exactly one state")
        with torch.no_grad():
            distribution = self._distribution(state_tensor)
            if deterministic:
                latent_action = distribution.mean
            else:
                latent_action = distribution.sample()
            action = torch.tanh(latent_action)
            log_probability = self._tanh_log_probability(
                distribution, latent_action
            )
        if not torch.isfinite(action).all() or not torch.isfinite(log_probability).all():
            raise FloatingPointError("Non-finite HRL-PPO lower action sample")
        return (
            action[0].cpu().numpy(),
            latent_action[0].cpu().numpy(),
            float(log_probability[0].cpu()),
        )

    def get_mean_action(self, state):
        """Return only the deterministic action needed by environment evaluation."""
        state_tensor = self._state_batch(state)
        if state_tensor.shape[0] != 1:
            raise ValueError("LowerAgent.get_mean_action accepts exactly one state")
        with torch.no_grad():
            action = torch.tanh(self.actor(state_tensor))[0].cpu().numpy()
        # Validate on CPU after the one required transfer, avoiding a GPU scalar sync.
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite HRL-PPO lower mean action")
        return action

    def evaluate(self, states, latent_actions):
        """Return stored-latent log probabilities and a tanh-action entropy estimate.

        The entropy is estimated as ``-log pi(a|s)`` for a fresh
        reparameterized sample, including the tanh Jacobian correction.
        """
        state_tensor = self._state_batch(states)
        latent_tensor = self._latent_batch(latent_actions, state_tensor.shape[0])
        distribution = self._distribution(state_tensor)
        log_probabilities = self._tanh_log_probability(
            distribution, latent_tensor
        )
        entropy_latents = distribution.rsample()
        entropy = -self._tanh_log_probability(distribution, entropy_latents)
        if not torch.isfinite(log_probabilities).all() or not torch.isfinite(entropy).all():
            raise FloatingPointError("Non-finite HRL-PPO lower policy evaluation")
        return log_probabilities, entropy

    def pretrain(self, states, teacher_actions):
        """Initialize the actor from navigation demonstrations using fixed settings."""
        state_tensor = self._state_batch(states)
        teacher_tensor = torch.as_tensor(
            teacher_actions, dtype=torch.float32, device=self.device
        )
        if teacher_tensor.ndim == 1:
            teacher_tensor = teacher_tensor.unsqueeze(0)
        if teacher_tensor.shape != (state_tensor.shape[0], self.action_size):
            raise ValueError(
                "HRL-PPO teacher actions must match the lower state batch and action size"
            )
        if state_tensor.shape[0] == 0:
            raise ValueError("HRL-PPO navigation pretraining data cannot be empty")

        losses = []
        for _ in range(HRL_PPO_LOWER_SETTING.pretrain_batches):
            batch_indices = torch.randint(
                state_tensor.shape[0],
                (HRL_PPO_LOWER_SETTING.pretrain_batch_size,),
                device=self.device,
            )
            predicted_actions = torch.tanh(self.actor(state_tensor[batch_indices]))
            loss = F.mse_loss(
                predicted_actions, teacher_tensor[batch_indices]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite HRL-PPO pretraining loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_(self.actor.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite HRL-PPO pretraining gradient")
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else 0.0


def compute_gae(rewards, values, next_values, trace_ends, ratios=None):
    """Compute GAE; optional ratios correct each trace using the next step's ratio.

    Episode and time-limit boundaries stop the recursive trace, while each
    supplied ``next_values`` still contributes to its TD residual. In SCGAE
    mode ``ratios[t + 1]`` multiplies the continuation from step ``t + 1``;
    ratios at inactive timesteps should therefore be supplied as one.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    trace_ends = np.asarray(trace_ends, dtype=np.bool_)
    if any(
        field.ndim != 1
        for field in (rewards, values, next_values, trace_ends)
    ):
        raise ValueError("GAE inputs must be one-dimensional arrays")
    if not (
        rewards.shape == values.shape == next_values.shape == trace_ends.shape
    ):
        raise ValueError("GAE inputs must have matching rollout lengths")
    if ratios is not None:
        ratios = np.asarray(ratios, dtype=np.float64)
        if ratios.shape != rewards.shape:
            raise ValueError("SCGAE ratios must match the rollout length")
        if not np.isfinite(ratios).all() or np.any(ratios < 0.0):
            raise ValueError("SCGAE ratios must be finite and non-negative")

    deltas = rewards + HRL_PPO_LOWER_SETTING.gamma * next_values - values
    advantages = np.zeros_like(deltas)
    continuation = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        trace_weight = 1.0
        if ratios is not None and index + 1 < len(deltas):
            trace_weight = min(
                HRL_PPO_LOWER_SETTING.trace_clip, ratios[index + 1]
            )
        continuation = deltas[index] + (
            HRL_PPO_LOWER_SETTING.gamma
            * HRL_PPO_LOWER_SETTING.gae_lambda
            * (not trace_ends[index])
            * trace_weight
            * continuation
        )
        advantages[index] = continuation
    return advantages.astype(np.float32)


class LowerPPO:
    """Centralized-value, per-UAV PPO or sequentially corrected A2PO learner."""

    def __init__(self):
        self.state_size = HRL_PPO_LOWER_SETTING.state_size
        self.action_size = HRL_PPO_LOWER_SETTING.action_size
        self.hidden_size = HRL_PPO_LOWER_SETTING.hidden_size
        self.device = TRAIN_SETTING.device
        self.critic_state_size = HRL_PPO_LOWER_SETTING.critic_state_size
        self.agent_count = ENV_SETTING.U

        self.agents = [LowerAgent() for _ in range(self.agent_count)]
        self.critic = StateValueCritic(
            state_size=self.critic_state_size,
            layer_size=self.hidden_size,
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=HRL_PPO_LOWER_SETTING.critic_lr
        )

    def values(self, states):
        state_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        scalar_input = state_tensor.ndim == 1
        if scalar_input:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim != 2 or state_tensor.shape[1] != self.critic_state_size:
            raise ValueError(
                "Centralized critic states do not match the configured state size"
            )
        with torch.no_grad():
            result = self.critic(state_tensor).squeeze(-1).cpu().numpy()
        if scalar_input:
            return float(result[0])
        return result

    def sample(self, states, active, deterministic=False):
        states = np.asarray(states, dtype=np.float32)
        active = np.asarray(active, dtype=np.bool_)
        if states.shape != (self.agent_count, self.state_size):
            raise ValueError(
                f"Expected lower states shaped ({self.agent_count}, {self.state_size})"
            )
        if active.shape != (self.agent_count,):
            raise ValueError("Lower active mask must have one flag per UAV")

        actions = np.zeros((self.agent_count, self.action_size), dtype=np.float32)
        latents = np.zeros_like(actions)
        log_probabilities = np.zeros(self.agent_count, dtype=np.float32)
        for agent_index, is_active in enumerate(active):
            if is_active:
                (
                    actions[agent_index],
                    latents[agent_index],
                    log_probabilities[agent_index],
                ) = self.agents[agent_index].get_action(
                    states[agent_index], deterministic=deterministic
                )
        return actions, latents, log_probabilities

    def get_mean_actions(self, states, active):
        """Evaluate active actors without rollout latents or log probabilities."""
        states = np.asarray(states, dtype=np.float32)
        active = np.asarray(active, dtype=np.bool_)
        if states.shape != (self.agent_count, self.state_size):
            raise ValueError(
                f"Expected lower states shaped ({self.agent_count}, {self.state_size})"
            )
        if active.shape != (self.agent_count,):
            raise ValueError("Lower active mask must have one flag per UAV")

        actions = np.zeros((self.agent_count, self.action_size), dtype=np.float32)
        for agent_index, is_active in enumerate(active):
            if is_active:
                actions[agent_index] = self.agents[agent_index].get_mean_action(
                    states[agent_index]
                )
        return actions

    def _prepare_rollout(self, rollout):
        if not rollout:
            raise ValueError("LowerPPO.learn requires a non-empty rollout")
        states = torch.as_tensor(
            np.stack([step["states"] for step in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        critic_states = torch.as_tensor(
            np.stack([step["critic_state"] for step in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        latents = torch.as_tensor(
            np.stack([step["latents"] for step in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probabilities = torch.as_tensor(
            np.stack([step["log_probs"] for step in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        active = torch.as_tensor(
            np.stack([step["active"] for step in rollout]),
            dtype=torch.bool,
            device=self.device,
        )
        rewards = np.asarray([step["reward"] for step in rollout], dtype=np.float32)
        values = np.asarray([step["value"] for step in rollout], dtype=np.float32)
        next_values = np.asarray(
            [step["next_value"] for step in rollout], dtype=np.float32
        )
        trace_ends = np.asarray(
            [step["trace_end"] for step in rollout], dtype=np.bool_
        )
        rollout_size = len(rollout)
        if states.shape != (rollout_size, self.agent_count, self.state_size):
            raise ValueError("Rollout lower states have an unexpected shape")
        if critic_states.shape != (rollout_size, self.critic_state_size):
            raise ValueError("Rollout centralized critic states have an unexpected shape")
        if latents.shape != (rollout_size, self.agent_count, self.action_size):
            raise ValueError("Rollout latent actions have an unexpected shape")
        if old_log_probabilities.shape != (rollout_size, self.agent_count):
            raise ValueError("Rollout old log probabilities have an unexpected shape")
        if active.shape != (rollout_size, self.agent_count):
            raise ValueError("Rollout active mask has an unexpected shape")
        if any(
            field.shape != (rollout_size,)
            for field in (rewards, values, next_values, trace_ends)
        ):
            raise ValueError("Rollout scalar fields must match the rollout length")
        if not all(
            torch.isfinite(field).all()
            for field in (states, critic_states, latents, old_log_probabilities)
        ) or not all(
            np.isfinite(field).all()
            for field in (rewards, values, next_values)
        ):
            raise ValueError("Rollout contains non-finite policy or value data")
        return (
            states,
            critic_states,
            latents,
            old_log_probabilities,
            active,
            rewards,
            values,
            next_values,
            trace_ends,
        )

    def _critic_update(self, critic_states, returns):
        sample_count = critic_states.shape[0]
        weighted_losses = []
        for _ in range(HRL_PPO_LOWER_SETTING.epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, HRL_PPO_LOWER_SETTING.minibatch_size):
                indices = permutation[
                    start:start + HRL_PPO_LOWER_SETTING.minibatch_size
                ]
                predictions = self.critic(critic_states[indices]).squeeze(-1)
                loss = 0.5 * F.mse_loss(predictions, returns[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite HRL-PPO critic loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = clip_grad_norm_(self.critic.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite HRL-PPO critic gradient")
                self.critic_optimizer.step()
                weighted_losses.append(
                    (float(loss.detach().cpu()), indices.numel())
                )
        if not weighted_losses:
            return 0.0
        return sum(loss * count for loss, count in weighted_losses) / sum(
            count for _, count in weighted_losses
        )

    @staticmethod
    def _safe_ratio(log_ratio):
        return torch.exp(log_ratio.clamp(min=-20.0, max=20.0))

    def _prior_ratio_products(
        self, updated_agents, states, latents, old_log_probabilities, active
    ):
        products = torch.ones_like(old_log_probabilities)
        with torch.no_grad():
            for agent_index in updated_agents:
                active_indices = torch.nonzero(
                    active[:, agent_index], as_tuple=False
                ).squeeze(-1)
                if active_indices.numel() == 0:
                    continue
                new_log_probabilities = self.agents[agent_index].log_prob(
                    states[active_indices, agent_index],
                    latents[active_indices, agent_index],
                )
                log_ratio = (
                    new_log_probabilities
                    - old_log_probabilities[active_indices, agent_index]
                )
                products[active_indices, agent_index] = self._safe_ratio(
                    log_ratio
                )
        return products.prod(dim=1)

    def _agent_order(self, advantages, values, active):
        if HRL_PPO_LOWER_SETTING.algorithm == "ppo":
            return list(range(self.agent_count))
        if HRL_PPO_LOWER_SETTING.algorithm != "a2po":
            raise ValueError("HRL-PPO lower algorithm must be 'ppo' or 'a2po'")
        if HRL_PPO_LOWER_SETTING.order == "random":
            order = list(range(self.agent_count))
            random.shuffle(order)
            return order
        if HRL_PPO_LOWER_SETTING.order != "greedy":
            raise ValueError("HRL-PPO A2PO order must be 'greedy' or 'random'")

        advantages = torch.as_tensor(advantages, device=self.device)
        values = torch.as_tensor(values, device=self.device)
        scores = []
        for agent_index in range(self.agent_count):
            mask = active[:, agent_index]
            if not mask.any():
                score = float("-inf")
            else:
                # The denominator is per-step centralized value magnitude; the
                # mask excludes idle UAV steps from this heuristic only.
                score_values = (
                    torch.abs(advantages[mask])
                    / (torch.abs(values[mask]) + 1e-8)
                )
                score = float(score_values.mean().cpu())
            scores.append((score, agent_index))
        sorted_scores = sorted(
            scores, key=lambda pair: (-pair[0], pair[1])
        )
        return [agent_index for _, agent_index in sorted_scores]

    @staticmethod
    def _approximate_kl(log_ratio):
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        return ((ratio - 1.0) - log_ratio).mean()

    def _update_agent(
        self,
        agent_index,
        advantages,
        prior_products,
        states,
        latents,
        old_log_probabilities,
        active,
    ):
        active_indices = torch.nonzero(
            active[:, agent_index], as_tuple=False
        ).squeeze(-1)
        if active_indices.numel() == 0:
            return {
                "actor_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "updates": 0,
            }

        active_advantages = advantages[active_indices]
        if active_advantages.numel() > 1:
            active_advantages = (
                active_advantages - active_advantages.mean()
            ) / active_advantages.std(unbiased=False).clamp_min(1e-8)
        # A singleton has no meaningful sample variance to normalize by; keep
        # its advantage so an otherwise valid active transition still updates.

        weighted_actor_losses = []
        weighted_entropies = []
        weighted_kls = []
        weighted_clip_fractions = []
        update_count = 0
        should_stop = False
        actor = self.agents[agent_index]
        for _ in range(HRL_PPO_LOWER_SETTING.epochs):
            permutation = active_indices[
                torch.randperm(active_indices.numel(), device=self.device)
            ]
            for start in range(0, permutation.numel(), HRL_PPO_LOWER_SETTING.minibatch_size):
                indices = permutation[
                    start:start + HRL_PPO_LOWER_SETTING.minibatch_size
                ]
                local_positions = torch.searchsorted(active_indices, indices)
                old_log_prob = old_log_probabilities[indices, agent_index]
                new_log_prob, entropy = actor.evaluate(
                    states[indices, agent_index], latents[indices, agent_index]
                )
                own_log_ratio = new_log_prob - old_log_prob
                own_ratio = self._safe_ratio(own_log_ratio)
                if HRL_PPO_LOWER_SETTING.algorithm == "a2po":
                    joint_ratio = own_ratio * prior_products[indices]
                else:
                    joint_ratio = own_ratio
                minibatch_advantages = active_advantages[local_positions]
                clipped_ratio = joint_ratio.clamp(
                    1.0 - HRL_PPO_LOWER_SETTING.clip_ratio,
                    1.0 + HRL_PPO_LOWER_SETTING.clip_ratio,
                )
                surrogate = torch.minimum(
                    joint_ratio * minibatch_advantages,
                    clipped_ratio * minibatch_advantages,
                )
                entropy_mean = entropy.mean()
                policy_loss = -surrogate.mean()
                loss = (
                    policy_loss
                    - HRL_PPO_LOWER_SETTING.entropy_coef * entropy_mean
                )
                if not all(
                    torch.isfinite(field).all()
                    for field in (loss, policy_loss, entropy_mean, own_log_ratio)
                ):
                    raise FloatingPointError("Non-finite HRL-PPO actor objective")

                actor.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = clip_grad_norm_(actor.actor.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    actor.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite HRL-PPO actor gradient")
                actor.optimizer.step()
                with torch.no_grad():
                    actor.actor.log_std.clamp_(HRL_PPO_LOWER_SETTING.log_std_min,
                                               HRL_PPO_LOWER_SETTING.log_std_max)
                update_count += 1

                with torch.no_grad():
                    updated_log_prob = actor.log_prob(
                        states[indices, agent_index], latents[indices, agent_index]
                    )
                    updated_log_ratio = updated_log_prob - old_log_prob
                    approximate_kl = self._approximate_kl(updated_log_ratio)
                if not torch.isfinite(approximate_kl):
                    raise FloatingPointError("Non-finite HRL-PPO approximate KL")

                sample_count = indices.numel()
                weighted_actor_losses.append(
                    (float(loss.detach().cpu()), sample_count)
                )
                weighted_entropies.append(
                    (float(entropy_mean.detach().cpu()), sample_count)
                )
                weighted_kls.append(
                    (float(approximate_kl.cpu()), sample_count)
                )
                weighted_clip_fractions.append(
                    (
                        float(
                            ((joint_ratio - 1.0).abs() > HRL_PPO_LOWER_SETTING.clip_ratio)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        ),
                        sample_count,
                    )
                )

                if (
                    HRL_PPO_LOWER_SETTING.target_kl > 0.0
                    and float(approximate_kl.cpu())
                    > HRL_PPO_LOWER_SETTING.target_kl
                ):
                    should_stop = True
                    break
            if should_stop:
                break

        def weighted_mean(entries):
            if not entries:
                return 0.0
            return sum(value * count for value, count in entries) / sum(
                count for _, count in entries
            )

        return {
            "actor_loss": weighted_mean(weighted_actor_losses),
            "entropy": weighted_mean(weighted_entropies),
            "approx_kl": weighted_mean(weighted_kls),
            "clip_fraction": weighted_mean(weighted_clip_fractions),
            "updates": update_count,
        }

    def learn(self, rollout):
        (
            states,
            critic_states,
            latents,
            old_log_probabilities,
            active,
            rewards,
            values,
            next_values,
            trace_ends,
        ) = self._prepare_rollout(rollout)
        raw_advantages = compute_gae(
            rewards, values, next_values, trace_ends
        )
        returns = torch.as_tensor(
            raw_advantages + values, dtype=torch.float32, device=self.device
        )
        critic_loss = self._critic_update(critic_states, returns)
        raw_advantages_tensor = torch.as_tensor(
            raw_advantages, dtype=torch.float32, device=self.device
        )
        order = self._agent_order(raw_advantages, values, active)

        actor_statistics = {}
        updated_agents = []
        for agent_index in order:
            prior_products = torch.ones(
                len(rollout), dtype=torch.float32, device=self.device
            )
            actor_advantages = raw_advantages_tensor
            if HRL_PPO_LOWER_SETTING.algorithm == "a2po":
                prior_products = self._prior_ratio_products(
                    updated_agents,
                    states,
                    latents,
                    old_log_probabilities,
                    active,
                )
                corrected_advantages = compute_gae(
                    rewards,
                    values,
                    next_values,
                    trace_ends,
                    ratios=prior_products.detach().cpu().numpy(),
                )
                actor_advantages = torch.as_tensor(
                    corrected_advantages,
                    dtype=torch.float32,
                    device=self.device,
                )

            statistics = self._update_agent(
                agent_index,
                actor_advantages,
                prior_products,
                states,
                latents,
                old_log_probabilities,
                active,
            )
            actor_statistics[agent_index] = statistics
            if HRL_PPO_LOWER_SETTING.algorithm == "a2po":
                updated_agents.append(agent_index)

        total_actor_updates = sum(
            statistics["updates"] for statistics in actor_statistics.values()
        )
        if total_actor_updates:
            def mean_metric(key):
                return sum(
                    statistics[key] * statistics["updates"]
                    for statistics in actor_statistics.values()
                ) / total_actor_updates

            actor_loss = mean_metric("actor_loss")
            entropy = mean_metric("entropy")
            approximate_kl = mean_metric("approx_kl")
            clip_fraction = mean_metric("clip_fraction")
        else:
            actor_loss = entropy = approximate_kl = clip_fraction = 0.0
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "approx_kl": approximate_kl,
            "clip_fraction": clip_fraction,
            "actor_updates": [
                actor_statistics[agent_index]["updates"]
                for agent_index in range(self.agent_count)
            ],
            "update_order": order,
            "rollout_size": len(rollout),
        }
