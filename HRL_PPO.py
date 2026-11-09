"""Independent DQN service planning with on-policy PPO/A2PO navigation."""
from copy import deepcopy
import random
import time

import numpy as np
import torch

from Environment import Environment
from agent_HRL_PPO import LowerPPO, UpperAgent
from buffer import HRLUpperReplayBuffer
from runtime.hrl_ppo_output import HRLPPOFileIO
from runtime.progress import Progress
from runtime.record import Record
from runtime.training_evaluation import record_evaluation
import utils
from utils import (
    HRL_PPO_LOWER_SETTING,
    HRL_PPO_SETTING,
    HRL_PPO_UPPER_SETTING,
)


def upper_state(env, u):
    raw = env.get_state().copy()
    raw[:2 * env.U + 2 * env.M] /= env.map_size
    raw[2 * env.U + 2 * env.M:] /= utils.TRAIN_SETTING.max_steps
    return np.concatenate((env.UAV_coord[u] / env.map_size, raw))


def risk_observation(env):
    # Keep the configured dimension when risk is disabled; zeros mean no region.
    regions = np.zeros((len(utils.FIXED_SETTING.fixed_risk_region), 4))
    if utils.FUNCTIONAL_SETTING.risk:
        regions[:] = env.risk_region / env.map_size
    return regions.reshape(-1)


def lower_states(env, targets):
    raw = env.get_state()
    uav_xy = raw[:2 * env.U] / env.map_size
    risk = risk_observation(env)
    states = []
    for u, target in enumerate(targets):
        active = target != env.M
        if active:
            goal = env.device_coord[target]
        else:
            goal = env.UAV_coord[u]
        states.append(np.concatenate((
            env.UAV_coord[u] / env.map_size,
            goal / env.map_size,
            (goal - env.UAV_coord[u]) / env.map_size,
            [float(active)], uav_xy, risk,
        )))
    return np.asarray(states, dtype=np.float32)


def critic_state(env, targets):
    raw = env.get_state().copy()
    raw[:2 * env.U + 2 * env.M] /= env.map_size
    raw[2 * env.U + 2 * env.M:] /= utils.TRAIN_SETTING.max_steps
    goals = []
    for u, target in enumerate(targets):
        if target == env.M:
            goal = env.UAV_coord[u]
            active = 0.0
        else:
            goal = env.device_coord[target]
            active = 1.0
        goals.extend(goal / env.map_size)
        goals.append(active)
    return np.concatenate((raw, goals, risk_observation(env))).astype(np.float32)


def to_env_action(env, u, target, action):
    if target == env.M:
        return [env.M, np.zeros(2)]
    action = np.asarray(action, dtype=float)
    movement = action / max(1.0, float(np.linalg.norm(action))) * env.max_mov
    next_position = env.Update_Location(env.UAV_coord[u], movement)
    distance = np.linalg.norm(env.device_coord[target] - next_position)
    service = env.M
    if distance < env.map_size * HRL_PPO_SETTING.service_ratio:
        service = target
    return [service, movement]


def service_reductions(previous_aoi, env, actions, service_success=None):
    reductions = np.maximum(previous_aoi + 1.0 - env.AOI, 0.0)
    selected = np.asarray([action[0] for action in actions])
    if service_success is not None:
        selected = np.where(service_success, selected, env.M)
    rewards = np.zeros(env.U)
    for target in np.unique(selected):
        if target < env.M:
            matching = selected == target
            rewards[matching] = reductions[target] / np.count_nonzero(matching)
    return rewards


def upper_step_rewards(previous_aoi, env, actions, active):
    """Return primitive upper rewards for modes with per-step feedback."""
    if HRL_PPO_UPPER_SETTING.reward_mode != "aoi_risk_count":
        return np.zeros(env.U, dtype=float)
    rewards = service_reductions(
        previous_aoi, env, actions, getattr(env, "service_success", None)
    ) - (
        HRL_PPO_UPPER_SETTING.risk_penalty_weight * env.risk_penalty_counts
    )
    rewards[~np.asarray(active, dtype=bool)] = 0.0
    return rewards


def intrinsic_rewards(env, targets, before_distances, actions):
    """Compute each UAV's navigation reward after an environment step."""
    rewards = np.zeros(env.U)
    for u, target in enumerate(targets):
        if target != env.M:
            after_distance = np.linalg.norm(env.device_coord[target] - env.UAV_coord[u])
            progress = (before_distances[u] - after_distance) / env.max_mov
            rewards[u] = (
                HRL_PPO_SETTING.progress_weight * progress
                + HRL_PPO_SETTING.success_reward * (actions[u][0] == target)
                - HRL_PPO_SETTING.step_penalty
            )
        # Idle UAVs still incur environmental risk.
        rewards[u] -= (
            HRL_PPO_SETTING.risk_weight * env.risk_penalty_counts[u] * env.penalty
            / HRL_PPO_SETTING.risk_penalty_reference
        )
    return rewards


@record_evaluation
def eval_runs(env, upper_agents, lower_policy):
    returns = []
    for _ in range(utils.EVAL_SETTING.eval_runs):
        env.reset()
        targets = [None] * env.U
        episode_return = 0.0
        while env.steps < env.max_steps:
            for u in range(env.U):
                if targets[u] is None:
                    targets[u] = upper_agents[u].get_action(upper_state(env, u), 0.0)
            active = np.asarray(targets) != env.M
            moves = lower_policy.get_mean_actions(lower_states(env, targets), active)
            actions = [
                to_env_action(env, u, target, moves[u])
                for u, target in enumerate(targets)
            ]
            env.step(actions)
            episode_return += env.total_reward
            for u, target in enumerate(targets):
                if target == env.M or actions[u][0] == target:
                    targets[u] = None
        returns.append(episode_return)
    return float(np.mean(returns))


class HRLPPOTrainer:
    def __init__(self):
        self.training_started_at = time.perf_counter()
        self.rng = utils.RNGManager()
        self.initial_eval_rng = deepcopy(self.rng.get_state("eval"))
        torch.set_num_threads(HRL_PPO_SETTING.cpu_threads)
        self.env = Environment()
        self.env.reset()
        self.eval_env = Environment()
        self.upper_agents = [UpperAgent() for _ in range(self.env.U)]
        self.upper_buffers = [HRLUpperReplayBuffer() for _ in range(self.env.U)]
        self.lower = LowerPPO()
        self.lower_agents = self.lower.agents
        self.upper_decision_counts = np.zeros(self.env.U, dtype=int)
        self.upper_update_counts = np.zeros(self.env.U, dtype=int)
        self.lower_update_counts = np.zeros(self.env.U, dtype=int)
        self.navigation_episodes = int(
            utils.TRAIN_SETTING.episodes * HRL_PPO_SETTING.navigation_fraction
        )
        self.phase_counts = {"navigation": 0, "joint": 0}
        self.phase = "navigation"
        self.episode = 0
        self.pretrain_loss = []
        self.upper_prefill_env_steps = 0
        self.upper_prefill_buffer_sizes = [0 for _ in range(self.env.U)]
        self.files = HRLPPOFileIO(self)
        self.progress = Progress("HRL_PPO", self)
        self.record = Record("HRL_PPO", self)

    def pretrain_lower(self):
        if HRL_PPO_LOWER_SETTING.pretrain_batches == 0:
            return
        self.progress.set_stage("navigation_pretrain")
        rng = np.random.default_rng(utils.TRAIN_SETTING.seed)
        count = HRL_PPO_LOWER_SETTING.navigation_cases
        positions = rng.uniform(0, self.env.map_size, (count, self.env.U, 2))
        goals = rng.uniform(0, self.env.map_size, (count, 2))
        # Demonstrations teach only direction. Risk-aware improvement is on-policy.
        risk_features = risk_observation(self.env)
        risk = np.broadcast_to(risk_features, (count, len(risk_features)))
        uav_positions = positions.reshape(count, -1) / self.env.map_size
        for u, agent in enumerate(self.lower.agents):
            delta = goals - positions[:, u]
            states = np.concatenate((
                positions[:, u] / self.env.map_size,
                goals / self.env.map_size,
                delta / self.env.map_size,
                np.ones((count, 1)), uav_positions, risk,
            ), axis=1)
            teacher = delta / np.maximum(
                np.linalg.norm(delta, axis=1, keepdims=True), self.env.max_mov
            )
            self.pretrain_loss.append(agent.pretrain(states, teacher))

    def _choose_targets(self, *, count_decisions=True, random_targets=False):
        for u in range(self.env.U):
            if self.targets[u] is not None:
                continue
            state = upper_state(self.env, u)
            if random_targets:
                target = random.randrange(self.env.M + 1)
            elif self.phase == "navigation":
                target = random.randrange(self.env.M)
            else:
                fraction = min(
                    1.0, self.upper_decision_counts[u] / HRL_PPO_UPPER_SETTING.eps_decisions
                )
                epsilon = HRL_PPO_UPPER_SETTING.start_eps + fraction * (
                    HRL_PPO_UPPER_SETTING.min_eps - HRL_PPO_UPPER_SETTING.start_eps)
                target = self.upper_agents[u].get_action(state, epsilon)
                if count_decisions:
                    self.upper_decision_counts[u] += 1
            self.targets[u] = int(target)
            self.option_states[u] = state
            self.option_returns[u] = 0.0
            self.option_ages[u] = 0

    def _learn_upper(self):
        for u, buffer in enumerate(self.upper_buffers):
            if len(buffer) >= buffer.batch_size:
                loss = self.upper_agents[u].learn(buffer.sample())
                self.upper_losses.append(loss)
                self.upper_update_counts[u] += 1

    def _store_upper_fragments(self, *, episode_end, record_stats=True):
        """Store one step's completed options or continuing fragments.

        AOI rewards use bounded, discounted fragments while keeping the selected
        target locked. Environment reward modes store the whole option when it
        completes, or its continuing partial return at the episode boundary.
        """
        if HRL_PPO_UPPER_SETTING.reward_mode not in (
            "aoi_risk_count", "option_mean_environment", "average_reward_smdp"
        ):
            raise ValueError(
                f"Unknown HRL-PPO upper reward mode: {HRL_PPO_UPPER_SETTING.reward_mode}"
            )

        self.option_ends = []
        self.option_timeouts = [False] * self.env.U
        self.upper_fragment_stored = np.zeros(self.env.U, dtype=bool)
        self.upper_fragment_rewards = np.zeros(self.env.U, dtype=float)
        self.upper_fragment_durations = np.zeros(self.env.U, dtype=int)
        self.upper_next_states = [
            upper_state(self.env, u) for u in range(self.env.U)
        ]

        for u, target in enumerate(self.target_actions):
            active = target != self.env.M
            success = bool(active and self.env_actions[u][0] == target)
            ended = bool(not active or success)
            self.option_ends.append(ended)

            if not active:
                if HRL_PPO_UPPER_SETTING.reward_mode == "aoi_risk_count":
                    transition_reward = 0.0
                else:
                    transition_reward = float(self.env.total_reward)
                    self.upper_step_rewards[u] = transition_reward
                if self.phase == "joint":
                    self.upper_buffers[u].add(
                        self.option_states[u], [target], transition_reward,
                        self.upper_next_states[u], 1, True,
                    )
                    self.upper_fragment_stored[u] = True
                    self.upper_fragment_rewards[u] = transition_reward
                    self.upper_fragment_durations[u] = 1
                self.targets[u] = None
                self.option_states[u] = None
                self.option_returns[u] = 0.0
                self.option_ages[u] = 0
                continue

            if HRL_PPO_UPPER_SETTING.reward_mode == "aoi_risk_count":
                self.option_returns[u] += (
                    HRL_PPO_UPPER_SETTING.gamma ** self.option_ages[u]
                    * self.upper_step_rewards[u]
                )
                self.option_ages[u] += 1
                fragment_end = (
                    self.option_ages[u] >= HRL_PPO_SETTING.fragment_steps
                )
                should_store = success or fragment_end or episode_end
                if should_store and self.phase == "joint":
                    fragment_reward = float(self.option_returns[u])
                    duration = int(self.option_ages[u])
                    self.upper_buffers[u].add(
                        self.option_states[u], [target], fragment_reward,
                        self.upper_next_states[u], duration, success,
                    )
                    self.upper_fragment_stored[u] = True
                    self.upper_fragment_rewards[u] = fragment_reward
                    self.upper_fragment_durations[u] = duration
            else:
                self.option_returns[u] += float(self.env.total_reward)
                self.option_ages[u] += 1
                should_store = success or episode_end
                if should_store and self.phase == "joint":
                    duration = int(self.option_ages[u])
                    option_reward = float(self.option_returns[u])
                    if HRL_PPO_UPPER_SETTING.reward_mode == "option_mean_environment":
                        option_reward /= duration
                    self.upper_buffers[u].add(
                        self.option_states[u], [target], option_reward,
                        self.upper_next_states[u], duration, success,
                    )
                    self.upper_step_rewards[u] = option_reward
                    self.upper_fragment_stored[u] = True
                    self.upper_fragment_rewards[u] = option_reward
                    self.upper_fragment_durations[u] = duration

            if success:
                if record_stats:
                    self.reached += 1
                    self.attempts += 1
                self.targets[u] = None
                self.option_states[u] = None
                self.option_returns[u] = 0.0
                self.option_ages[u] = 0
            elif episode_end:
                if record_stats:
                    self.attempts += 1
                    self.uncompleted_options += 1
            elif (
                HRL_PPO_UPPER_SETTING.reward_mode == "aoi_risk_count"
                and self.option_ages[u] >= HRL_PPO_SETTING.fragment_steps
            ):
                # A fragment boundary starts a fresh replay segment while the
                # selected target remains active for the next primitive step.
                self.option_states[u] = self.upper_next_states[u].copy()
                self.option_returns[u] = 0.0
                self.option_ages[u] = 0

    def _reset_option_state(self):
        self.targets = [None] * self.env.U
        self.option_states = [None] * self.env.U
        self.option_ages = np.zeros(self.env.U, dtype=int)
        self.option_returns = np.zeros(self.env.U, dtype=float)

    def collect_random_upper(self):
        """Prefill all upper replay pools using random goals and sampled PPO moves."""
        required = max(
            HRL_PPO_UPPER_SETTING.warmup_steps,
            max(buffer.batch_size for buffer in self.upper_buffers),
        )
        self.upper_prefill_env_steps = 0
        self.env.reset()
        self.phase = "joint"
        while min(len(buffer) for buffer in self.upper_buffers) < required:
            self.env.reset()
            self._reset_option_state()
            self._choose_targets(count_decisions=False, random_targets=True)
            while self.env.steps < self.env.max_steps:
                states = lower_states(self.env, self.targets)
                active = np.asarray(self.targets) != self.env.M
                moves, _, _ = self.lower.sample(states, active)
                self.target_actions = list(self.targets)
                self.env_actions = [
                    to_env_action(self.env, u, target, moves[u])
                    for u, target in enumerate(self.targets)
                ]
                previous_aoi = self.env.AOI.copy()
                self.env.step(self.env_actions)
                episode_end = self.env.steps >= self.env.max_steps
                self.upper_step_rewards = upper_step_rewards(
                    previous_aoi, self.env, self.env_actions, active
                )
                self._store_upper_fragments(
                    episode_end=episode_end, record_stats=False
                )
                self.upper_prefill_env_steps += 1
                if episode_end:
                    break
                self._choose_targets(count_decisions=False, random_targets=True)

        self.upper_prefill_buffer_sizes = [
            len(buffer) for buffer in self.upper_buffers
        ]

    def _update_lower(self):
        update = self.lower.learn(self.rollout)
        self.lower_update_counts += np.asarray(update["actor_updates"], dtype=int)
        self.lower_metrics.append(update)
        self.rollout.clear()

    def run_episode(self):
        self.env.reset()
        self._reset_option_state()
        self.environment_return = 0.0
        self.upper_episode_reward = 0.0
        self.upper_agent_rewards = np.zeros(self.env.U)
        self.aoi_sum = 0.0
        self.lower_metrics = []
        self.upper_losses = []
        self.rollout = []
        self.reached = 0
        self.attempts = 0
        self.uncompleted_options = 0
        self.timeouts = 0
        self.evaluation_metrics = None
        self._choose_targets()
        self.files.episode_start()
        while self.env.steps < self.env.max_steps:
            states = lower_states(self.env, self.targets)
            central = critic_state(self.env, self.targets)
            active = np.asarray(self.targets) != self.env.M
            moves, latents, log_probs = self.lower.sample(states, active)
            value = float(self.lower.values(central))
            self.target_actions = list(self.targets)
            self.current_position_actions = (~active).tolist()
            distances = np.zeros(self.env.U)
            for u, target in enumerate(self.targets):
                if active[u]:
                    distances[u] = np.linalg.norm(self.env.device_coord[target]
                                                 - self.env.UAV_coord[u])
            self.env_actions = [to_env_action(self.env, u, target, moves[u])
                                for u, target in enumerate(self.targets)]
            previous_aoi = self.env.AOI.copy()
            self.env.step(self.env_actions)
            episode_end = self.env.steps >= self.env.max_steps
            self.lower_step_rewards = intrinsic_rewards(
                self.env, self.target_actions, distances, self.env_actions
            )
            self.upper_step_rewards = upper_step_rewards(
                previous_aoi, self.env, self.env_actions, active
            )
            self._store_upper_fragments(episode_end=episode_end)
            if self.phase == "joint":
                # The DQN updates from this primitive step before a new goal is
                # chosen; PPO/A2PO still updates only at rollout boundaries.
                self._learn_upper()
            self.lower_team_reward = (
                float(np.mean(self.lower_step_rewards))
                + HRL_PPO_SETTING.environment_reward_weight * self.env.total_reward
            )
            self.environment_return += self.env.total_reward
            self.upper_agent_rewards += self.upper_step_rewards
            self.upper_episode_reward += float(np.sum(self.upper_step_rewards))
            self.aoi_sum += float(np.mean(self.env.AOI))
            self.files.step()
            # Preserve unfinished targets; only completed options select a new
            # goal. A bootstrap-only selection at the time limit is not counted.
            self._choose_targets(count_decisions=not episode_end)
            next_value = float(self.lower.values(critic_state(self.env, self.targets)))
            self.rollout.append(dict(
                states=states, critic_state=central,
                latents=latents, log_probs=log_probs, active=active,
                reward=self.lower_team_reward, value=value, next_value=next_value,
                trace_end=episode_end,
            ))
            if len(self.rollout) >= HRL_PPO_LOWER_SETTING.rollout_steps or episode_end:
                self._update_lower()
            self.progress.step()

    def evaluate(self):
        self.rng.force_set_state("eval", deepcopy(self.initial_eval_rng))
        self.rng.switch_mode("eval")
        try:
            self.eval_environment_return = eval_runs(self.eval_env, self.upper_agents, self.lower)
            self.evaluation_metrics = deepcopy(self.eval_env.last_eval_metrics)
        finally:
            self.rng.switch_mode("train")

    def train(self):
        self.pretrain_lower()
        prefill_done = False
        for episode in range(1, utils.TRAIN_SETTING.episodes + 1):
            self.episode = episode
            self.phase = "joint"
            if episode <= self.navigation_episodes:
                self.phase = "navigation"
            elif not prefill_done:
                self.progress.set_stage("warmup")
                self.collect_random_upper()
                prefill_done = True
                self.phase = "joint"
            self.episode_started_at = time.perf_counter()
            self.progress.episode_start()
            self.run_episode()
            self.phase_counts[self.phase] += 1
            if episode % utils.EVAL_SETTING.eval_periods == 0 or episode == utils.TRAIN_SETTING.episodes:
                self.evaluate()
            self.record.episode()
            self.record.evaluation()
            self.files.episode()
            self.progress.episode_end()
        self.files.prepare()
        self.progress.set_stage("saving")
        self.files.save()


def train():
    HRLPPOTrainer().train()


if __name__ == "__main__":
    train()
