from runtime.progress import Progress, stage
from runtime.record import Record
from runtime.fileIO import FileIO
from runtime.training_evaluation import record_evaluation
from collections import deque
from copy import deepcopy
import time
import random
import numpy as np
import torch
import utils
from utils import (HRL_SETTING, HRL_UPPER_SETTING, HRL_LOWER_SETTING, RNGManager)
from Environment import Environment
from buffer import HRLLowerReplayBuffer, HRLUpperReplayBuffer
from agent_HRL import UpperAgent, LowerAgent


# 本檔案的索引約定：u 是 UAV 索引，U 是 UAV 數量，M 是裝置數量。
# 每個 upper_state 以焦點 UAV 為首；下層有效導航觀測為 (U, 6)，current-position 動作略過下層。


def _stable_window(values):
    """Check finite-value range against the configured relative scale."""
    if len(values) < HRL_SETTING.convergence_window:
        return False
    recent = np.asarray(  # 最近一個收斂視窗的指標序列。
        list(values)[-HRL_SETTING.convergence_window:], dtype=float
    )
    if not np.isfinite(recent).all():
        return False
    mean = float(np.mean(recent))  # 視窗平均值，供穩定度尺度化。
    relative_range = float(np.max(recent) - np.min(recent)) / max(1., abs(mean))  # 相對極差，避免小平均值放大波動。
    return relative_range <= HRL_SETTING.convergence_tolerance


# 下層保留目標條件導航的 6 維原始座標，不正規化或加入風險區資訊。
def lower_state(env, u, target_action):
    # 欄位順序：[UAV x/y, 目標 x/y, 目標相對 UAV 的 dx/dy]。
    return np.concatenate((
        env.UAV_coord[u],
        env.device_coord[target_action],
        env.device_coord[target_action] - env.UAV_coord[u],
    ))


# 上層 DQN 選 Planning 目標；下層輸出正規化 Moving，環境執行 [Service, 位移]。
def to_env_action(env, u, target_action, move_action):
    move_action = np.asarray(move_action, dtype=float)
    # 將正規化 Moving 縮放為實際位移，保留原有超長向量裁切方式。
    move_action = move_action / max(1., np.linalg.norm(move_action)) * env.max_mov
    next_coord = env.Update_Location(env.UAV_coord[u], move_action)  # 預計移動後、已限制於地圖內的座標。
    distance = np.linalg.norm(env.device_coord[target_action] - next_coord)  # 移動後到目標的距離。
    # 移動後接近目標才服務；M 表示本步不服務。
    if distance < env.map_size * HRL_SETTING.service_ratio:
        select_action = target_action
    else:
        select_action = env.M
    return [select_action, move_action]


# 產生 upper 階段訓練與評估使用的直線導航控制。
def straight_action(env, u, target_action):
    delta = env.device_coord[target_action] - env.UAV_coord[u]  # 從當前 UAV 指向目標的向量。
    distance = np.linalg.norm(delta)  # 移動前到目標的直線距離。
    # 遠處輸出單位方向，近處輸出距離比例，換成位移後剛好到達目標。
    return delta / max(distance, env.max_mov)

# 計算上層外部獎勵，同步服務同一裝置時平均分配降低量。
def aoi_reduction_rewards(previous_aoi, current_aoi, env_actions, service_success=None):
    # 以「本步自然增加後、尚未服務」為基準，不把自然老化算成負獎勵。
    reductions = np.maximum(  # 各裝置因本步服務減少的 AoI，shape=(M,)。
        np.asarray(previous_aoi) + 1. - current_aoi,
        0.,
    )
    select_actions = np.asarray([env_action[0] for env_action in env_actions])
    if service_success is not None:
        # 只將 AoI 降低量分給本步送達封包的鏈路。
        select_actions = np.where(service_success, select_actions, len(reductions))
    rewards = np.zeros(len(env_actions))  # 各 UAV 分得的上層獎勵，shape=(U,)。
    for select_action in np.unique(select_actions):
        if 0 <= select_action < len(reductions):
            selected = select_actions == select_action  # 同步服務目前裝置的 UAV 遮罩。
            # 同步服務同一裝置時平均分配，避免重複計算且不依 UAV 順序分配。
            rewards[selected] = reductions[select_action] / np.count_nonzero(selected)
    return rewards


# 計算下層內部獎勵：接近目標、完成服務、耗時及實際風險。
def intrinsic_rewards(env, target_actions, before_distances, env_actions):
    rewards = []  # 依 UAV 順序保存本步下層導航獎勵。
    for u, target_action in enumerate(target_actions):
        after_distance = np.linalg.norm(env.device_coord[target_action] - env.UAV_coord[u])  # 移動後到原目標的距離。
        progress = (before_distances[u] - after_distance) / env.max_mov  # 以單步最大位移尺度化的接近量，可為負。
        # 規則服務觸發後，代表下層已完成導航目標。
        finished = env_actions[u][0] == target_action
        # 直接使用環境本步已抽樣的懲罰事件，保留原本的懲罰金額。
        risk_penalty = env.risk_penalty_counts[u] * env.penalty
        rewards.append(
            HRL_SETTING.progress_weight * progress
            + HRL_SETTING.success_reward * finished
            - HRL_SETTING.step_penalty
            - HRL_SETTING.risk_weight * risk_penalty / HRL_SETTING.risk_penalty_reference
        )
    return rewards


def upper_state(env, u):
    """Build one normalized observation with UAV ``u`` as the focal UAV."""
    environment_state = env.get_state()
    uav_end = 2 * env.U
    device_xy_end = uav_end + 2 * env.M
    return np.concatenate(
        (
            env.UAV_coord[u] / env.map_size,
            environment_state[:uav_end] / env.map_size,
            environment_state[uav_end:device_xy_end] / env.map_size,
            environment_state[device_xy_end:] / utils.TRAIN_SETTING.max_steps,
        )
    ).astype(float, copy=False)


# 訓練器持有環境與學習狀態；各階段入口保留自己的動作及更新順序。
class HRLTrainer:
    """Train one independent upper and lower policy for each UAV."""

    def __init__(self):
        self.training_started_at = time.perf_counter()
        self.episode_started_at = None
        self.upper_min_episodes = int(
            utils.TRAIN_SETTING.episodes * HRL_SETTING.upper_fraction
        )
        self.lower_min_episodes = int(
            utils.TRAIN_SETTING.episodes * HRL_SETTING.lower_fraction
        )

        self.rng = RNGManager()
        self.initial_eval_rng = deepcopy(self.rng.get_state("eval"))
        torch.set_num_threads(HRL_SETTING.cpu_threads)

        self.env = Environment()
        self.env.reset()
        self.upper_agents = [UpperAgent() for _ in range(self.env.U)]
        self.upper_buffers = [
            HRLUpperReplayBuffer() for _ in range(self.env.U)
        ]

        lower_prototype = LowerAgent()
        self.lower_agents = [lower_prototype]
        self.lower_agents.extend(
            deepcopy(lower_prototype) for _ in range(1, self.env.U)
        )
        if HRL_LOWER_SETTING.algorithm == "a2c":
            self.lower_buffers = [None for _ in range(self.env.U)]
        else:
            self.lower_buffers = [
                HRLLowerReplayBuffer() for _ in range(self.env.U)
            ]

        self.upper_decision_counts = np.zeros(self.env.U, dtype=int)
        self.episode = 0
        self.phase = None
        self.phase_counts = {"upper": 0, "lower": 0, "joint": 0}
        self.upper_converged = False
        self.lower_converged = False
        self.upper_eval_returns = deque(maxlen=HRL_SETTING.convergence_window)
        self.lower_reach_rates = deque(maxlen=HRL_SETTING.convergence_window)
        self.upper_stable_checks = 0
        self.lower_stable_checks = 0
        self.upper_update_counts = [0 for _ in range(self.env.U)]
        self.total_upper_updates = 0
        self.lower_update_counts = [0 for _ in range(self.env.U)]
        self.total_lower_updates = 0
        self.pending_initialization_elapsed_seconds = 0.0

        self.progress = Progress("HRL", self)
        self.record = Record("HRL", self)
        self.files = FileIO("HRL", self)

    def _reset_option_state(self):
        self.locked_target_actions = [None] * self.env.U
        self.upper_fragment_states = [None] * self.env.U
        self.upper_fragment_rewards = np.zeros(self.env.U)
        self.upper_fragment_steps = np.zeros(self.env.U, dtype=int)

    def _reset_episode_state(self):
        self.env.reset()
        self._reset_option_state()
        self.upper_episode_reward = 0.0
        self.aoi_sum = 0.0
        self.environment_return = 0.0
        self.upper_agent_rewards = np.zeros(self.env.U)
        self.files.episode_start()

    def _choose_upper_targets(self):
        for u in range(self.env.U):
            if self.locked_target_actions[u] is not None:
                continue
            fraction = min(
                1.0, self.upper_decision_counts[u] / HRL_SETTING.eps_decisions
            )
            epsilon = HRL_SETTING.start_eps + fraction * (
                HRL_SETTING.min_eps - HRL_SETTING.start_eps
            )
            self.upper_decision_counts[u] += 1
            state = upper_state(self.env, u)
            self.locked_target_actions[u] = int(
                self.upper_agents[u].get_action(state, epsilon=epsilon)[0]
            )
            self.upper_fragment_states[u] = state.copy()
            self.upper_fragment_rewards[u] = 0.0
            self.upper_fragment_steps[u] = 0

    def _prepare_lower_observations(self):
        self.target_actions = list(self.locked_target_actions)
        self.current_position_actions = []
        self.lower_states = []
        self.before_distances = []
        for u, target_action in enumerate(self.target_actions):
            current_position = target_action == self.env.M
            self.current_position_actions.append(current_position)
            if current_position:
                self.lower_states.append(None)
                self.before_distances.append(None)
            else:
                self.lower_states.append(
                    lower_state(self.env, u, target_action)
                )
                distance = np.linalg.norm(
                    self.env.device_coord[target_action] - self.env.UAV_coord[u]
                )
                self.before_distances.append(float(distance))
        if all(state is not None for state in self.lower_states):
            self.lower_states = np.stack(self.lower_states)

    def _execute_environment_step(self, move_actions):
        self.move_actions = np.asarray(move_actions, dtype=float).reshape(
            self.env.U, 2
        )
        self.env_actions = []
        for u, target_action in enumerate(self.target_actions):
            if self.current_position_actions[u]:
                # This candidate is one no-service, zero-movement step.
                self.move_actions[u] = 0.0
                self.env_actions.append([self.env.M, np.zeros(2, dtype=float)])
            else:
                self.env_actions.append(
                    to_env_action(
                        self.env, u, target_action, self.move_actions[u]
                    )
                )

        previous_aoi = self.env.AOI.copy()
        self.env.step(self.env_actions)
        aoi_rewards = aoi_reduction_rewards(
            previous_aoi, self.env.AOI, self.env_actions,
            getattr(self.env, "service_success", None),
        )
        self.upper_step_rewards = np.zeros(self.env.U)
        reward_mode = HRL_UPPER_SETTING.reward_mode
        if self.phase in ("upper", "joint"):
            for u, target_action in enumerate(self.target_actions):
                if self.current_position_actions[u]:
                    continue
                if self.upper_fragment_states[u] is None:
                    continue
                if reward_mode == "aoi_risk_count":
                    self.upper_step_rewards[u] = (
                        aoi_rewards[u]
                        - HRL_UPPER_SETTING.risk_penalty_weight
                        * self.env.risk_penalty_counts[u]
                    )
                elif reward_mode in (
                    "option_mean_environment", "average_reward_smdp"
                ):
                    self.upper_step_rewards[u] = 0.0
                else:
                    raise ValueError(
                        f"Unknown HRL upper reward mode: {reward_mode}"
                    )

        self.lower_next_states = []
        for u, target_action in enumerate(self.target_actions):
            if self.current_position_actions[u]:
                self.lower_next_states.append(None)
            else:
                self.lower_next_states.append(
                    lower_state(self.env, u, target_action)
                )
        if all(state is not None for state in self.lower_next_states):
            self.lower_next_states = np.stack(self.lower_next_states)
        self.lower_step_rewards = [0.0] * self.env.U
        for u, target_action in enumerate(self.target_actions):
            if self.current_position_actions[u]:
                continue
            after_distance = np.linalg.norm(
                self.env.device_coord[target_action] - self.env.UAV_coord[u]
            )
            progress = (self.before_distances[u] - after_distance) / self.env.max_mov
            finished = self.env_actions[u][0] == target_action
            risk_penalty = self.env.risk_penalty_counts[u] * self.env.penalty
            self.lower_step_rewards[u] = (
                HRL_SETTING.progress_weight * progress
                + HRL_SETTING.success_reward * finished
                - HRL_SETTING.step_penalty
                - HRL_SETTING.risk_weight
                * risk_penalty / HRL_SETTING.risk_penalty_reference
            )

        self.upper_next_state = [
            upper_state(self.env, u) for u in range(self.env.U)
        ]
        self.option_ends = []
        for u in range(self.env.U):
            if self.current_position_actions[u]:
                self.option_ends.append(True)
            else:
                self.option_ends.append(
                    self.env_actions[u][0] == self.target_actions[u]
                )
        self.episode_end = self.env.steps >= self.env.max_steps

    def _store_upper_fragments(self):
        if self.phase not in ("upper", "joint"):
            return
        reward_mode = HRL_UPPER_SETTING.reward_mode
        if reward_mode not in (
            "aoi_risk_count", "option_mean_environment", "average_reward_smdp"
        ):
            raise ValueError(f"Unknown HRL upper reward mode: {reward_mode}")

        for u, target_action in enumerate(self.target_actions):
            start_state = self.upper_fragment_states[u]
            if start_state is None:
                continue
            next_state = self.upper_next_state[u]

            if self.current_position_actions[u]:
                if reward_mode == "aoi_risk_count":
                    option_reward = 0.0
                else:
                    option_reward = float(self.env.total_reward)
                    self.upper_step_rewards[u] = option_reward
                self.upper_buffers[u].add(
                    start_state,
                    [target_action],
                    option_reward,
                    next_state,
                    1,
                    True,
                )
                self.upper_fragment_states[u] = None
                self.upper_fragment_rewards[u] = 0.0
                self.upper_fragment_steps[u] = 0
                continue

            if reward_mode == "aoi_risk_count":
                self.upper_fragment_rewards[u] += (
                    HRL_UPPER_SETTING.gamma
                    ** self.upper_fragment_steps[u]
                    * self.upper_step_rewards[u]
                )
                self.upper_fragment_steps[u] += 1
                fragment_end = (
                    self.upper_fragment_steps[u] >= HRL_SETTING.fragment_steps
                )
                if not (
                    self.option_ends[u] or fragment_end or self.episode_end
                ):
                    continue
                self.upper_buffers[u].add(
                    start_state,
                    [target_action],
                    self.upper_fragment_rewards[u],
                    next_state,
                    self.upper_fragment_steps[u],
                    self.option_ends[u],
                )
                if self.option_ends[u]:
                    self.upper_fragment_states[u] = None
                    self.upper_fragment_rewards[u] = 0.0
                    self.upper_fragment_steps[u] = 0
                else:
                    # Fragment boundaries preserve the same locked device and
                    # focal UAV in the next replay state.
                    self.upper_fragment_states[u] = next_state.copy()
                    self.upper_fragment_rewards[u] = 0.0
                    self.upper_fragment_steps[u] = 0
                continue

            self.upper_fragment_rewards[u] += float(self.env.total_reward)
            self.upper_fragment_steps[u] += 1
            if not (self.option_ends[u] or self.episode_end):
                continue
            duration = self.upper_fragment_steps[u]
            option_reward = self.upper_fragment_rewards[u]
            if reward_mode == "option_mean_environment":
                option_reward /= duration
            self.upper_buffers[u].add(
                start_state,
                [target_action],
                option_reward,
                next_state,
                duration,
                self.option_ends[u],
            )
            self.upper_step_rewards[u] = option_reward
            self.upper_fragment_states[u] = None
            self.upper_fragment_rewards[u] = 0.0
            self.upper_fragment_steps[u] = 0

    def collect_random_upper(self):
        required = max(
            HRL_UPPER_SETTING.warmup_steps,
            max(buffer.batch_size for buffer in self.upper_buffers),
        )
        self.env.reset()
        self._reset_option_state()
        while min(len(buffer) for buffer in self.upper_buffers) < required:
            while self.env.steps < self.env.max_steps:
                for u in range(self.env.U):
                    if self.locked_target_actions[u] is not None:
                        continue
                    target_action = random.randrange(self.env.M + 1)
                    self.locked_target_actions[u] = target_action
                    self.upper_fragment_states[u] = upper_state(self.env, u)
                    self.upper_fragment_rewards[u] = 0.0
                    self.upper_fragment_steps[u] = 0

                self._prepare_lower_observations()
                if self.phase == "upper":
                    move_actions = np.stack(
                        [
                            np.zeros(2)
                            if target_action == self.env.M
                            else straight_action(self.env, u, target_action)
                            for u, target_action in enumerate(self.target_actions)
                        ]
                    )
                elif self.phase == "joint":
                    move_actions = self._choose_lower_actions(
                        deterministic=HRL_LOWER_SETTING.algorithm == "a2c"
                    )
                else:
                    raise ValueError(
                        "collect_random_upper requires upper or joint phase"
                    )
                self._execute_environment_step(move_actions)
                self._store_upper_fragments()
                for u in range(self.env.U):
                    if self.option_ends[u]:
                        self.locked_target_actions[u] = None
                if self.episode_end:
                    break
            self.env.reset()
            self._reset_option_state()

    def _learn_lower(self):
        if self.phase == "joint":
            joint_episodes = self.phase_counts["joint"]
        else:
            joint_episodes = 0
        for u in range(self.env.U):
            state = self.lower_states[u]
            if state is None:
                continue
            next_state = self.lower_next_states[u]
            delta = state[4:6]
            teacher_action = delta / max(float(np.linalg.norm(delta)), self.env.max_mov)
            action = self.move_actions[u]
            experience = (
                np.asarray(state).reshape(1, -1),
                np.asarray(action).reshape(1, -1),
                np.asarray([self.lower_step_rewards[u]], dtype=float),
                np.asarray(next_state).reshape(1, -1),
                np.asarray([self.option_ends[u]], dtype=float),
                np.asarray(teacher_action).reshape(1, -1),
            )
            if HRL_LOWER_SETTING.algorithm == "a2c":
                update = self.lower_agents[u].learn(
                    experience,
                    rollout_end=self.option_ends[u] or self.episode_end,
                    joint_episodes=joint_episodes,
                )
                if update is not None:
                    self.lower_update_counts[u] += 1
                    self.total_lower_updates += 1
                continue

            self.lower_buffers[u].add(
                state,
                action,
                self.lower_step_rewards[u],
                next_state,
                self.option_ends[u],
                teacher_action,
            )
            if len(self.lower_buffers[u]) >= self.lower_buffers[u].batch_size:
                self.lower_agents[u].learn(
                    self.lower_buffers[u].sample(),
                    joint_episodes=joint_episodes,
                )
                self.lower_update_counts[u] += 1
                self.total_lower_updates += 1

    def _choose_lower_actions(self, deterministic=False):
        actions = np.zeros((self.env.U, 2), dtype=float)
        for u, target_action in enumerate(self.target_actions):
            if self.current_position_actions[u]:
                continue
            state = np.asarray(self.lower_states[u]).reshape(1, -1)
            action = self.lower_agents[u].get_action(
                state, deterministic=deterministic
            )
            actions[u] = np.asarray(action)[0]
        return actions

    def _learn_upper(self):
        # Update each eligible UAV once per primitive environment step.
        for u in range(self.env.U):
            buffer = self.upper_buffers[u]
            if len(buffer) >= buffer.batch_size:
                self.upper_agents[u].learn(buffer.sample())
                self.upper_update_counts[u] += 1
                self.total_upper_updates += 1

    def _record_step_metrics(self):
        self.upper_episode_reward += float(np.sum(self.upper_step_rewards))
        self.upper_agent_rewards += self.upper_step_rewards
        self.aoi_sum += float(np.mean(self.env.AOI))
        self.environment_return += self.env.total_reward
        self.files.step()
        for u in range(self.env.U):
            if self.option_ends[u]:
                self.locked_target_actions[u] = None

    def _evaluate_policy_and_upper_convergence(self):
        scheduled_evaluation = (
            self.episode % utils.EVAL_SETTING.eval_periods == 0
        )
        upper_convergence_check = self.phase == "upper"
        if not scheduled_evaluation and not upper_convergence_check:
            return None
        if upper_convergence_check:
            self.rng.force_set_state("eval", deepcopy(self.initial_eval_rng))
        self.rng.switch_mode("eval")
        try:
            self.eval_environment_return, self.eval_mean_aoi = eval_runs(
                self.env,
                self.upper_agents,
                self.lower_agents,
                self.phase,
                return_metrics=True,
            )
        finally:
            self.rng.switch_mode("train")

        if upper_convergence_check:
            self.upper_eval_returns.append(self.eval_environment_return)
            upper_stable = _stable_window(self.upper_eval_returns)
            eligible = (
                self.phase_counts["upper"] >= HRL_SETTING.convergence_min_episodes
                and self.total_upper_updates > 0
            )
            if eligible and upper_stable:
                self.upper_stable_checks += 1
            else:
                self.upper_stable_checks = 0
            self.upper_converged = (
                self.upper_stable_checks >= HRL_SETTING.convergence_patience
            )
        return self.env.last_eval_metrics

    def train_upper(self):
        self._reset_episode_state()
        while self.env.steps < self.env.max_steps:
            self._choose_upper_targets()
            self._prepare_lower_observations()
            move_actions = np.stack(
                [
                    np.zeros(2)
                    if target_action == self.env.M
                    else straight_action(self.env, u, target_action)
                    for u, target_action in enumerate(self.target_actions)
                ]
            )
            self._execute_environment_step(move_actions)
            self._store_upper_fragments()
            self._learn_upper()
            self._record_step_metrics()
            self.progress.step()

    def train_joint(self):
        self._reset_episode_state()
        while self.env.steps < self.env.max_steps:
            self._choose_upper_targets()
            self._prepare_lower_observations()
            move_actions = self._choose_lower_actions()
            self._execute_environment_step(move_actions)
            self._store_upper_fragments()
            self._learn_lower()
            self._learn_upper()
            self._record_step_metrics()
            self.progress.step()

    def collect_lower(self):
        if HRL_LOWER_SETTING.algorithm == "a2c":
            return
        if HRL_LOWER_SETTING.warmup_steps == 0:
            return
        required = max(
            HRL_LOWER_SETTING.warmup_steps,
            max(buffer.batch_size for buffer in self.lower_buffers),
        )
        self.env.reset()
        self.locked_target_actions = [None] * self.env.U
        while min(len(buffer) for buffer in self.lower_buffers) < required:
            if self.env.steps >= self.env.max_steps:
                self.env.reset()
                self.locked_target_actions = [None] * self.env.U
            for u in range(self.env.U):
                if self.locked_target_actions[u] is None:
                    self.locked_target_actions[u] = random.randrange(self.env.M)
            self._prepare_lower_observations()
            delta = self.lower_states[:, 4:6]
            teacher_actions = delta / np.maximum(
                np.linalg.norm(delta, axis=1, keepdims=True), self.env.max_mov,
            )
            # 未執行離線預訓練時，預填一律使用均勻隨機動作。
            if (HRL_SETTING.pretrain_batches <= 0
                    or HRL_LOWER_SETTING.warmup_mode == "random"):
                move_actions = np.random.uniform(-1., 1., size=(self.env.U, 2))
            else:
                move_actions = teacher_actions.copy()
            self._execute_environment_step(move_actions)
            for u in range(self.env.U):
                self.lower_buffers[u].add(
                    self.lower_states[u], move_actions[u], self.lower_step_rewards[u],
                    self.lower_next_states[u], self.option_ends[u], teacher_actions[u],
                )
                if self.option_ends[u]:
                    self.locked_target_actions[u] = None
            if self.env.steps >= self.env.max_steps:
                self.env.reset()
                self.locked_target_actions = [None] * self.env.U

    def evaluate_and_record_episode(self):
        """更新回合計數、回報結果、評估與檢查收斂，最後保存回合指標。"""
        self.episode_steps = self.env.steps  # 保留訓練步數，避免被後續評估重設。
        self.phase_counts[self.phase] += 1
        self.progress.episode_end()
        self.record.episode()
        self.evaluation_metrics = self._evaluate_policy_and_upper_convergence()  # 策略評估指標；本回合未評估時為 None。
        self.record.evaluation()
        self.navigation_metrics = self._check_navigation_convergence()  # 固定導航案例指標；upper 階段為空字典。
        self.files.episode()
        self.pending_initialization_elapsed_seconds = 0.

    def _check_navigation_convergence(self):
        if self.phase == "upper":
            return {}
        with stage("navigation_check"):
            score = navigation_scores(self.env, self.lower_agents)  # 以最弱 agent 的到達率與最慢成功步數檢查導航。

        if self.phase == "lower":
            self.lower_reach_rates.append(
                min(score["deterministic"], score["sampled"])
            )
            lower_stable = (  # 視窗內到達率皆達標，且波動符合穩定度要求。
                min(self.lower_reach_rates) >= HRL_SETTING.reach_threshold
                and _stable_window(self.lower_reach_rates)
            )
            eligible = (  # 最低回合數已達標，且有下層 RL 更新或示範初始化。
                self.phase_counts["lower"] >= HRL_SETTING.convergence_min_episodes
                and (
                    self.total_lower_updates > 0
                    or HRL_SETTING.pretrain_batches > 0
                )
            )
            if eligible and lower_stable:
                self.lower_stable_checks += 1
            else:
                self.lower_stable_checks = 0
            self.lower_converged = (
                self.lower_stable_checks >= HRL_SETTING.convergence_patience
            )
        return score

    def pretrain_lower(self):
        if HRL_SETTING.pretrain_batches <= 0:
            return
        positions, goals = navigation_cases(  # 離線示範的起點、目標座標，皆為 (案例數, 2)。
            self.env, HRL_SETTING.navigation_cases,
            utils.TRAIN_SETTING.seed,
        )
        delta = goals - positions  # 每筆示範的目標相對起點向量。
        teacher_actions = delta / np.maximum(  # 下層 actor 要模仿的正規化直線動作。
            np.linalg.norm(delta, axis=1, keepdims=True), self.env.max_mov,
        )
        self.lower_agents[0].Pretrain(navigation_states(positions, goals), teacher_actions)
        for u in range(1, self.env.U):
            self.lower_agents[u] = deepcopy(self.lower_agents[0])

    def train_lower(self):
        self._reset_episode_state()
        while self.env.steps < self.env.max_steps:
            for u in range(self.env.U):
                if self.locked_target_actions[u] is None:
                    # 下層階段隨機指定目標，讓導航不依賴上層策略。
                    self.locked_target_actions[u] = random.randrange(self.env.M)
            self._prepare_lower_observations()
            move_actions = self._choose_lower_actions()
            self._execute_environment_step(move_actions)
            self._learn_lower()
            self._record_step_metrics()
            self.progress.step()


# 關閉探索與模型更新，回傳與 DQN 相同的累積環境獎勵指標。
@record_evaluation
def eval_runs(env, upper_agents, lower_policies, phase="joint", return_metrics=False):
    if len(upper_agents) != env.U:
        raise ValueError("HRL evaluation requires one upper agent per UAV")
    if phase != "upper" and isinstance(lower_policies, list):
        if len(lower_policies) == env.U:
            lower_mode = "per_agent"
        elif len(lower_policies) == 1:
            lower_mode = "shared"
        else:
            raise ValueError(
                "eval_runs lower policy list must contain one shared policy "
                "or one policy per UAV"
            )
    else:
        lower_mode = "single"

    rewards = []
    aois = []
    for _ in range(utils.EVAL_SETTING.eval_runs):
        env.reset()
        locked_target_actions = [None] * env.U
        aoi_sum = 0.0
        environment_return = 0.0
        while env.steps < env.max_steps:
            for u in range(env.U):
                if locked_target_actions[u] is not None:
                    continue
                if phase == "lower":
                    locked_target_actions[u] = random.randrange(env.M)
                else:
                    state = upper_state(env, u)
                    locked_target_actions[u] = int(
                        upper_agents[u].get_action(state, epsilon=0.0)[0]
                    )

            target_actions = list(locked_target_actions)
            move_actions = np.zeros((env.U, 2), dtype=float)
            for u, target_action in enumerate(target_actions):
                if target_action == env.M:
                    continue
                if phase == "upper":
                    move_actions[u] = straight_action(env, u, target_action)
                    continue

                state = lower_state(env, u, target_action).reshape(1, -1)
                if lower_mode == "per_agent":
                    lower_policy = lower_policies[u]
                elif lower_mode == "shared":
                    lower_policy = lower_policies[0]
                else:
                    lower_policy = lower_policies
                move_actions[u] = lower_policy.get_action(
                    state, deterministic=True
                )[0]

            env_actions = []
            for u, target_action in enumerate(target_actions):
                if target_action == env.M:
                    env_actions.append([env.M, np.zeros(2, dtype=float)])
                else:
                    env_actions.append(
                        to_env_action(env, u, target_action, move_actions[u])
                    )
            env.step(env_actions)
            option_ends = []
            for u, target_action in enumerate(target_actions):
                if target_action == env.M:
                    option_ends.append(True)
                else:
                    option_ends.append(
                        env_actions[u][0] == target_action
                    )
            aoi_sum += float(np.mean(env.AOI))
            environment_return += env.total_reward
            for u in range(env.U):
                if option_ends[u]:
                    locked_target_actions[u] = None

        rewards.append(environment_return)
        aois.append(aoi_sum / env.steps)

    mean_aoi = float(np.mean(aois))
    environment_return = float(np.mean(rewards))
    if return_metrics:
        return environment_return, mean_aoi
    return environment_return


# 以獨立 NumPy RNG 產生導航案例，不改變環境訓練的隨機序列。
def navigation_cases(env, count, seed):
    rng = np.random.default_rng(seed)  # 僅供導航案例使用的局部 RNG。
    positions = rng.uniform(0., env.map_size, (count, 2))  # count 筆起點座標。
    goals = rng.uniform(0., env.map_size, (count, 2))  # 與各起點配對的目標座標。
    return positions, goals


# 批次建立導航案例的下層觀測，欄位順序與 lower_state 相同。
def navigation_states(positions, goals):
    # 每列為一個案例的 [起點 x/y, 目標 x/y, 相對位移 dx/dy]。
    return np.concatenate((positions, goals, goals - positions), axis=1)


# 分別測量確定性與採樣控制的到達率、成功案例平均步數。
def navigation_score(env, lower_agent):
    # 固定驗證案例、不參與示範訓練；保存 torch RNG，避免影響訓練探索。
    starts, goals = navigation_cases(  # 確定性與採樣模式共用的固定驗證起點、目標。
        env, HRL_SETTING.reach_cases, HRL_SETTING.navigation_eval_seed)
    scores = {}  # 模式名稱對應到達率；名稱加 _steps 對應成功案例的平均步數。
    if utils.TRAIN_SETTING.device == "cuda":
        cuda_devices = list(range(torch.cuda.device_count()))  # fork_rng 需保存及恢復 RNG 的 CUDA 裝置。
    else:
        cuda_devices = []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(HRL_SETTING.navigation_eval_seed)
        for name, deterministic in (("deterministic", True), ("sampled", False)):
            # name 是結果字典的模式鍵；deterministic 決定取策略均值或採樣。
            positions = starts.copy()  # 此模式下各案例的當前位置，每步更新未抵達案例。
            reached = np.zeros(len(starts), dtype=bool)  # 各案例是否曾經抵達目標。
            first_steps = np.zeros(len(starts), dtype=int)  # 首次成功步數；未成功時保留 0。
            for step in range(1, HRL_SETTING.reach_steps + 1):
                move_actions = lower_agent.get_action(
                    navigation_states(positions, goals), deterministic
                )
                displacements = move_actions / np.maximum(  # 依環境最大位移縮放的實際移動向量。
                    1., np.linalg.norm(move_actions, axis=1, keepdims=True)
                ) * env.max_mov
                # 只推進未抵達案例，保留已抵達案例的首次成功步數。
                positions[~reached] = np.clip(
                    positions[~reached] + displacements[~reached],
                    env.L_corner,
                    env.H_corner,
                )
                # success 是本步位於目標服務範圍內的案例遮罩。
                success = np.linalg.norm(positions - goals, axis=1) < env.map_size * HRL_SETTING.service_ratio
                first_steps[success & ~reached] = step
                reached |= success
                if reached.all():
                    break
            scores[name] = float(reached.mean())
            if reached.any():
                scores[name + "_steps"] = float(first_steps[reached].mean())
            else:
                # 沒有成功案例時，平均到達步數沒有定義。
                scores[name + "_steps"] = np.nan
    return scores


def navigation_scores(env, lower_policies):
    """Evaluate each independent lower policy on the same fixed navigation cases."""
    per_agent = [navigation_score(env, lower_agent) for lower_agent in lower_policies]
    if not per_agent:
        raise ValueError("navigation_scores requires at least one lower policy")

    scores = {"per_agent": per_agent}
    for mode in ("deterministic", "sampled"):
        scores[mode] = min(agent_score[mode] for agent_score in per_agent)
        step_values = [agent_score[mode + "_steps"] for agent_score in per_agent]
        finite_steps = [value for value in step_values if np.isfinite(value)]
        if finite_steps:
            scores[mode + "_steps"] = max(finite_steps)
        else:
            scores[mode + "_steps"] = np.nan
    return scores


# 訓練入口：初始化、分階段訓練、定期評估，最後保存結果。
def train():
    trainer = HRLTrainer()

    trainer.phase = "upper"
    if (
        trainer.upper_min_episodes > 0
        and trainer.episode < utils.TRAIN_SETTING.episodes
    ):
        trainer.progress.set_stage("warmup")
        trainer.collect_random_upper()
    while (
        trainer.phase_counts["upper"] < trainer.upper_min_episodes
        and trainer.episode < utils.TRAIN_SETTING.episodes
    ):
        trainer.episode_started_at = time.perf_counter()
        trainer.episode += 1
        trainer.progress.episode_start()
        trainer.train_upper()
        trainer.evaluate_and_record_episode()

    trainer.phase = "pretrain"
    if trainer.episode < utils.TRAIN_SETTING.episodes:
        initialization_started_at = time.perf_counter()
        trainer.progress.set_stage("navigation_pretrain")
        trainer.pretrain_lower()
        trainer.pending_initialization_elapsed_seconds += (
            time.perf_counter() - initialization_started_at
        )

    trainer.phase = "lower"
    if (
        trainer.lower_min_episodes > 0
        and trainer.episode < utils.TRAIN_SETTING.episodes
    ):
        trainer.progress.set_stage("warmup")
        trainer.collect_lower()
    while (
        trainer.phase_counts["lower"] < trainer.lower_min_episodes
        and trainer.episode < utils.TRAIN_SETTING.episodes
    ):
        trainer.episode_started_at = time.perf_counter()
        trainer.episode += 1
        trainer.progress.episode_start()
        trainer.train_lower()
        trainer.evaluate_and_record_episode()

    trainer.phase = "joint"
    for buffer in trainer.upper_buffers:
        buffer.memory.clear()
    if trainer.episode < utils.TRAIN_SETTING.episodes:
        trainer.progress.set_stage("warmup")
        trainer.collect_random_upper()
    while trainer.episode < utils.TRAIN_SETTING.episodes:
        trainer.episode_started_at = time.perf_counter()
        trainer.episode += 1
        trainer.progress.episode_start()
        trainer.train_joint()
        trainer.evaluate_and_record_episode()

    trainer.files.prepare()
    trainer.progress.set_stage("saving")
    trainer.files.save()



if __name__ == "__main__":
    train()
