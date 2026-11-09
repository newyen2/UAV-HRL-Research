"""依方法解析訓練狀態，統一管理檔案讀取、記錄與保存。"""

import json
import math
import os
import time
from pathlib import Path


HRL_METHODS = ("HRL",)
HISTORICAL_RESULT_METHODS = frozenset({
    "DQN_greedy", "HRL_original", "HRL_risk", "HRL_uav_risk", "HRL_RDSAC",
    "HRL-SA", "HRL-MA",
})


def _require_method(method):
    if method not in ("DQN", "DQN_service", *HRL_METHODS, "HRL_PPO", "MA-CQR"):
        raise ValueError(f"Unsupported training save method: {method!r}")


def _episode_record(episode, env):
    """建立非 HRL 方法共用的初始場景記錄。"""
    record = {
        "episode": episode,
        "init_UAV_pos": env.UAV_coord.tolist(),
        "init_device_pos": env.device_coord.tolist(),
        "init_device_AOI": env.AOI.tolist(),
        "init_source_AOI": env.source_AOI.tolist(),
        "risk_region": env.risk_region.tolist(),
        "map_size": env.map_size,
    }
    record.update(_communication_episode_record(env))
    return record


def _trace_value(value):
    """Convert NumPy values to Arrow-friendly trace values when available."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _communication_episode_record(env):
    """Return optional communication metadata without requiring it on old envs."""
    model = getattr(env, "communication_model", None)
    if model is None:
        return {}
    return {"communication_model": _trace_value(model)}


def _communication_step_record(env):
    """Return optional per-step link telemetry when the environment exposes it."""
    if getattr(env, "communication_model", None) is None:
        return {}
    fields = {
        "communication_scheme": "communication_scheme",
        "UAV_service_success": "service_success",
        "UAV_transmission_power": "transmission_power",
        "UAV_transmission_peak_power": "transmission_peak_power",
        "UAV_transmission_time": "transmission_time",
    }
    record = {}
    for trace_column, env_attribute in fields.items():
        if not hasattr(env, env_attribute):
            continue
        value = getattr(env, env_attribute)
        if value is not None:
            record[trace_column] = _trace_value(value)
    return record


def _step_record(method, episode, env, env_actions, service_actions=None, *,
                 target_actions=None):
    """建立非 HRL 方法共用的逐步 trace 記錄。"""
    if method == "DQN_service":
        if target_actions is None:
            raise ValueError("DQN_service step records require target_actions")
        if service_actions is None:
            raise ValueError("DQN_service step records require service_actions")
        record = {
            "episode": episode,
            "step": env.steps,
            "UAV_pos": env.UAV_coord.tolist(),
            "device_pos": env.device_coord.tolist(),
            "device_AOI": env.AOI.tolist(),
            "source_AOI": env.source_AOI.tolist(),
            "UAV_target": [int(target_action) for target_action in target_actions],
            # 舊欄名保存本步二值服務嘗試（含強制嘗試），不是服務權限。
            "UAV_allow_service": list(service_actions),
            "UAV_action_select": [int(env_action[0]) for env_action in env_actions],
            "UAV_action_move": [env_action[1].tolist() for env_action in env_actions],
            "reward": list(env.rewards),
            "total_reward": env.total_reward,
        }
        record.update(_communication_step_record(env))
        return record
    if method == "MA-CQR":
        action_select = [env_action[0] for env_action in env_actions]
    else:
        action_select = [int(env_action[0]) for env_action in env_actions]
    record = {
        "episode": episode,
        "step": env.steps,
        "UAV_pos": env.UAV_coord.tolist(),
        "device_pos": env.device_coord.tolist(),
        "device_AOI": env.AOI.tolist(),
        "source_AOI": env.source_AOI.tolist(),
        "UAV_action_select": action_select,
        "UAV_action_move": [env_action[1].tolist() for env_action in env_actions],
        "reward": list(env.rewards),
        "total_reward": env.total_reward,
    }
    record.update(_communication_step_record(env))
    return record


def _episode_result(method, episode, reward, mean_aoi, epsilon, losses=None):
    """建立各方法訓練結果列，保留原有欄位順序。"""
    if method == "MA-CQR":
        import numpy as np

        return {
            "episode": episode,
            "reward": reward,
            "mean_aoi": mean_aoi,
            "loss": float(np.mean(losses)),
            "updates": len(losses),
            "epsilon": epsilon,
            "eval_aoi": float("nan"),
        }
    return {
        "episode": episode,
        "reward": reward,
        "mean_aoi": mean_aoi,
        "epsilon": epsilon,
        "eval_aoi": float("nan"),
    }


def _model_checkpoint(method, agents, env):
    """建立非 HRL 方法原有的模型 checkpoint 字典。"""
    import utils

    if method == "MA-CQR":
        return {
            "format_version": 3,
            "algorithm": "online_cooperative_QR_no_CQL",
            "networks": [agent.network.state_dict() for agent in agents],
            "state_size": utils.MA_CQR_SETTING.state_size,
            "action_size": utils.MA_CQR_SETTING.action_size,
            "N": utils.MA_CQR_SETTING.N,
            "eta": utils.MA_CQR_SETTING.eta,
            "U": env.U,
            "M": env.M,
            "gamma": utils.MA_CQR_SETTING.gamma,
            "tau": utils.MA_CQR_SETTING.tau,
            "risk_measure": "lower_tail_cvar",
            "quantile_schema": "equal_mass_uniform_atoms_v1",
            "observation_schema": "global_raw_coords_aoi_v2",
            "state_layout": ["UAV_coord", "device_coord", "AOI"],
            "env_setting": vars(utils.ENV_SETTING),
            "qr_setting": vars(utils.MA_CQR_SETTING),
        }
    if method == "DQN":
        state_size = utils.DQN_SETTING.state_size
        action_size = utils.DQN_SETTING.action_size
    else:
        state_size = utils.DQN_SERVICE_SETTING.state_size
        action_size = utils.DQN_SERVICE_SETTING.action_size
    return {
        "networks": [agent.network.state_dict() for agent in agents],
        "state_size": state_size,
        "action_size": action_size,
        "U": env.U,
        "M": env.M,
        "state_layout": ["UAV_coord", "device_coord", "AOI"],
    }


class FileIO:
    """持有各方法的結果、trace 與 checkpoint 狀態。"""

    def __init__(self, method, trainer):
        _require_method(method)
        self.method = method
        self.trainer = trainer
        self.results = []
        self.episode_records = []
        self.step_records = []
        self.model = None
        self.method_config = None
        self.save_started_at = None

    def episode_start(self):
        trainer = self.trainer
        if self.method in HRL_METHODS:
            # 以回合開始快照計算本回合更新差值，避免輸出累積總數。
            self.upper_updates_at_episode_start = trainer.total_upper_updates
            self.lower_updates_at_episode_start = trainer.total_lower_updates
            self.upper_update_counts_at_episode_start = list(
                trainer.upper_update_counts
            )
            record = {
                "episode": trainer.episode,
                "phase": trainer.phase,
                "init_UAV_pos": trainer.env.UAV_coord.tolist(),
                "init_device_pos": trainer.env.device_coord.tolist(),
                "init_device_AOI": trainer.env.AOI.tolist(),
                "init_source_AOI": trainer.env.source_AOI.tolist(),
                "risk_region": trainer.env.risk_region.tolist(),
                "map_size": trainer.env.map_size,
            }
            record.update(_communication_episode_record(trainer.env))
        else:
            record = _episode_record(trainer.episode, trainer.env)
        self.episode_records.append(record)

    def step(self):
        trainer = self.trainer
        if self.method in HRL_METHODS:
            import numpy as np

            record = {
                "episode": trainer.episode,
                "phase": trainer.phase,
                "step": trainer.env.steps,
                "UAV_pos": trainer.env.UAV_coord.tolist(),
                "device_pos": trainer.env.device_coord.tolist(),
                "device_AOI": trainer.env.AOI.tolist(),
                "source_AOI": trainer.env.source_AOI.tolist(),
                "UAV_target": trainer.target_actions,
                "UAV_action_select": [
                    int(env_action[0]) for env_action in trainer.env_actions
                ],
                "UAV_action_move": [
                    env_action[1].tolist() for env_action in trainer.env_actions
                ],
                "upper_reward": trainer.upper_step_rewards.tolist(),
                "mean_aoi": float(np.mean(trainer.env.AOI)),
                "reward": list(trainer.env.rewards),
                "total_reward": trainer.env.total_reward,
                "intrinsic_reward": trainer.lower_step_rewards,
                "option_completed": trainer.option_ends,
            }
            record["UAV_current_position"] = [
                bool(current_position)
                for current_position in trainer.current_position_actions
            ]
            record.update(_communication_step_record(trainer.env))
        elif self.method == "DQN_service":
            record = _step_record(
                self.method, trainer.episode, trainer.env, trainer.env_actions,
                trainer.select_actions, target_actions=trainer.target_actions,
            )
        else:
            record = _step_record(
                self.method, trainer.episode, trainer.env, trainer.env_actions,
            )
        self.step_records.append(record)

    def episode(self):
        trainer = self.trainer
        if self.method in HRL_METHODS:
            import utils

            upper_setting = utils.HRL_UPPER_SETTING

            result = {
                "episode": trainer.episode,
                "phase": trainer.phase,
                "reward": trainer.environment_return,
                "learning_reward": trainer.upper_episode_reward,
                "mean_aoi": trainer.aoi_sum / trainer.episode_steps,
                "upper_updates": (
                    trainer.total_upper_updates - self.upper_updates_at_episode_start
                ),
                "lower_updates": (
                    trainer.total_lower_updates - self.lower_updates_at_episode_start
                ),
                "eval_aoi": float("nan"),
                "upper_decisions": int(trainer.upper_decision_counts.sum()),
                "reach_rate": float("nan"),
                "sampled_reach_rate": float("nan"),
            }
            if trainer.evaluation_metrics is not None:
                result.update(trainer.evaluation_metrics)
                result["eval_return"] = trainer.eval_environment_return
                result["eval_aoi"] = trainer.eval_mean_aoi
                result["evaluation_episode"] = trainer.episode
                result["evaluation_phase"] = trainer.phase
            if trainer.navigation_metrics:
                result["reach_rate"] = trainer.navigation_metrics["deterministic"]
                result["sampled_reach_rate"] = trainer.navigation_metrics["sampled"]
            upper_algorithm = "device_scoring_q"
            result.update({
                "upper_algorithm": upper_algorithm,
                "lower_algorithm": utils.HRL_LOWER_SETTING.algorithm,
                "nominal_upper_episodes": trainer.upper_min_episodes,
                "nominal_lower_episodes": trainer.lower_min_episodes,
                "actual_upper_episodes": trainer.phase_counts["upper"],
                "actual_lower_episodes": trainer.phase_counts["lower"],
                "actual_joint_episodes": trainer.phase_counts["joint"],
                "upper_converged": trainer.upper_converged,
                "lower_converged": trainer.lower_converged,
                "joint_started": trainer.phase_counts["joint"] > 0,
            })
            result["upper_updates_per_agent"] = [
                int(current - previous)
                for current, previous in zip(
                    trainer.upper_update_counts,
                    self.upper_update_counts_at_episode_start,
                )
            ]
            if upper_setting.reward_mode == "average_reward_smdp":
                result["upper_learning_rule"] = "differential_q"
                result["upper_average_reward_rates"] = [
                    agent.average_reward_rate for agent in trainer.upper_agents
                ]
        else:
            if self.method == "MA-CQR":
                losses = trainer.losses
            else:
                losses = None
            result = _episode_result(
                self.method,
                trainer.episode,
                trainer.episode_reward,
                trainer.aoi_sum / trainer.episode_steps,
                trainer.epsilon,
                losses,
            )
            if trainer.evaluated:
                result.update(trainer.eval_env.last_eval_metrics)
        now = time.perf_counter()
        episode_elapsed = now - trainer.episode_started_at
        if self.method in HRL_METHODS:
            episode_elapsed += trainer.pending_initialization_elapsed_seconds
        result["episode_wall_time_seconds"] = episode_elapsed
        result["wall_time_seconds"] = now - trainer.training_started_at
        self.results.append(result)

    def prepare(self):
        trainer = self.trainer
        if self.method in HRL_METHODS:
            import numpy as np
            import utils

            lower_setting = utils.HRL_LOWER_SETTING
            upper_size_setting = utils.HRL_UPPER_SETTING
            expected_network_class = "SharedDeviceQNetwork"
            checkpoint_schema = "hrl_independent_device_uav_context_q_v2"
            observation_schema = "normalized_focal_coords_aoi_v1"
            normalization_schema = "normalized_focal_coords_aoi_v1"
            uav_count = trainer.env.U
            upper_agents = list(trainer.upper_agents)
            lower_agents = list(trainer.lower_agents)
            if len(upper_agents) != uav_count:
                raise ValueError("HRL checkpoints require one upper agent per UAV")
            if len({id(agent) for agent in upper_agents}) != uav_count:
                raise ValueError("HRL upper agents must be independent")
            if len({id(agent.network) for agent in upper_agents}) != uav_count:
                raise ValueError("HRL upper networks must be independent")
            if len({id(agent.target_net) for agent in upper_agents}) != uav_count:
                raise ValueError("HRL upper target networks must be independent")
            if any(type(agent.network).__name__ != expected_network_class
                   for agent in upper_agents):
                raise ValueError(
                    f"{self.method} checkpoints require {expected_network_class} upper networks"
                )
            if len(lower_agents) != uav_count:
                raise ValueError("HRL checkpoints require one lower agent per UAV")
            if len({id(agent) for agent in lower_agents}) != uav_count:
                raise ValueError("HRL lower agents must be independent")
            if len({id(agent.actor) for agent in lower_agents}) != uav_count:
                raise ValueError("HRL lower actors must be independent")
            upper_update_counts = list(trainer.upper_update_counts)
            if (len(upper_update_counts) != uav_count
                    or any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, np.integer))
                           or value < 0 for value in upper_update_counts)):
                raise ValueError("HRL upper_update_counts must contain one non-negative integer per UAV")
            upper_update_counts = [int(value) for value in upper_update_counts]
            total_upper_updates = trainer.total_upper_updates
            if (isinstance(total_upper_updates, (bool, np.bool_))
                    or not isinstance(total_upper_updates, (int, np.integer))
                    or int(total_upper_updates) != sum(upper_update_counts)):
                raise ValueError("HRL total_upper_updates must equal the sum of per-UAV updates")

            lower_update_counts = list(trainer.lower_update_counts)
            if (len(lower_update_counts) != uav_count
                    or any(isinstance(value, (bool, np.bool_))
                           or not isinstance(value, (int, np.integer))
                           or value < 0 for value in lower_update_counts)):
                raise ValueError("HRL lower_update_counts must contain one non-negative integer per UAV")
            lower_update_counts = [int(value) for value in lower_update_counts]

            phase_summary = {
                "nominal_upper_episodes": trainer.upper_min_episodes,
                "nominal_lower_episodes": trainer.lower_min_episodes,
                "actual_upper_episodes": trainer.phase_counts["upper"],
                "actual_lower_episodes": trainer.phase_counts["lower"],
                "actual_joint_episodes": trainer.phase_counts["joint"],
                "upper_update_counts": upper_update_counts,
                "total_upper_updates": int(total_upper_updates),
                "lower_updates_per_agent": lower_update_counts,
                "total_lower_updates": trainer.total_lower_updates,
                "upper_converged": trainer.upper_converged,
                "lower_converged": trainer.lower_converged,
                "joint_started": trainer.phase_counts["joint"] > 0,
                "budget_exhausted_before_joint": trainer.phase_counts["joint"] == 0,
            }
            for result in self.results:
                result["budget_exhausted_before_joint"] = phase_summary[
                    "budget_exhausted_before_joint"
                ]
                result["final_upper_converged"] = trainer.upper_converged
                result["final_lower_converged"] = trainer.lower_converged

            lower_checkpoints = []
            for agent_index, lower_agent in enumerate(lower_agents):
                lower_checkpoint = {
                    "agent_index": agent_index,
                    "actor": lower_agent.actor.state_dict(),
                }
                if utils.HRL_LOWER_SETTING.algorithm in ("dsac", "dsac_t"):
                    lower_checkpoint["target_actor"] = lower_agent.target_actor.state_dict()
                if utils.HRL_LOWER_SETTING.algorithm in ("a2c", "dsac"):
                    lower_checkpoint["critic"] = lower_agent.critic.state_dict()
                else:
                    lower_checkpoint["critic_1"] = lower_agent.critic_1.state_dict()
                    lower_checkpoint["critic_2"] = lower_agent.critic_2.state_dict()
                if utils.HRL_LOWER_SETTING.algorithm == "dsac":
                    lower_checkpoint["target_critic"] = lower_agent.target_critic.state_dict()
                elif utils.HRL_LOWER_SETTING.algorithm in ("sac", "dsac_t"):
                    lower_checkpoint["target_critic_1"] = lower_agent.target_critic_1.state_dict()
                    lower_checkpoint["target_critic_2"] = lower_agent.target_critic_2.state_dict()
                if utils.HRL_LOWER_SETTING.algorithm == "dsac_t":
                    lower_checkpoint["mean_std1"] = lower_agent.mean_std1
                    lower_checkpoint["mean_std2"] = lower_agent.mean_std2
                lower_checkpoint["update_count"] = lower_agent.update_count
                lower_checkpoints.append(lower_checkpoint)

            upper_checkpoints = []
            for agent_index, agent in enumerate(upper_agents):
                upper_checkpoint = {
                    "agent_index": agent_index,
                    "network": agent.network.state_dict(),
                    "target_net": agent.target_net.state_dict(),
                }
                if utils.HRL_UPPER_SETTING.reward_mode == "average_reward_smdp":
                    upper_checkpoint["rho"] = agent.average_reward_rate
                upper_checkpoints.append(upper_checkpoint)

            if utils.HRL_UPPER_SETTING.reward_mode == "average_reward_smdp":
                upper_reward_definition = "option_team_environment_sum_differential_smdp_v1"
            elif utils.HRL_UPPER_SETTING.reward_mode == "option_mean_environment":
                upper_reward_definition = "option_mean_team_environment_reward_v1"
            else:
                upper_reward_definition = "option_aoi_risk_current_position_zero_v1"

            if upper_size_setting.action_size != trainer.env.M + 1:
                raise ValueError(
                    f"{self.method} 上層 action_size 必須是 M 個裝置動作加上 UAV 目前位置動作"
                )
            effective_hrl_setting = dict(vars(utils.HRL_SETTING))
            effective_upper_setting = dict(vars(utils.HRL_UPPER_SETTING))
            effective_hrl_setting.update(
                state_size=upper_size_setting.state_size,
                action_size=upper_size_setting.action_size,
            )
            effective_upper_setting.update(
                state_size=upper_size_setting.state_size,
                action_size=upper_size_setting.action_size,
            )
            state_layout = [
                "focal_UAV_xy", "all_UAV_xy", "all_device_xy", "AOI",
            ]
            self.model = {
                "format_version": 1,
                "checkpoint_schema": checkpoint_schema,
                "method": self.method,
                "upper_mode": "independent",
                "upper_algorithm": "device_scoring_q",
                "upper_reward_definition": upper_reward_definition,
                "upper_reward_mode": utils.HRL_UPPER_SETTING.reward_mode,
                "upper_risk_penalty_weight": utils.HRL_UPPER_SETTING.risk_penalty_weight,
                "upper": upper_checkpoints,
                "upper_update_counts": upper_update_counts,
                "total_upper_updates": int(total_upper_updates),
                "lower_mode": "independent",
                "lower_algorithm": utils.HRL_LOWER_SETTING.algorithm,
                "lower_agents": lower_checkpoints,
                "lower_update_counts": lower_update_counts,
                "total_lower_updates": trainer.total_lower_updates,
                "lower_setting": vars(lower_setting),
                "state_size": upper_size_setting.state_size,
                "action_size": upper_size_setting.action_size,
                "lower_state_size": lower_setting.state_size,
                "U": uav_count,
                "M": trainer.env.M,
                "current_position_action": trainer.env.M,
                "network_architecture": expected_network_class,
                "observation_schema": observation_schema,
                "normalization_schema": normalization_schema,
                "coordinate_scale": trainer.env.map_size,
                "age_scale": utils.TRAIN_SETTING.max_steps,
                "state_layout": state_layout,
                "lower_state_layout": ["UAV_coord_u", "target_coord", "target_delta"],
                "hrl_setting": effective_hrl_setting,
                "upper_setting": effective_upper_setting,
                "env_setting": vars(utils.ENV_SETTING),
                "training_controller": phase_summary,
            }
            self.model.update(
                candidate_context_schema="other_uav_relative_position_distance_v1",
                candidate_uav_relation_layout=[
                    "candidate_minus_uav_dx",
                    "candidate_minus_uav_dy",
                    "candidate_uav_distance",
                ],
                candidate_uav_pooling="mean_excluding_focal",
                candidate_uav_context_size=utils.HRL_UPPER_SETTING.hidden_size,
                scorer_input_size=3 * utils.HRL_UPPER_SETTING.hidden_size,
            )
            if utils.HRL_UPPER_SETTING.reward_mode == "average_reward_smdp":
                self.model["upper_learning_rule"] = "differential_q"
                self.model["upper_average_reward_lr"] = (
                    utils.HRL_UPPER_SETTING.average_reward_lr
                )
            self.method_config = {
                "HRL_SETTING": {
                    **effective_hrl_setting,
                    "upper_episodes": trainer.upper_min_episodes,
                    "lower_episodes": trainer.lower_min_episodes,
                    **phase_summary,
                },
                "HRL_UPPER_SETTING": effective_upper_setting,
                "HRL_LOWER_SETTING": dict(vars(lower_setting)),
            }
        else:
            self.model = _model_checkpoint(self.method, trainer.agents, trainer.env)
        self.save_started_at = time.perf_counter()

    def save(self):
        save_results(
            self.method,
            self.results,
            self.episode_records,
            self.step_records,
            self.model,
            method_config=self.method_config,
            training_started_at=self.trainer.training_started_at,
            save_started_at=self.save_started_at,
        )


def _new_result_figure():
    """結果圖使用獨立 Agg canvas，不更改其他繪圖後端。"""
    import utils
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=utils.RECORD_SETTING.figure_size)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


KEYS = ["method", "iteration", "iteration_unit"]
META = ["eval_episode_count", "eval_max_steps"]


def read_metric(path, column):
    """讀取指定評估指標，檢查必要欄位、索引與數值。"""
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    required = KEYS + META + [column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少欄位：{sorted(missing)}")
    frame = frame[required].copy()
    if frame[required].isna().any().any():
        raise ValueError(f"{path} 包含空值")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} 有重複的評估索引")
    for name in ["iteration"] + META + [column]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"{path} 的 {name} 包含非有限數值")
    for name in META:
        if ((frame[name] <= 0) | (frame[name] % 1 != 0)).any():
            raise ValueError(f"{path} 的 {name} 必須是正整數")
    if (frame[column] < 0).any():
        raise ValueError(f"{path} 的 {column} 不可為負值")
    return frame


def merge_metric(frame, other):
    """對齊評估索引並核對步數、回合數後合併指標。"""
    merged = frame.merge(other, on=KEYS, how="outer", validate="one_to_one",
                         suffixes=("", "_other"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("各 CSV 的 method、iteration、iteration_unit 必須完全對應")
    for name in META:
        if not merged[name].eq(merged[name + "_other"]).all():
            raise ValueError(f"各 CSV 的 {name} 不一致")
    return merged.drop(columns=["_merge"] + [name + "_other" for name in META])


def combine_rewards(aoi, power, violations):
    """由 AoI、加權功耗與風險懲罰重建評估回報及平滑值。"""
    import utils

    frame = merge_metric(read_metric(aoi, "eval_aoi"),
                         read_metric(power, "eval_weighted_power"))
    frame = merge_metric(frame, read_metric(violations, "eval_violation_penalty"))
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    # 使用評估當時記錄的加權分項，不從目前設定推測歷史權重。
    frame["eval_reward"] = (-frame.eval_aoi * frame.eval_max_steps
                            - frame.eval_weighted_power - frame.eval_violation_penalty)
    frame["risk_included"] = True
    frame["smoothing_window"] = utils.RECORD_SETTING.reward_smoothing_window
    # rolling(window).mean() 會保留不足視窗的前幾筆空值。
    frame["eval_reward_smoothed"] = frame.groupby(
        ["method", "iteration_unit"], sort=False)["eval_reward"].transform(
            lambda values: values.rolling(
                window=utils.RECORD_SETTING.reward_smoothing_window
            ).mean())
    return frame


def plot_eval_reward(csv_path, png_path):
    """只讀 CSV 重繪評估回報；沒有資料時仍輸出空圖。"""
    import pandas as pd
    import utils

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    # 圖片從落盤 CSV 重讀，讓輸出與繪圖來源一致。
    plotted = pd.read_csv(csv_path)
    fig, ax = _new_result_figure()
    for (method, unit), group in plotted.groupby(["method", "iteration_unit"], sort=False):
        ax.plot(group.iteration, group.eval_reward_smoothed, label=f"{method} ({unit})")
    if not plotted.empty:
        window = int(plotted.smoothing_window.iloc[0])
    else:
        window = utils.RECORD_SETTING.reward_smoothing_window
    if plotted.risk_included.all():
        risk = "including risk penalty"
    else:
        risk = "excluding risk penalty"
    ax.set(xlabel="Training iteration", ylabel="Evaluation cumulative reward",
           title=f"Reconstructed eval reward ({risk}, rolling window={window})")
    if plotted.empty:
        ax.text(.5, .5, "No data", transform=ax.transAxes, ha="center")
    elif plotted.eval_reward_smoothed.notna().any():
        ax.legend()
    else:
        ax.text(.5, .5, "Not enough evaluations for the configured smoothing window",
                transform=ax.transAxes, ha="center")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.clear()


METRIC_COLUMNS = ["method", "iteration", "iteration_unit", "episode", "epoch", "phase",
                  "reward", "learning_reward", "mean_aoi", "eval_aoi",
                  "eval_max_steps", "eval_cumulative_power", "eval_violation_count", "eval_episode_count",
                  "eval_weighted_power", "eval_violation_penalty", "eval_total_violation_penalty",
                  "loss", "qr_loss", "cql_loss", "updates", "upper_updates", "lower_updates",
                  "epsilon", "upper_decisions", "reach_rate", "sampled_reach_rate",
                  "episode_wall_time_seconds", "wall_time_seconds"]
EPISODE_COLUMNS = ["method", "trace_kind", "episode", "phase", "init_UAV_pos",
                   "init_device_pos", "init_device_AOI", "init_source_AOI",
                   "risk_region", "map_size"]
STEP_COLUMNS = ["method", "trace_kind", "episode", "phase", "step", "UAV_pos", "device_pos",
                "device_AOI", "source_AOI", "UAV_action_select", "UAV_action_move",
                "reward", "total_reward"]

TRACE_EXTENSION_DESCRIPTIONS = {
    "episode": {
        "communication_model": {
            "description": "環境使用的通訊模型",
        },
    },
    "step": {
        "communication_scheme": {
            "description": "本環境步實際採用的通訊配置",
        },
        "UAV_service_success": {
            "description": "各 UAV 本步是否成功傳送所選裝置的封包",
        },
        "UAV_transmission_power": {
            "description": "各 UAV 在 1 秒環境時槽內的平均傳輸功率",
            "unit": "W",
        },
        "UAV_transmission_peak_power": {
            "description": "各 UAV 實際傳輸期間的峰值功率",
            "unit": "W",
        },
        "UAV_transmission_time": {
            "description": "各 UAV 本環境時槽內的實際傳輸時間",
            "unit": "seconds",
        },
    },
}

EVAL_COMPONENTS = {
    "eval_aoi": ("AoI", "Mean AoI (devices, steps and evaluation episodes)"),
    "eval_weighted_power": ("Power", "Weighted transmission cost / UAV\n(mean over evaluation episodes)"),
    # 舊 CSV 可能保存受罰次數，不把未知歷史定義直接改標為風險區步數。
    "eval_violation_count": ("Violations", "Recorded risk count, summed over UAVs\n(mean over evaluation episodes)"),
}
PLOT_COMPONENTS = {
    "reward": ("Reward", "Training environment return"),
    "wall_time_seconds": ("Wall_Time", "Cumulative training wall time (seconds)"),
    **EVAL_COMPONENTS,
}


def _metric_directories(root, method):
    # 歷史方法只允許讀取既有結果，不開放建立訓練輸出。
    if method is not None and method not in HISTORICAL_RESULT_METHODS:
        _require_method(method)
    root = Path(root)
    if not root.is_dir():
        return []
    if method is not None:
        directories = [root / method / "Result_Datas"]
    else:
        directories = sorted(
            path / "Result_Datas" for path in root.iterdir()
            if path.is_dir() and (path / "Result_Datas").is_dir())
    return directories


def read_plot_paths(root, method=None):
    """讀取可重繪的指標清單，依繪圖順序提供對應檔案路徑。"""
    outputs = []
    for metric_dir in _metric_directories(root, method):
        if not metric_dir.is_dir():
            continue
        for suffix, _ in PLOT_COMPONENTS.values():
            if (metric_dir / f"{suffix}.csv").is_file():
                outputs.append(metric_dir.parent / "Results" / f"{suffix}.png")
        if (metric_dir / "eval_reward.csv").is_file():
            outputs.append(metric_dir.parent / "Results" / "eval_reward.png")
    return outputs


def plot_results(root, method=None):
    """從分項 CSV 重建圖表，不回傳輸出資料。"""
    import pandas as pd

    directories = _metric_directories(root, method)
    for metric_dir in directories:
        if not metric_dir.is_dir():
            continue
        for column, (suffix, label) in PLOT_COMPONENTS.items():
            csv_path = metric_dir / f"{suffix}.csv"
            if not csv_path.is_file():
                continue
            component = pd.read_csv(csv_path)
            plot_column, plot_label = column, label
            # 舊版 Power CSV 保留未加權語意，避免將歷史值誤標為加權功耗。
            if suffix == "Power" and column not in component:
                plot_column = "eval_cumulative_power"
                plot_label = "Cumulative transmission power (unweighted, sum over UAVs)"
            fig_dir = metric_dir.parent / "Results"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = _new_result_figure()
            if component.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                unit = "Iteration"
            else:
                ax.plot(component.iteration, component[plot_column], label=plot_label)
                ax.legend()
                unit = str(component.iteration_unit.iloc[0]).capitalize()
            ax.set(xlabel=unit, ylabel=plot_label, title=f"{metric_dir.parent.name} {suffix}")
            ax.grid(True)
            output = fig_dir / f"{suffix}.png"
            fig.savefig(output)
            fig.clear()
        eval_csv = metric_dir / "eval_reward.csv"
        if eval_csv.is_file():
            output = metric_dir.parent / "Results" / "eval_reward.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            plot_eval_reward(eval_csv, output)


def save_trace(trace_dir, episode_frame, step_frame, extensions=None, *,
               core_columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    extensions = dict(extensions or {})
    core_columns = core_columns or {}
    manifest = {
        "trace_schema_version": 2,
        "default_renderer": "uav_2d",
        "aoi_semantics": {
            "device_AOI": "Collector/UAV global age used by rewards and costs.",
            "init_device_AOI": "Initial collector/UAV global age.",
            "source_AOI": "Latest locally generated packet age at each device.",
            "init_source_AOI": "Initial latest locally generated packet age.",
        },
        "tables": {},
    }
    for name, frame, default_core in [("episode", episode_frame, EPISODE_COLUMNS),
                                      ("step", step_frame, STEP_COLUMNS)]:
        core = core_columns.get(name, default_core)
        descriptions = dict(extensions.get(name, {}))
        for column, description in TRACE_EXTENSION_DESCRIPTIONS[name].items():
            if column in frame:
                descriptions.setdefault(column, description)
        unknown = set(descriptions) - set(frame.columns)
        if unknown:
            raise ValueError(f"擴充說明引用不存在的欄位：{unknown}")
        manifest["tables"][name] = {
            "required_columns": list(core),
            "extensions": {
                column: {"dtype": str(frame[column].dtype), **descriptions.get(column, {})}
                for column in frame if column not in core
            },
        }
    for name, frame in [("episode", episode_frame), ("step", step_frame)]:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"uav_trace_schema"] = json.dumps(
            manifest, ensure_ascii=False
        ).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), trace_dir / f"{name}.parquet")
    (trace_dir / "trace_schema.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ordered(frame, columns):
    """共同欄位排在前方，各方法 trace 擴充欄位接續其後。"""
    import numpy as np

    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns + [column for column in frame if column not in columns]]


def save_results(method, records, episodes, steps, model, iteration_unit="episode",
                 trace_kind="train", trace_extensions=None, method_config=None,
                 training_started_at=None, *, save_started_at=None):
    """保存結果 CSV、trace、圖表與 checkpoint。"""
    _require_method(method)
    if training_started_at is not None:
        if save_started_at is None:
            save_started_at = time.perf_counter()
        training_wall_time_seconds = save_started_at - training_started_at
    else:
        training_wall_time_seconds = None
    import torch
    import pandas as pd
    import tempfile
    import utils

    root = Path(utils.RECORD_SETTING.PATH)
    method_root = root / method
    metric_dir = method_root / "Result_Datas"
    trace_dir = method_root / "Stored_Datas"
    frame = pd.DataFrame(records).copy()
    frame["method"] = method
    frame["iteration_unit"] = iteration_unit
    frame["iteration"] = frame[iteration_unit]
    # 每個評估回合另表保存，主表只保存跨回合平均。
    eval_rows = []
    if "eval_episodes" in frame:
        for _, row in frame.iterrows():
            if isinstance(row.eval_episodes, list):
                for evaluated_episode in row.eval_episodes:
                    eval_rows.append({"method": method, "iteration": row.iteration,
                                      "iteration_unit": iteration_unit, **evaluated_episode})
        frame = frame.drop(columns="eval_episodes")
    if "phase" not in frame:
        if iteration_unit == "epoch":
            frame["phase"] = "offline"
        else:
            frame["phase"] = "train"
    # HRL 學習回報與其他方法的環境回報分欄保存。
    if "environment_return" in frame:
        frame["learning_reward"] = frame["reward"]
        frame["reward"] = frame.pop("environment_return")
    frame = ordered(frame, METRIC_COLUMNS)
    episode_frame, step_frame = pd.DataFrame(episodes).copy(), pd.DataFrame(steps).copy()
    for table in (episode_frame, step_frame):
        table["method"] = method
        table["trace_kind"] = trace_kind
        if "phase" not in table:
            table["phase"] = trace_kind
    if trace_kind == "train" and "episode" in episode_frame:
        for column in ("episode_wall_time_seconds", "wall_time_seconds"):
            if column in frame and frame[column].notna().any():
                time_by_episode = frame.dropna(subset=["episode"]).drop_duplicates(
                    "episode"
                ).set_index("episode")[column]
                mapped_times = episode_frame["episode"].map(time_by_episode)
                if column in episode_frame:
                    episode_frame[column] = episode_frame[column].fillna(mapped_times)
                else:
                    episode_frame[column] = mapped_times
    if training_wall_time_seconds is not None:
        episode_frame["training_wall_time_seconds"] = training_wall_time_seconds
    uses_service_range = method in ("DQN_service", *HRL_METHODS, "HRL_PPO")
    if uses_service_range:
        # 保存訓練時實際服務距離，回放不從目前設定猜測歷史範圍。
        if "map_size" not in episode_frame:
            episode_frame["map_size"] = utils.ENV_SETTING.map_size
        else:
            episode_frame["map_size"] = episode_frame["map_size"].fillna(
                utils.ENV_SETTING.map_size
            )
        if method == "HRL_PPO":
            radius = episode_frame["map_size"] * utils.HRL_PPO_SETTING.service_ratio
            if model.get("checkpoint_schema") == "hrl_ppo_ctde_v2":
                rule = "within_range"
            else:
                rule = "within_range_inclusive"
        elif method in HRL_METHODS:
            radius = episode_frame["map_size"] * utils.HRL_SETTING.service_ratio
            rule = "within_range"
        else:
            radius = episode_frame["map_size"] * utils.DQN_SERVICE_SETTING.service_ratio
            rule = "force_within_range_allow_early"
        if "service_radius" not in episode_frame:
            episode_frame["service_radius"] = radius
        else:
            episode_frame["service_radius"] = episode_frame["service_radius"].fillna(radius)
        if "service_rule" not in episode_frame:
            episode_frame["service_rule"] = rule
        else:
            episode_frame["service_rule"] = episode_frame["service_rule"].fillna(rule)
    time_descriptions = {
        "episode_wall_time_seconds": {
            "description": "本訓練回合從開始到訓練、評估與導航檢查完成的經過時間",
            "unit": "seconds",
        },
        "wall_time_seconds": {
            "description": "從 train() 進入到本訓練回合完成的累積經過時間",
            "unit": "seconds",
        },
        "training_wall_time_seconds": {
            "description": "從 train() 進入到結果保存開始的經過時間",
            "unit": "seconds",
        },
    }
    if uses_service_range or any(column in episode_frame for column in time_descriptions):
        trace_extensions = dict(trace_extensions or {})
        descriptions = dict(trace_extensions.get("episode", {}))
        if uses_service_range:
            descriptions.setdefault("service_radius", {
                "description": "裝置服務範圍的半徑（本回合 map_size × service_ratio）",
                "renderer": "device_service_range", "unit": "map_coordinate",
            })
            descriptions.setdefault("service_rule", {
                "description": "within_range：移動後距離 < 半徑才服務；within_range_inclusive：移動後距離 ≤ 半徑才服務；force_within_range_allow_early：範圍內強制服務，範圍外可提前服務",
            })
        for column, description in time_descriptions.items():
            if column in episode_frame:
                descriptions.setdefault(column, description)
        trace_extensions["episode"] = descriptions
    upper_reward_description = "各 UAV 保存的上層學習回饋，依訓練時的獎勵模式解讀"
    if method == "HRL_PPO":
        upper_setting = utils.HRL_PPO_UPPER_SETTING
    elif method in HRL_METHODS:
        upper_setting = utils.HRL_UPPER_SETTING
    else:
        upper_setting = None
    if method in (*HRL_METHODS, "HRL_PPO") and upper_setting.reward_mode == "average_reward_smdp":
        upper_reward_description = (
            "各 UAV 獨立上層在目標 Option 或目前位置一步完成時結算團隊環境 reward 的未折扣總和；"
            "完整區間包含 AoI、加權傳輸成本及實際原額風險懲罰，中途為 0。"
            "學習時另以 R−rho×duration 加上後續價值，目前位置動作只涵蓋一個原始環境步。"
        )
        if method == "HRL_PPO":
            upper_reward_description += (
                "回合上限截斷時保存目前區間的未折扣總和，並以未完成片段對同一目標 bootstrap。"
            )
    elif method in (*HRL_METHODS, "HRL_PPO") and upper_setting.reward_mode == "option_mean_environment":
        upper_reward_description = (
            "各 UAV 獨立上層在目標 Option 或目前位置一步完成時結算團隊環境 reward 的每步算術平均；"
            "完整區間包含 AoI、加權傳輸成本及實際原額風險懲罰，中途為 0。"
            "目前位置動作只涵蓋一個原始環境步。"
        )
        if method == "HRL_PPO":
            upper_reward_description += (
                "回合上限截斷時以目前區間的每步算術平均保存，並以未完成片段對同一目標 bootstrap。"
            )
    elif method in (*HRL_METHODS, "HRL_PPO"):
        upper_reward_description = (
            "本步各 UAV 的獨立上層 Option 獎勵；目前位置動作固定為 0，服務目標 Option 依 AoI 降低與風險事件計算。"
            "同步服務時均分降低量，"
            f"再扣該 UAV 實際受罰次數乘以係數 {upper_setting.risk_penalty_weight:g}；"
            "不含罰款金額或傳輸成本。"
        )
    step_descriptions = {
        "UAV_target": {
            "description": "各 UAV 本步鎖定的規劃目標，未必等於本步實際服務的裝置",
        },
        "UAV_allow_service": {
            "description": "各 UAV 本步的二值服務嘗試：0 不嘗試，1 嘗試服務，包含距離觸發的強制嘗試；不是服務權限或成功標記",
        },
        "upper_reward": {
            "description": upper_reward_description,
        },
        "intrinsic_reward": {
            "description": "本步各 UAV 的下層導航獎勵，包含接近目標、完成導航、耗時及實際風險的加權項",
        },
        "UAV_idle": {
            "description": "HRL-SA 本步上層是否選擇 Idle；True 表示停留一個原始環境步後重新選擇，不代表由服務頭拒絕服務",
        },
        "UAV_current_position": {
            "description": "HRL 本步上層是否選擇 UAV 目前位置；True 表示原地不服務一個環境步後重新選擇，AoI 與風險仍依環境前進",
        },
    }
    if any(column in step_frame for column in step_descriptions):
        trace_extensions = dict(trace_extensions or {})
        descriptions = dict(trace_extensions.get("step", {}))
        for column, description in step_descriptions.items():
            if column in step_frame:
                descriptions.setdefault(column, description)
        trace_extensions["step"] = descriptions
    episode_frame = ordered(episode_frame, EPISODE_COLUMNS)
    step_frame = ordered(step_frame, STEP_COLUMNS)
    trace_dir.mkdir(parents=True, exist_ok=True)
    save_trace(trace_dir, episode_frame, step_frame, trace_extensions)
    csv_path = metric_dir / f"Result_{method}.csv"
    metric_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    pd.DataFrame(eval_rows, columns=["method", "iteration", "iteration_unit", "eval_episode",
                                    "steps", "max_steps", "mean_aoi", "cumulative_power",
                                    "weighted_power", "violation_count", "violation_penalty",
                                    "total_violation_penalty"]).to_csv(
        metric_dir / f"Eval_Episodes_{method}.csv", index=False)
    # 指標先落盤，圖表由已保存的 CSV 重建。
    # Reward.csv 只保存訓練 phase；HRL reward 已轉為環境回報。
    non_training_phases = {
        "collect", "warmup", "offline", "pretrain", "eval", "evaluation",
    }
    # 僅正規化篩選用的階段名稱，主表保留原始 phase。
    normalized_phase = frame["phase"].fillna("train").astype(str).str.strip().str.casefold()
    normalized_phase = normalized_phase.mask(normalized_phase.eq(""), "train")
    training = frame.loc[
        frame.reward.notna() & ~normalized_phase.isin(non_training_phases),
        ["method", "iteration", "iteration_unit", "phase", "reward", "learning_reward"],
    ]
    training.to_csv(metric_dir / "Reward.csv", index=False)
    for column, (suffix, _) in EVAL_COMPONENTS.items():
        extra = {"Power": ["eval_cumulative_power"],
                 "Violations": ["eval_violation_penalty", "eval_total_violation_penalty"]}.get(
                     suffix, []
                 )
        component = frame.loc[
            frame[column].notna(),
            ["method", "iteration", "iteration_unit", "eval_episode_count", "eval_max_steps",
             column] + extra,
        ]
        component.to_csv(metric_dir / f"{suffix}.csv", index=False)
    wall_time_csv = metric_dir / "Wall_Time.csv"
    wall_time_png = method_root / "Results" / "Wall_Time.png"
    if "wall_time_seconds" in frame and frame["wall_time_seconds"].notna().any():
        wall_time = frame.loc[
            frame.wall_time_seconds.notna(),
            ["method", "iteration", "iteration_unit",
             "episode_wall_time_seconds", "wall_time_seconds"],
        ]
        wall_time.to_csv(wall_time_csv, index=False)
    else:
        if wall_time_csv.is_file():
            wall_time_csv.unlink()
        if wall_time_png.is_file():
            wall_time_png.unlink()
    # 從分項 CSV 合成評估 reward，使用既有公式與平滑視窗。
    eval_reward = combine_rewards(metric_dir / "AoI.csv", metric_dir / "Power.csv",
                                  metric_dir / "Violations.csv")
    eval_reward.to_csv(metric_dir / "eval_reward.csv", index=False)
    plot_results(root, method)
    model = dict(model)
    # 共用 weights 只收集此方法實際保存的網路，保留原生欄位。
    weights = {}
    for key in ("networks", "upper", "lower", "lower_agents", "centralized_critic", "lower_critics",
                "lower_target_critics"):
        if key in model:
            weights[key] = model[key]
    metadata = model.get("dataset_metadata", model.get("metadata", {}))
    model_config = {}
    for key in ("state_size", "action_size", "lower_state_size", "N", "eta", "U", "M",
                "idle_action", "current_position_action", "coordinate_scale", "age_scale", "normalization_schema"):
        # 實際模型尺寸優先於 dataset metadata，避免快照保存舊尺寸。
        if key in model:
            model_config[key] = model[key]
        elif key in metadata:
            model_config[key] = metadata[key]
    if method == "HRL":
        for key in (
                "checkpoint_schema", "network_architecture", "observation_schema",
                "state_layout", "candidate_context_schema",
                "candidate_uav_relation_layout", "candidate_uav_pooling",
                "candidate_uav_context_size", "scorer_input_size"):
            if key in model:
                model_config[key] = model[key]
    if method == "DQN":
        effective_method_config = {
            "DQN_SETTING": dict(vars(utils.DQN_SETTING)),
        }
    elif method == "DQN_service":
        effective_method_config = {
            "DQN_SERVICE_SETTING": dict(vars(utils.DQN_SERVICE_SETTING)),
        }
    elif method == "HRL_PPO":
        effective_method_config = {
            "HRL_PPO_SETTING": dict(vars(utils.HRL_PPO_SETTING)),
            "HRL_PPO_UPPER_SETTING": dict(vars(utils.HRL_PPO_UPPER_SETTING)),
            "HRL_PPO_LOWER_SETTING": dict(vars(utils.HRL_PPO_LOWER_SETTING)),
        }
    elif method in HRL_METHODS:
        effective_hrl_setting = dict(vars(utils.HRL_SETTING))
        effective_upper_setting = dict(vars(utils.HRL_UPPER_SETTING))
        effective_method_config = {
            "HRL_SETTING": effective_hrl_setting,
            "HRL_UPPER_SETTING": effective_upper_setting,
            "HRL_LOWER_SETTING": dict(vars(utils.HRL_LOWER_SETTING)),
        }
    else:
        effective_method_config = {
            "MA_CQR_SETTING": dict(vars(utils.MA_CQR_SETTING)),
        }
    for name, overrides in (method_config or {}).items():
        effective_method_config[name] = {
            **effective_method_config.get(name, {}), **overrides
        }
    # CLI 覆寫只保存於本次快照，不回寫全域設定。
    model.update(output_schema_version=5, method=method, weights=weights,
                 model_config=model_config,
                 env_setting=dict(vars(utils.ENV_SETTING)),
                 training_config=dict(vars(utils.TRAIN_SETTING)),
                 method_config=effective_method_config,
                 evaluation_config=dict(vars(utils.EVAL_SETTING)),
                 functional_setting=dict(vars(utils.FUNCTIONAL_SETTING)))
    if training_wall_time_seconds is not None:
        model["training_wall_time_seconds"] = training_wall_time_seconds
    checkpoint_path = trace_dir / "model.pt"
    temporary_checkpoint = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=trace_dir, prefix=".model.pt-", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_checkpoint = Path(temporary_file.name)
        # 先關閉暫存檔再交給 Torch，避免 Windows 的檔案分享限制。
        torch.save(model, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
    finally:
        if temporary_checkpoint is not None and temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
    if training_started_at is not None:
        runtime = {
            "schema_version": 1,
            "method": method,
            "training_wall_time_seconds": training_wall_time_seconds,
            "wall_time_seconds": time.perf_counter() - training_started_at,
            "training_scope": "train entry through initialization, warmup, training, evaluation; before result serialization",
            "wall_time_scope": "train entry through result and checkpoint saving",
            "unit": "seconds",
        }
        write_json(trace_dir / "runtime.json", runtime, atomic=True)
    else:
        runtime_path = trace_dir / "runtime.json"
        if runtime_path.is_file():
            runtime_path.unlink()


def write_text(path, text, *, encoding="utf-8"):
    Path(path).write_text(text, encoding=encoding)


def read_text(path, encoding="utf-8"):
    return Path(path).read_text(encoding=encoding)


def make_directory(path, *, parents=False, exist_ok=False):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def read_log_tail(path, *, max_bytes, max_lines):
    """限制讀取量與行數；省略首個殘行並保留原本換行。"""
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        byte_truncated = file_size > max_bytes
        if byte_truncated:
            start_offset = file_size - max_bytes
            handle.seek(start_offset - 1)
            previous_byte = handle.read(1)
            handle.seek(start_offset)
        else:
            previous_byte = b""
            handle.seek(0)
        raw_bytes = handle.read(max_bytes)
    if byte_truncated and previous_byte != b"\n":
        first_newline = raw_bytes.find(b"\n")
        if first_newline >= 0 and first_newline < len(raw_bytes) - 1:
            raw_bytes = raw_bytes[first_newline + 1:]
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    lines = text.splitlines(keepends=True)
    line_truncated = len(lines) > max_lines
    if line_truncated:
        lines = lines[-max_lines:]
    return "".join(lines), byte_truncated or line_truncated


def read_json(path, encoding="utf-8"):
    return json.loads(read_text(path, encoding=encoding))


def read_csv(path, **kwargs):
    import pandas as pd

    return pd.read_csv(path, **kwargs)


def read_parquet(path, **kwargs):
    import pandas as pd

    return pd.read_parquet(path, **kwargs)


def open_read(path, *, encoding="utf-8", newline=None):
    return Path(path).open("r", encoding=encoding, newline=newline)


def touch_file(path):
    """建立空檔案，已存在時保留其內容。"""
    Path(path).touch()


def save_figure(figure, path, *, dpi=None, bbox_inches=None, image_format=None):
    """保存呼叫者已完成的圖，沿用其指定的輸出選項。"""
    options = {}
    if dpi is not None:
        options["dpi"] = dpi
    if bbox_inches is not None:
        options["bbox_inches"] = bbox_inches
    if image_format is not None:
        options["format"] = image_format
    figure.savefig(path, **options)


def save_demo_trace(save_dir, episode_frame, step_frame):
    """保存互動軌跡工具原有的兩個資料表。"""
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Demo traces use the same optional-column metadata as training traces while
    # retaining the lean table contract accepted by the replay panel.
    save_trace(
        destination,
        episode_frame,
        step_frame,
        core_columns={
            "episode": ["episode", "init_UAV_pos", "init_device_pos", "risk_region"],
            "step": ["episode", "step", "UAV_pos", "device_pos", "device_AOI",
                     "UAV_action_select", "UAV_action_move", "reward"],
        },
    )


def _safe_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return value


def write_json(path, value, *, safe=False, strict=False, newline=False, atomic=False):
    destination = Path(path)
    if safe:
        value = _safe_json_value(value)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=not (safe or strict))
    if newline:
        text += "\n"
    if atomic:
        import tempfile

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            write_text(temporary_path, text, encoding="utf-8")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    else:
        write_text(destination, text, encoding="utf-8")


def write_csv(path, headers, rows, *, encoding="utf-8-sig", dict_rows=False):
    import csv

    headers = list(headers)
    with Path(path).open("w", newline="", encoding=encoding) as stream:
        if dict_rows:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
        else:
            writer = csv.writer(stream)
            writer.writerow(headers)
        writer.writerows(rows)


def write_json_line(stream, value, *, safe=False):
    if safe:
        value = _safe_json_value(value)
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def open_log(path, *, append=False, errors="replace"):
    if append:
        mode = "a"
    else:
        mode = "w"
    return Path(path).open(mode, encoding="utf-8", errors=errors)


def write_log(stream, line):
    stream.write(line)
    stream.flush()


def save_preset(destination, preset):
    """以原子取代保存範本；路徑與覆寫確認由呼叫端管理。"""
    import tempfile

    destination = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                prefix=f".{destination.stem}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(preset, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def save_profile(profiler):
    """保存 TrainingProfiler 報告的 JSON、CSV、pstats 與文字摘要。"""
    import csv
    import pstats

    profiler.output_dir.mkdir(parents=True, exist_ok=True)
    (profiler.output_dir / "profile.json").write_text(
        json.dumps(profiler.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (profiler.output_dir / "profile.csv").open(
            "w", newline="", encoding="utf-8-sig") as csv_file:
        columns = (
            "category",
            "calls",
            "inclusive_seconds",
            "exclusive_seconds",
            "percent_total",
        )
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in columns}
            for row in profiler.report["categories"]
        )

    profiler._profiler.dump_stats(str(profiler.output_dir / "profile.prof"))
    with (profiler.output_dir / "profile_calls.txt").open(
            "w", encoding="utf-8") as calls_file:
        for sort_key, heading in (
            ("tottime", "Top functions by self time"),
            ("cumulative", "Top functions by cumulative time"),
        ):
            calls_file.write(f"{heading}\n")
            stats = pstats.Stats(profiler._profiler, stream=calls_file)
            stats.strip_dirs().sort_stats(sort_key).print_stats(40)
            calls_file.write("\n")


def save_profile_summary(output_dir, rows):
    """寫入 profile_training 的 JSON、CSV 與 Markdown 摘要。"""
    import csv

    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        columns = ["name", "method", "device", "status", "wall_seconds",
                   "process_seconds", "top_category", "top_category_seconds", "report_dir",
                   "exit_code", "report_read_error"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    lines = ["# 訓練耗時分析", "",
             "各場在獨立程序依序執行。分類比例使用 exclusive 時間，inclusive 欄位供查看完整呼叫成本，不能直接相加。",
             "分析包含計時與 cProfile 開銷；比較優化速度時應另跑未開啟分析的相同工作量。", "",
             "| 實驗 | 方法／裝置 | 訓練分析秒數 | 最耗時分類 | 該分類秒數 | 狀態 |",
             "|---|---|---:|---|---:|---|"]
    for row in rows:
        name = row["name"].replace("|", "\\|")
        wall = row.get("wall_seconds")
        if wall is not None:
            seconds = f"{wall:.3f}"
        else:
            seconds = "—"
        top_seconds = row.get("top_category_seconds")
        if top_seconds is not None:
            top_text = f"{top_seconds:.3f}"
        else:
            top_text = "—"
        lines.append(f"| {name} | {row['method']} / {row['device']} | {seconds} | "
                     f"{row.get('top_category', '—')} | {top_text} | {row['status']} |")
    report_errors = [row for row in rows if row.get("report_read_error")]
    if report_errors:
        lines.extend(["", "## 無法讀取的分析報告", ""])
        for row in report_errors:
            lines.append(f"- {row['name']}（exit code {row.get('exit_code')}）："
                         f"{row['report_read_error']}")
    lines.extend(["", "正常完成訓練與分析的目錄包含 config.json、training.log、profile.json、profile.csv、",
                  "profile.prof、profile_calls.txt 與正常的 results/。CUDA 未同步時為主機端耗時，",
                  "不能視為各 GPU 核心的實際執行時間；使用 --cuda-sync 可在分類邊界同步。", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_legacy_training(method, config, agents, eval_rewards, *,
                         training_rewards=None, max_steps=None, model=None):
    """保存舊版 MA_C* 或 DQN_Online 的訓練輸出。"""
    if method not in ("MA_CCQL", "MA_CCQR", "MA_CIQL", "MA_CIQR", "DQN_Online"):
        raise ValueError(f"Unsupported legacy training method: {method!r}")

    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    if method == "DQN_Online":
        if training_rewards is None:
            raise ValueError("DQN_Online requires its accumulated training DataFrame")
        if max_steps is None:
            raise ValueError("DQN_Online requires env.max_steps as max_steps")
        dataset_path = Path(config.PATH) / "Datasets" / (
            f"Dataset_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_"
            f"{str(config.penalty)}pen_RND.csv"
        )
        save_start = int((config.episodes * max_steps) / 2)
        save_end = int(save_start + config.data_size * (config.episodes * max_steps) / 100)
        training_rewards.iloc[save_start:save_end].to_csv(dataset_path)

        figure_path = Path(config.PATH) / "Results" / (
            f"Result_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_"
            f"{str(config.penalty)}pen_RND.png"
        )
        data_path = Path(config.PATH) / "Result_Datas" / (
            f"Result_Online_DQN_{str(config.data_size)}%_{str(config.U)}UAVs_"
            f"{str(config.penalty)}pen_RND.csv"
        )
        title = "Online DQN Reward"
    else:
        if model is None:
            raise ValueError("Legacy MA output requires its caller-provided model name")
        for u in range(config.U):
            model_path = Path(config.PATH) / "Saved_Models" / (
                f"Model_Offline_{model}_{str(config.data_size)}%_UAV_{str(u)}_"
                f"{str(config.penalty)}pen.pth"
            )
            torch.save(agents[u].network.state_dict(), model_path)
        figure_path = Path(config.PATH) / "Results" / (
            f"Result_Offline_{model}_{str(config.data_size)}%_{str(config.U)}UAVs_"
            f"{str(config.penalty)}pen.png"
        )
        data_path = Path(config.PATH) / "Result_Datas" / (
            f"Result_Offline_{model}_{str(config.data_size)}%_{str(config.U)}UAVs_"
            f"{str(config.penalty)}pen.csv"
        )
        title = f"MA_{model} Reward"

    window_size = 10
    eval_rewards_smooth = pd.Series(eval_rewards).rolling(window=window_size).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards_smooth, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_path)
    pd.DataFrame(eval_rewards).to_csv(data_path, index=False)
