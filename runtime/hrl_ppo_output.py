"""HRL_PPO records and checkpoints, using the shared result serialization."""
import time

import numpy as np

from runtime.fileIO import (
    FileIO,
    _communication_episode_record,
    _communication_step_record,
)
import utils


class HRLPPOFileIO(FileIO):
    def __init__(self, trainer):
        super().__init__("HRL_PPO", trainer)

    def episode_start(self):
        t = self.trainer
        self.upper_start = t.upper_update_counts.copy()
        self.lower_start = t.lower_update_counts.copy()
        record = dict(episode=t.episode, phase=t.phase,
            init_UAV_pos=t.env.UAV_coord.tolist(), init_device_pos=t.env.device_coord.tolist(),
            init_device_AOI=t.env.AOI.tolist(), init_source_AOI=t.env.source_AOI.tolist(),
            risk_region=t.env.risk_region.tolist(),
            map_size=t.env.map_size,
            service_radius=t.env.map_size * utils.HRL_PPO_SETTING.service_ratio,
            service_rule="within_range")
        record.update(_communication_episode_record(t.env))
        self.episode_records.append(record)

    def step(self):
        t = self.trainer
        record = dict(episode=t.episode, phase=t.phase, step=t.env.steps,
            UAV_pos=t.env.UAV_coord.tolist(), device_pos=t.env.device_coord.tolist(),
            device_AOI=t.env.AOI.tolist(), source_AOI=t.env.source_AOI.tolist(),
            UAV_target=list(t.target_actions),
            UAV_current_position=list(t.current_position_actions),
            UAV_action_select=[int(a[0]) for a in t.env_actions],
            UAV_action_move=[a[1].tolist() for a in t.env_actions],
            reward=list(t.env.rewards), total_reward=t.env.total_reward,
            upper_reward=t.upper_step_rewards.tolist(),
            upper_fragment_stored=t.upper_fragment_stored.tolist(),
            upper_fragment_reward=t.upper_fragment_rewards.tolist(),
            upper_fragment_duration=t.upper_fragment_durations.tolist(),
            upper_reward_mode=utils.HRL_PPO_UPPER_SETTING.reward_mode,
            upper_rho=[a.average_reward_rate for a in t.upper_agents],
            intrinsic_reward=t.lower_step_rewards.tolist(), lower_team_reward=t.lower_team_reward,
            option_completed=list(t.option_ends), option_timeout=list(t.option_timeouts),
            mean_aoi=float(np.mean(t.env.AOI)), lower_algorithm=utils.HRL_PPO_LOWER_SETTING.algorithm)
        record.update(_communication_step_record(t.env))
        self.step_records.append(record)

    def episode(self):
        t = self.trainer
        now = time.perf_counter()
        upper_algorithm = (
            "average_reward_smdp"
            if utils.HRL_PPO_UPPER_SETTING.reward_mode == "average_reward_smdp"
            else "dqn_duration"
        )
        row = dict(episode=t.episode, phase=t.phase, reward=t.environment_return,
            learning_reward=t.upper_episode_reward, mean_aoi=t.aoi_sum / t.env.steps,
            upper_algorithm=upper_algorithm,
            upper_reward_mode=utils.HRL_PPO_UPPER_SETTING.reward_mode,
            upper_rho=[a.average_reward_rate for a in t.upper_agents],
            upper_fragment_steps=utils.HRL_PPO_SETTING.fragment_steps,
            upper_prefill_env_steps=t.upper_prefill_env_steps,
            upper_prefill_buffer_sizes=list(t.upper_prefill_buffer_sizes),
            lower_algorithm=utils.HRL_PPO_LOWER_SETTING.algorithm,
            upper_updates=int(np.sum(t.upper_update_counts - self.upper_start)),
            lower_updates=int(np.sum(t.lower_update_counts - self.lower_start)),
            upper_updates_per_agent=(t.upper_update_counts - self.upper_start).tolist(),
            lower_updates_per_agent=(t.lower_update_counts - self.lower_start).tolist(),
            upper_decisions=t.upper_decision_counts.tolist(),
            upper_loss=float(np.mean(t.upper_losses)) if t.upper_losses else float("nan"),
            reached=t.reached, option_attempts=t.attempts, option_timeouts=t.timeouts,
            uncompleted_options=t.uncompleted_options,
            reach_rate=t.reached / max(1, t.attempts),
            rollout_count=len(t.lower_metrics),
            update_orders=[m["update_order"] for m in t.lower_metrics],
            navigation_episodes=t.phase_counts["navigation"], joint_episodes=t.phase_counts["joint"],
            episode_wall_time_seconds=now - t.episode_started_at,
            wall_time_seconds=now - t.training_started_at)
        for metric in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"):
            row[metric] = float(np.mean([m[metric] for m in t.lower_metrics]))
        if t.evaluation_metrics is not None:
            row.update(t.evaluation_metrics)
            row["eval_return"] = t.eval_environment_return
            row["evaluation_phase"] = t.phase
            row["evaluation_episode"] = t.episode
        self.results.append(row)

    def prepare(self):
        t = self.trainer
        upper_algorithm = (
            "average_reward_smdp"
            if utils.HRL_PPO_UPPER_SETTING.reward_mode == "average_reward_smdp"
            else "dqn_duration"
        )
        self.model = dict(
            format_version=2, checkpoint_schema="hrl_ppo_ctde_v2",
            upper_algorithm=upper_algorithm,
            upper_reward_mode=utils.HRL_PPO_UPPER_SETTING.reward_mode,
            upper_fragment_steps=utils.HRL_PPO_SETTING.fragment_steps,
            upper_prefill_env_steps=t.upper_prefill_env_steps,
            upper_prefill_buffer_sizes=list(t.upper_prefill_buffer_sizes),
            lower_algorithm=utils.HRL_PPO_LOWER_SETTING.algorithm,
            U=t.env.U, M=t.env.M,
            state_size=utils.HRL_PPO_UPPER_SETTING.state_size,
            action_size=utils.HRL_PPO_UPPER_SETTING.action_size,
            lower_state_size=utils.HRL_PPO_LOWER_SETTING.state_size,
            critic_state_size=utils.HRL_PPO_LOWER_SETTING.critic_state_size,
            observation_schema="normalized_goal_uavs_risk_v1",
            critic_observation_schema="normalized_global_aoi_goals_active_risk_v1",
            action_schema="latent_gaussian_tanh_radial_projection_v1",
            value_semantics="continuing_team_reward_across_options_time_limit_bootstrap",
            upper_value_semantics=(
                "average_reward_smdp_rho_duration"
                if utils.HRL_PPO_UPPER_SETTING.reward_mode == "average_reward_smdp"
                else "duration_discounted_dqn_continuing_option_bootstrap"
            ),
            upper=[dict(network=a.network.state_dict(), target_net=a.target_net.state_dict(),
                        optimizer=a.optimizer.state_dict(),
                        average_reward_rate=a.average_reward_rate) for a in t.upper_agents],
            lower_agents=[dict(actor=a.actor.state_dict(), optimizer=a.optimizer.state_dict())
                          for a in t.lower_agents],
            centralized_critic=t.lower.critic.state_dict(),
            critic_optimizer=t.lower.critic_optimizer.state_dict(),
            upper_update_counts=t.upper_update_counts.tolist(),
            lower_update_counts=t.lower_update_counts.tolist(),
            upper_decision_counts=t.upper_decision_counts.tolist(),
            phase_counts=dict(t.phase_counts), pretrain_loss=t.pretrain_loss,
            training_episode=t.episode,
        )
        self.method_config = {
            "HRL_PPO_SETTING": dict(vars(utils.HRL_PPO_SETTING)),
            "HRL_PPO_UPPER_SETTING": dict(vars(utils.HRL_PPO_UPPER_SETTING)),
            "HRL_PPO_LOWER_SETTING": dict(vars(utils.HRL_PPO_LOWER_SETTING)),
        }
        self.save_started_at = time.perf_counter()
