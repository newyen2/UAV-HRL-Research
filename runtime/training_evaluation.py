"""正式訓練方法共用的評估記錄裝飾器。"""
from functools import wraps
from numbers import Real

import numpy as np

from runtime import progress
import utils


# eval_runs 預設回傳平均環境回報，完整指標模式回傳 (平均環境回報, 平均 AoI)。
def _evaluation_reward(result):
    if isinstance(result, tuple) and result:
        value = result[0]
    else:
        value = result
    if isinstance(value, (Real, np.number)) and not isinstance(value, (bool, np.bool_)):
        return float(value)
    return None


# 統計評估 AoI、功耗與風險，並在結束後恢復環境設定。
def record_evaluation(function):
    # 執行評估並彙整指標，結束後恢復訓練設定。
    @wraps(function)
    def recorded(env, *args, **kwargs):
        original_step = env.step
        had_override = 'step' in env.__dict__
        override = env.__dict__.get('step')
        episodes = []
        train_max_steps = env.max_steps
        max_steps = utils.EVAL_SETTING.max_steps
        env.max_steps = max_steps  # 評估暫用獨立長度，結束後恢復訓練設定。

        # 包裝環境步進，累計評估的 AoI、功耗與風險指標。
        def step(actions):
            result = original_step(actions)
            if env.steps == 1 or not episodes:
                episodes.append({'eval_episode': len(episodes) + 1, 'steps': 0,
                                 'cumulative_aoi': 0., 'cumulative_power': 0.,
                                 'weighted_power': 0., 'violation_penalty': 0.,
                                 'total_violation_penalty': 0., 'violation_count': 0})
            row = episodes[-1]
            row['steps'] += 1
            row['cumulative_aoi'] += float(np.mean(env.AOI))
            # 各鏈路的平均發射功率已按發射時間折算至一秒時槽。
            transmission_power = float(np.sum(env.transmission_power))
            row['cumulative_power'] += transmission_power
            # 每架 UAV 每步移動後在風險區即計一次，包含停留及未抽中懲罰。
            if utils.FUNCTIONAL_SETTING.risk:
                row['violation_count'] += sum(env.is_coord_risky(coord) for coord in env.UAV_coord)
            # 與 env.total_reward 同尺度：各 UAV 加權功耗及實際懲罰取平均。
            row['weighted_power'] += transmission_power * env.energy_weight / env.U
            actual_penalty = float(np.sum(env.risk_penalty_counts)) * env.penalty
            row['total_violation_penalty'] += actual_penalty
            row['violation_penalty'] += actual_penalty / env.U
            progress.step(env.steps, max_steps)
            return result

        env.step = step
        try:
            with progress.stage("evaluation"):
                result = function(env, *args, **kwargs)
        finally:
            env.max_steps = train_max_steps
            if had_override:
                env.step = override
            else:
                del env.step
        for row in episodes:
            row['max_steps'] = max_steps
            # 正式 eval_runs 均跑滿固定步數，因此 mean_aoi × max_steps 可還原累計 AoI。
            row['mean_aoi'] = row.pop('cumulative_aoi') / max_steps
        env.last_eval_metrics = {
            'eval_reward': _evaluation_reward(result),
            'eval_aoi': float(np.mean([r['mean_aoi'] for r in episodes])),
            'eval_max_steps': max_steps,
            'eval_cumulative_power': float(np.mean([r['cumulative_power'] for r in episodes])),
            'eval_violation_count': float(np.mean([r['violation_count'] for r in episodes])),
            'eval_weighted_power': float(np.mean([r['weighted_power'] for r in episodes])),
            'eval_violation_penalty': float(np.mean([r['violation_penalty'] for r in episodes])),
            'eval_total_violation_penalty': float(np.mean([r['total_violation_penalty'] for r in episodes])),
            'eval_episode_count': len(episodes), 'eval_episodes': episodes,
        }
        progress.emit("evaluation_end", stage="evaluation",
                      eval_aoi=env.last_eval_metrics['eval_aoi'],
                      eval_reward=env.last_eval_metrics['eval_reward'])
        return result
    return recorded
