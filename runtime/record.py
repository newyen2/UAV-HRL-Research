"""訓練與評估的命令列文字輸出。"""
import sys
import traceback


_HRL_METHODS = ("HRL", "HRL_PPO")
_CANONICAL_METHODS = ("DQN", "DQN_service", *_HRL_METHODS, "MA-CQR")


# 輸出各個結果圖檔路徑。
def paths(outputs):
    for output in outputs:
        print(output)


# 顯示耗時分析工作開始資訊。
def profile_started(index, total, spec):
    print(f"[{index}/{total}] {spec['name']} ({spec['method']})", flush=True)


# 顯示耗時分析工作完成資訊。
def profile_completed(row):
    print(f"  {row['status']}: {row['top_category']} "
          f"({row['top_category_seconds']} s)")


# 輸出舊版離線方法的 epoch 摘要。
def legacy_epoch(method, epoch, eval_reward, loss):
    if method in ("MA_CCQR", "MA_CCQL", "MA_CIQL", "MA_CIQR"):
        print(f"Epoch: {epoch} | Eval_Reward: {eval_reward} | Q Loss: {loss}")
    else:
        raise ValueError(f"Unsupported legacy epoch report method: {method!r}")


# 輸出舊版線上 DQN 的單回合摘要。
def legacy_episode(method, episode, reward, *, agent_rewards=None, steps=None):
    if method != "DQN_Online":
        raise ValueError(f"Unsupported legacy episode report method: {method!r}")
    print(f"Episode: {episode} | Reward: {reward} | "
          f"Reward_u{agent_rewards} | Steps: {steps}")


# 輸出舊版線上 DQN 評估回報。
def legacy_evaluation(method, eval_reward):
    if method != "DQN_Online":
        raise ValueError(f"Unsupported legacy evaluation report method: {method!r}")
    print(f"Eval_Reward: {eval_reward}")


# 顯示文件中的備忘錄文字。
def memo(text):
    print(text)


# 將目前例外的 traceback 輸出至標準錯誤。
def exception():
    traceback.print_exc(file=sys.stderr)


class Record:
    """依方法解析執行狀態；只輸出人類閱讀的 CLI 文字。"""

    def __init__(self, method, training):
        if method not in _CANONICAL_METHODS:
            raise ValueError(f"Unsupported record method: {method!r}")
        self.method = method
        self.training = training

    def episode(self):
        training = self.training
        if self.method == "MA-CQR":
            import numpy as np
            loss = float(np.mean(training.losses))
            print(f"Episode: {training.episode} | Train Reward: "
                  f"{training.episode_reward:.3f} | QR Loss: {loss:.5f}")
        elif self.method in _HRL_METHODS:
            print(f"Episode: {training.episode} | Phase: {training.phase} | "
                  f"Train Reward: {training.environment_return} | "
                  f"Upper Reward: {training.upper_episode_reward} | "
                  f"Upper Reward_u{training.upper_agent_rewards}")
            if self.method == "HRL_PPO" and training.lower_metrics:
                import numpy as np
                import utils
                kl = np.mean([row["approx_kl"] for row in training.lower_metrics])
                print(f"Lower: {utils.HRL_PPO_LOWER_SETTING.algorithm} | "
                      f"KL: {kl:.5f} | Reached: {training.reached} | "
                      f"Timeouts: {training.timeouts}")
        else:
            print(f"Episode: {training.episode} | Train Reward: "
                  f"{training.episode_reward} | Reward_u{training.agent_rewards} "
                  f"| Steps: {training.env.steps}")

    def evaluation(self):
        training = self.training
        if self.method in _HRL_METHODS:
            metrics = training.evaluation_metrics
        elif training.evaluated:
            metrics = training.eval_env.last_eval_metrics
        else:
            return
        if metrics is None:
            if self.method in _HRL_METHODS:
                return
            raise ValueError(f"Evaluation metrics are required for {self.method!r}")

        # 與 CSV 合成公式一致；成本及實際懲罰均已按 UAV 數平均。
        eval_reward = -(metrics['eval_aoi'] * metrics['eval_max_steps']
                        + metrics['eval_weighted_power'] + metrics['eval_violation_penalty'])
        print(
            f"Eval (mean over {metrics['eval_episode_count']} episodes) | "
            f"Mean AoI: {metrics['eval_aoi']:.3f} | "
            f"Weighted Tx Cost/UAV: {metrics['eval_weighted_power']:.3f} | "
            f"Risk Zone Steps (all UAVs): {metrics['eval_violation_count']:.3f} | "
            f"Risk Penalty/UAV: {metrics['eval_violation_penalty']:.3f} | "
            f"Eval Reward: {eval_reward:.3f}"
        )
