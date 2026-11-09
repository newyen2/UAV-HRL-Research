from runtime.progress import Progress
from runtime.record import Record
from runtime.fileIO import FileIO
from types import SimpleNamespace
from runtime.training_evaluation import record_evaluation
import time
import random
import numpy as np
import torch
import utils
from utils import MA_CQR_SETTING, RNGManager
from Environment import Environment
from buffer import JointReplayBuffer
from agent_MA_CQR import CQRAgent, CentralQRTrainer


MOVE_VECTORS = np.array([[0., 1.], [0., -1.], [-1., 0.], [1., 0.], [0., 0.]])


# 與 DQN 相同的 2U + 3M 觀測，使用原始座標與 AoI，不加入風險區座標。
# 每步聯合動作立即指定 Service 與 Moving；action_id = move_action * (M + 1) + select_action。
# 其中 M 表示本步不服務，且本方法不維持上層 Planning 目標或限制服務距離。
def to_env_actions(env, action_ids):
    env_actions = []
    for action_id in action_ids:
        move_action, select_action = divmod(int(action_id), env.M + 1)
        env_actions.append(
            [select_action, MOVE_VECTORS[move_action] * env.max_mov]
        )
    return env_actions


# 聯合 replay 保存 (s, [a_1,...,a_U], global_r, s')，避免個別取樣打散同一步資料。
def collect_random(env, replay):
    steps = max(MA_CQR_SETTING.warmup_steps, replay.batch_size)
    env.reset()
    state = env.get_state()
    for _ in range(steps):
        action_ids = np.asarray([random.randrange(5 * (env.M + 1)) for _ in range(env.U)])
        env_actions = to_env_actions(env, action_ids)
        env.step(env_actions)
        next_state = env.get_state()
        replay.add(state, action_ids, env.total_reward, next_state)
        # 先保存真正的 s'，再重置訓練片段，不把 reset 狀態當成轉移結果。
        if env.steps >= env.max_steps:
            env.reset()
            state = env.get_state()
        else:
            state = next_state


# 關閉探索進行評估，不寫入 replay、不更新模型參數。
@record_evaluation
def eval_runs(env, agents, return_metrics=False):
    rewards = []
    aois = []
    for _ in range(utils.EVAL_SETTING.eval_runs):
        env.reset()
        episode_reward = 0.
        aoi_sum = 0.
        while env.steps < env.max_steps:
            state = env.get_state()
            action_ids = [agent.get_action(state, epsilon=0.)[0] for agent in agents]
            env_actions = to_env_actions(env, action_ids)
            env.step(env_actions)
            episode_reward += env.total_reward
            aoi_sum += float(np.mean(env.AOI))
        rewards.append(episode_reward)
        aois.append(aoi_sum / env.steps)
    metrics = (float(np.mean(rewards)), float(np.mean(aois)))
    # 需要完整指標時回傳獎勵與 AoI，否則只回傳獎勵。
    if return_metrics:
        return metrics
    return metrics[0]


# Online 訓練入口；預設沿用 utils 的 episode、batch、容量、環境與評估頻率。
def train():
    training = SimpleNamespace(training_started_at=time.perf_counter(), phase="train")
    torch.set_num_threads(MA_CQR_SETTING.cpu_threads)
    rng = RNGManager()
    env = Environment()
    eval_env = Environment()
    env.reset()
    agents = [CQRAgent() for _ in range(env.U)]
    trainer = CentralQRTrainer(agents)
    replay = JointReplayBuffer()
    training.env = env
    training.eval_env = eval_env
    training.agents = agents
    progress = Progress('MA-CQR', training)
    record = Record('MA-CQR', training)
    files = FileIO('MA-CQR', training)
    progress.set_stage("warmup")
    collect_random(env, replay)
    steps = 0
    for episode in range(1, utils.TRAIN_SETTING.episodes + 1):
        training.episode = episode
        training.evaluated = False
        training.episode_started_at = time.perf_counter()
        progress.episode_start()
        env.reset()
        state = env.get_state()
        files.episode_start()
        training.episode_reward = 0.  # 合作式環境回報，已按 UAV 取平均後逐步累積。
        training.aoi_sum = 0.
        training.losses = []
        while env.steps < env.max_steps:
            # 每個環境步線性降低 epsilon，並限制在 min_eps 以上。
            training.epsilon = max(
                MA_CQR_SETTING.start_eps
                - (steps * (MA_CQR_SETTING.start_eps - MA_CQR_SETTING.min_eps))
                / MA_CQR_SETTING.eps_frames,
                MA_CQR_SETTING.min_eps,
            )
            action_ids = np.asarray(
                [agent.get_action(state, epsilon=training.epsilon)[0] for agent in agents],
                dtype=np.int64,
            )
            training.env_actions = to_env_actions(env, action_ids)
            env.step(training.env_actions)
            next_state = env.get_state()
            replay.add(state, action_ids, env.total_reward, next_state)
            training.losses.append(trainer.learn(replay.sample()))
            training.episode_reward += env.total_reward
            training.aoi_sum += float(np.mean(env.AOI))
            progress.step()
            files.step()
            state = next_state
            steps += 1
        training.episode_steps = env.steps
        progress.episode_end()
        record.episode()
        if episode % utils.EVAL_SETTING.eval_periods == 0:
            rng.switch_mode("eval")
            try:
                eval_runs(eval_env, agents, return_metrics=True)
                training.evaluated = True
                record.evaluation()
            finally:
                rng.switch_mode("train")
        files.episode()

    files.prepare()
    progress.set_stage("saving")
    files.save()



if __name__ == "__main__":
    train()
