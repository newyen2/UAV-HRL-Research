from runtime.progress import Progress
from runtime.record import Record
from runtime.fileIO import FileIO
from types import SimpleNamespace
from runtime.training_evaluation import record_evaluation
import time
import numpy as np
import random
from buffer import ReplayBuffer
from utils import DQN_SETTING, RNGManager
from agent_DQN import DQNAgent
from Environment import Environment
import utils

MOVE_VECTORS = np.array([[0., 1.], [0., -1.], [-1., 0.], [1., 0.], [0., 0.]])

# action_id = move_action * (M + 1) + select_action；M 代表不服務。
def to_env_action(env, move_action, select_action):
    return [select_action, MOVE_VECTORS[move_action] * env.max_mov]


# 依設定步數隨機採樣，保存單步經驗至各 UAV 的 replay。
def collect_random(env, dataset):
    steps = max(DQN_SETTING.warmup_steps, max(buffer.batch_size for buffer in dataset))
    env.reset()
    state = env.get_state()
    for _ in range(steps):
        env_actions = []
        action_ids = []

        for _ in range(env.U):
            select_action = random.randrange(env.M + 1)
            move_action = random.randrange(5)

            env_actions.append(to_env_action(env, move_action, select_action))
            action_ids.append(move_action * (env.M + 1) + select_action)

        env.step(env_actions)
        reward = env.rewards
        next_state = env.get_state()
        
        for u in range(env.U):
            dataset[u].add(state, [action_ids[u]], reward[u], next_state)

        if env.steps >= env.max_steps:
            env.reset()
            state = env.get_state()
        else:
            state = next_state

# 依 utils.EVAL_SETTING.eval_runs 執行評估並取得相容的回傳值
@record_evaluation
def eval_runs(env, agent, return_metrics=False):
    rewards = []
    aois = []

    for _ in range(utils.EVAL_SETTING.eval_runs):
        env.reset()
        state = env.get_state()
        aoi_sum = 0.
        episode_reward = 0.

        while env.steps < env.max_steps:
            env_actions = []
            for u in range(env.U):
                action_id = int(agent[u].get_action(state, epsilon=0.)[0])
                move_action, select_action = divmod(action_id, env.M + 1)
                env_actions.append(to_env_action(env, move_action, select_action))
                
            env.step(env_actions)
            state = env.get_state()
            
            episode_reward += env.total_reward
            aoi_sum += float(np.mean(env.AOI))

        rewards.append(episode_reward)
        aois.append(aoi_sum / env.steps)

    # 需要完整指標時回傳獎勵與 AoI，否則只回傳獎勵。
    if return_metrics:
        return float(np.mean(rewards)), float(np.mean(aois))
    return np.mean(rewards)

# 執行 replay 預填、DQN 訓練、定期評估與結果存檔。
def train():
    training = SimpleNamespace(training_started_at=time.perf_counter(), phase="train")
    # 初始化RNG, 環境與超參數
    rng = RNGManager()
    env = Environment()

    steps = 0 

    # 初始化Agent與ReplayBuffer
    agents = []
    buffers = []
    for _ in range(env.U):
        agents.append(DQNAgent())
        buffers.append(ReplayBuffer())

    # 三個輸出組件共用執行狀態，各自解析格式與管理輸出。
    training.env = env
    training.eval_env = env
    training.agents = agents
    progress = Progress('DQN', training)
    record = Record('DQN', training)
    files = FileIO('DQN', training)
    # 按 DQN_SETTING.warmup_steps 隨機採取動作，預填每個 Agent 的 Replay Buffer。
    progress.set_stage("warmup")
    collect_random(env, buffers)

    for episode in range(1, utils.TRAIN_SETTING.episodes + 1):
        training.episode = episode
        training.evaluated = False
        training.episode_started_at = time.perf_counter()
        progress.episode_start()
        env.reset()
        state = env.get_state()

        files.episode_start()
        training.aoi_sum = 0.
        training.episode_reward = 0.  # 本回合每步各 UAV 平均環境獎勵的累積值。
        training.agent_rewards = np.zeros(env.U)  # 各 UAV 的回合環境回報；平均值等於 episode_reward。

        while env.steps < env.max_steps:
            # 這是本環境步使用的探索率，回合結果因此記錄最後環境步的值。
            training.epsilon = max(
                DQN_SETTING.start_eps
                - (steps * (DQN_SETTING.start_eps - DQN_SETTING.min_eps))
                / DQN_SETTING.eps_frames,
                DQN_SETTING.min_eps,
            )
            # 個別UAV取得動作並執行
            training.env_actions = []
            action_ids = []
            for u in range(env.U):
                action_id = agents[u].get_action(state, epsilon=training.epsilon)[0]
                action_ids.append(action_id)
                move_action, select_action = divmod(int(action_id), env.M + 1)
                training.env_actions.append(to_env_action(env, move_action, select_action))

            env.step(training.env_actions)
            reward = env.rewards
            next_state = env.get_state()
            files.step()

            # 紀錄Experiment至Replay Buffer並採樣訓練
            for u in range(env.U):
                buffers[u].add(state, [action_ids[u]], reward[u], next_state)
                training.agent_rewards[u] += reward[u]
                agents[u].learn(buffers[u].sample())

            training.episode_reward += env.total_reward
            training.aoi_sum += float(np.mean(env.AOI))
            progress.step()

            state = next_state
            steps += 1

        training.episode_steps = env.steps
        record.episode()
        progress.episode_end()
        # 定時進行測試評估
        if episode % utils.EVAL_SETTING.eval_periods == 0:
            rng.switch_mode("eval")
            try:
                eval_runs(env, agents, return_metrics=True)
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
