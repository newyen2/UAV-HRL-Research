from runtime.progress import Progress
from runtime.record import Record
from runtime.fileIO import FileIO
from types import SimpleNamespace
from runtime.training_evaluation import record_evaluation
import time
import numpy as np
import random
from utils import DQN_SERVICE_SETTING, RNGManager
from buffer import DurationReplayBuffer, ServiceReplayBuffer
from agent_DQN_service import DQNAgent
from Environment import Environment
import utils

# Planning 的 target_action 是目標裝置索引；Service 輸入為二值 select_action。
def to_env_action(env, u, target_action, select_action):
    if target_action == env.M:
        return [env.M, np.zeros(2)]
    delta = env.device_coord[target_action] - env.UAV_coord[u]
    distance = np.linalg.norm(delta)
    # move_action 是 Moving 位移；超過單步上限才縮短，較近時直接抵達目標。
    if distance > env.max_mov:
        # 乘上「允許距離 / 原距離」，保持方向並把長度縮到 max_mov。
        move_action = delta * (env.max_mov / distance)
    else:
        move_action = delta
    next_coord = env.Update_Location(env.UAV_coord[u], move_action)
    in_service_range = (np.linalg.norm(env.device_coord[target_action] - next_coord)
                        <= env.map_size * DQN_SERVICE_SETTING.service_ratio)
    # 輸入 select_action 是二值服務請求；以下輸出實際環境服務索引或 env.M。
    # Environment 先移動再服務；門檻含等號，服務頭也可提早服務。
    if select_action or in_service_range:
        select_action = target_action
    else:
        select_action = env.M
    return [select_action, move_action]


# 預先收集足夠的完整決策，避免長距離移動造成樣本不足。
def collect_random(env, dataset, service_dataset):
    all_buffers = dataset + service_dataset
    required = max(DQN_SERVICE_SETTING.warmup_steps,
                   max(buffer.batch_size for buffer in all_buffers))
    env.reset()
    # None 表示該 UAV 目前沒有進行中的 Planning 決策。
    target_actions = [None] * env.U
    decision_states = [None] * env.U
    decision_rewards = np.zeros(env.U)
    decision_steps = np.zeros(env.U, dtype=int)
    state = env.get_state()
    while min(len(buffer) for buffer in all_buffers) < required:
        env_actions = []
        # select_actions 記錄二值服務嘗試，供 replay 使用。
        select_actions = []
        for u in range(env.U):
            if target_actions[u] is None:
                target_actions[u] = random.randrange(env.M + 1)
                decision_states[u] = state
                decision_rewards[u] = 0.
                decision_steps[u] = 0
            if target_actions[u] != env.M:
                select_action = random.randrange(2)
            else:
                # 懸停沒有服務目標，不執行二值服務決策。
                select_action = 0
            env_action = to_env_action(env, u, target_actions[u], select_action)
            # 強制服務也要記成實際執行的服務動作，供兩個 replay 使用。
            select_actions.append(int(
                target_actions[u] != env.M and env_action[0] == target_actions[u]
            ))
            env_actions.append(env_action)

        env.step(env_actions)
        next_state = env.get_state()
        for u in range(env.U):
            if target_actions[u] != env.M:
                next_forced_service = (
                    select_actions[u] == 0
                    and to_env_action(env, u, target_actions[u], select_action=False)[0] == target_actions[u]
                )
                service_dataset[u].add(state, [target_actions[u]], [select_actions[u]],
                                       env.rewards[u], next_state, bool(next_forced_service))
            # 按目前決策的持續步數折扣，完成目標時再寫入跨步 replay。
            decision_rewards[u] += DQN_SERVICE_SETTING.gamma ** decision_steps[u] * env.rewards[u]
            decision_steps[u] += 1
            if target_actions[u] == env.M or select_actions[u]:
                dataset[u].add(decision_states[u], [target_actions[u]], decision_rewards[u],
                               next_state, decision_steps[u])
                target_actions[u] = None
        state = next_state
        if env.steps >= env.max_steps:
            # episode 終點不封存未完成目標；重置時丟棄其區域累積值。
            env.reset()
            target_actions = [None] * env.U
            decision_states = [None] * env.U
            decision_rewards = np.zeros(env.U)
            decision_steps = np.zeros(env.U, dtype=int)
            state = env.get_state()


# 依 utils.EVAL_SETTING.eval_runs 執行評估並取得相容的回傳值。
@record_evaluation
def eval_runs(env, agent, return_metrics=False):
    rewards = []
    aois = []
    for _ in range(utils.EVAL_SETTING.eval_runs):
        env.reset()
        # None 表示該 UAV 目前沒有進行中的 Planning 決策。
        target_actions = [None] * env.U
        state = env.get_state()
        aoi_sum = 0.
        episode_reward = 0.
        while env.steps < env.max_steps:
            env_actions = []
            for u in range(env.U):
                if target_actions[u] is None:
                    target_actions[u] = int(agent[u].get_action(state, epsilon=0.)[0])
                if target_actions[u] != env.M:
                    select_action = int(agent[u].get_service_action(
                        state, target_actions[u], epsilon=0.
                    )[0])
                else:
                    select_action = 0
                env_action = to_env_action(env, u, target_actions[u], select_action)
                env_actions.append(env_action)
            env.step(env_actions)
            state = env.get_state()
            for u in range(env.U):
                if target_actions[u] == env.M or env_actions[u][0] == target_actions[u]:
                    target_actions[u] = None
            episode_reward += env.total_reward
            aoi_sum += float(np.mean(env.AOI))
        rewards.append(episode_reward)
        aois.append(aoi_sum / env.steps)
    # 需要完整指標時回傳獎勵與 AoI，否則只回傳獎勵。
    if return_metrics:
        return float(np.mean(rewards)), float(np.mean(aois))
    return np.mean(rewards)


# 預填目標與服務經驗，訓練雙頭 DQN 並評估、存檔。
def train():
    training = SimpleNamespace(training_started_at=time.perf_counter(), phase="train")
    # 初始化RNG, 環境與超參數
    rng = RNGManager()
    env = Environment()

    steps = 0

    # 初始化Agent與ReplayBuffer
    agents = []
    buffers = []
    service_buffers = []
    for _ in range(env.U):
        agents.append(DQNAgent())
        buffers.append(DurationReplayBuffer())
        service_buffers.append(ServiceReplayBuffer())

    # 透過隨機採取完整動作，預先填入每個Agent的Replay Buffer
    training.env = env
    training.eval_env = env
    training.agents = agents
    progress = Progress('DQN_service', training)
    record = Record('DQN_service', training)
    files = FileIO('DQN_service', training)
    progress.set_stage("warmup")
    collect_random(env, buffers, service_buffers)

    for episode in range(1, utils.TRAIN_SETTING.episodes + 1):
        training.episode = episode
        training.evaluated = False
        training.episode_started_at = time.perf_counter()
        progress.episode_start()
        # 沿用訓練片段重置；上個片段尚未完成的決策不寫入 replay，亦不標成終止。
        env.reset()
        state = env.get_state()

        # None 表示該 UAV 目前沒有進行中的 Planning 決策。
        training.target_actions = [None] * env.U
        decision_states = [None] * env.U
        decision_rewards = np.zeros(env.U)
        decision_steps = np.zeros(env.U, dtype=int)
        files.episode_start()
        training.aoi_sum = 0.
        training.episode_reward = 0.  # 本回合每步各 UAV 平均環境獎勵的累積值。
        training.agent_rewards = np.zeros(env.U)  # 各 UAV 的回合環境回報；平均值等於 episode_reward。

        while env.steps < env.max_steps:
            # 這是本環境步使用的探索率，回合結果因此記錄最後環境步的值。
            training.epsilon = max(
                DQN_SERVICE_SETTING.start_eps
                - (steps * (DQN_SERVICE_SETTING.start_eps - DQN_SERVICE_SETTING.min_eps))
                / DQN_SERVICE_SETTING.eps_frames,
                DQN_SERVICE_SETTING.min_eps,
            )
            # 個別UAV取得動作並執行
            training.env_actions = []
            # select_actions 記錄實際執行的二值服務動作，供 trace 與 replay 使用。
            training.select_actions = []
            for u in range(env.U):
                if training.target_actions[u] is None:
                    training.target_actions[u] = int(agents[u].get_action(state, epsilon=training.epsilon)[0])
                    decision_states[u] = state
                    decision_rewards[u] = 0.
                    decision_steps[u] = 0
                if training.target_actions[u] != env.M:
                    select_action = int(agents[u].get_service_action(
                        state, training.target_actions[u], epsilon=training.epsilon
                    )[0])
                else:
                    select_action = 0
                env_action = to_env_action(env, u, training.target_actions[u], select_action)
                # replay 與 trace 都記錄服務嘗試，包含距離觸發的強制嘗試。
                training.select_actions.append(int(
                    training.target_actions[u] != env.M and env_action[0] == training.target_actions[u]
                ))
                training.env_actions.append(env_action)

            env.step(training.env_actions)
            reward = env.rewards
            next_state = env.get_state()
            files.step()

            # 先寫完所有 UAV 的兩種 replay，再進入原本逐 UAV 的學習順序。
            for u in range(env.U):
                if training.target_actions[u] != env.M:
                    next_forced_service = (
                        training.select_actions[u] == 0
                        and to_env_action(env, u, training.target_actions[u], select_action=False)[0] == training.target_actions[u]
                    )
                    service_buffers[u].add(state, [training.target_actions[u]], [training.select_actions[u]],
                                           reward[u], next_state, bool(next_forced_service))
                # gamma 指數使用此目標先前已持續的步數，再增加本步長度。
                decision_rewards[u] += DQN_SERVICE_SETTING.gamma ** decision_steps[u] * reward[u]
                decision_steps[u] += 1
                if training.target_actions[u] == env.M or training.select_actions[u]:
                    buffers[u].add(decision_states[u], [training.target_actions[u]], decision_rewards[u],
                                    next_state, decision_steps[u])
                    # 目標在懸停或提出服務嘗試後解鎖，由環境另判定是否送達。
                    training.target_actions[u] = None

            # 完成的決策才寫入Replay Buffer；每個環境步仍採樣訓練
            for u in range(env.U):
                training.agent_rewards[u] += reward[u]
                agents[u].learn(buffers[u].sample())
                agents[u].learn_service(service_buffers[u].sample())

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
