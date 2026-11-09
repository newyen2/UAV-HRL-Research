"""共用設定、參數驗證及亂數工具；不得反向引用方法入口、agent 或環境。"""
import argparse
import random
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# 以下為目前正式預設值；尺寸、索引和公式常數不當作超參數。
ENV_SETTING = SimpleNamespace(
    M=10,  # 裝置數量；預設 10。
    U=2,  # UAV/agent 數量；預設 2。
    map_size=50.0,  # 正方形地圖邊長；預設 50.0。
    max_mov=2.0,  # 每步最大移動距離；預設 2.0。
    energy_weight=500,  # 環境 reward 的傳輸功耗權重；預設 500。
    penalty=300,  # 每次抽中風險事件的懲罰；預設 300。
    prob_of_risk=0.1,  # 位於風險區時每步受罰機率；預設 0.1。
    arrival_model="always",  # 固定週期／Bernoulli／Poisson；沿用 always 鍵，預設週期 1 維持舊行為。
    arrival_interval=1,  # 固定週期模式每隔多少環境時槽產生新資料；正整數，預設 1。
    arrival_probability=0.2,  # Bernoulli 每裝置每步到達機率；預設 0.2。
    arrival_rate=0.2,  # Poisson 每裝置每步到達率；預設 0.2。
    channel_gain=1e-3,  # 通道增益（線性值，對應 -30 dB）；預設 1e-3。
    altitude=100.0,  # UAV 飛行高度（公尺）；預設 100.0。
    cell_distance=100.0,  # 每個地圖座標單位對應的實際距離（公尺）；預設 100.0。
    bandwidth=1e6,  # 傳輸頻寬（Hz）；預設 1e6。
    packet_size=5e6,  # 每次服務封包大小（bit）；預設 5e6。
    noise_power=1e-12,  # 功率公式的雜訊功率（W），保留舊公式有效值；預設 1e-12。
    communication_model="snr",  # 通訊排程與功率模型；SNR 保留原本的獨立鏈路。
    max_transmit_power=1.0,  # SINR 模式中每個地面裝置的峰值發射功率上限（W）。
    fixed_transmit_power=0.05,  # 自適應排程的固定地面裝置發射功率（W）。
    subchannel_count=2,  # 子頻道模式可用的最大等頻寬子頻道數。
    min_device_distance=5.0,  # 隨機裝置間的最低距離（地圖單位）；預設 5.0。
    max_coord_attempts=100000,  # 裝置座標採樣的總嘗試上限，亦為每架 UAV 的座標採樣上限；預設 100000。
)
TRAIN_DEVICE_OPTIONS = ("cpu", "cuda")
ENV_ARRIVAL_MODEL_OPTIONS = ("always", "bernoulli", "poisson")
ENV_COMMUNICATION_MODEL_OPTIONS = ("snr", "sinr_hybrid", "sinr_adaptive", "sinr_subchannels")
HRL_LOWER_ALGORITHM_OPTIONS = ("a2c", "sac", "dsac", "dsac_t")
HRL_LOWER_WARMUP_MODE_OPTIONS = ("random", "demonstration")
HRL_UPPER_REWARD_MODE_OPTIONS = [
    "aoi_risk_count", "option_mean_environment", "average_reward_smdp"]
TRAIN_SETTING = SimpleNamespace(
    device="cuda",  # 訓練裝置：cpu 或 cuda（CUDA GPU）；預設 cuda。
    max_steps=200,  # 每個訓練 episode 的步數；預設 200。
    episodes=200,  # 所有方法共用的總訓練回合預算，可由 train.py 覆寫；預設 200。
    buffer_size=100000,  # 每個 replay pool 最多保存的經驗數；預設 100000。
    batch_size=32,  # DQN 系列、MA-CQR 及 HRL 上層每次抽樣數；預設 32。
    seed=1,  # 訓練 NumPy、Python、Torch 隨機種子；預設 1。
)
EVAL_SETTING = SimpleNamespace(
    max_steps=200,  # 每個評估 episode 的步數；預設 200。
    eval_runs=10,  # 每次評估的回合數；預設 10。
    seed=1,  # 評估亂數序列的種子；預設 1。
    eval_periods=1,  # 每隔多少訓練回合執行評估；預設 1。
)
RECORD_SETTING = SimpleNamespace(
    PATH=Path(__file__).resolve().parent / 'Outputs' / 'Experiments',  # 結果根目錄；預設專案目錄下 Outputs/Experiments。
    reward_smoothing_window=10,  # eval reward 移動平均視窗，1 表示不平滑；預設 10。
    figure_size=(12, 6),  # 共用結果圖寬、高（英吋）；預設 (12, 6)。
)
FUNCTIONAL_SETTING = SimpleNamespace(
    random_device_coord=True,  # reset 時隨機放置裝置；預設 True。
    random_UAV_coord=True,  # reset 時隨機放置 UAV；預設 True。
    risk=True,  # 是否啟用風險區與隨機懲罰；預設 True。
)
FIXED_SETTING = SimpleNamespace(
    # 關閉隨機裝置座標時使用；預設以下 10 個 (x,y)，列數須等於 M。
    fixed_device_coord=np.array([[1,6], [3,1], [7,2], [6,7], [7,5], [8,5], [9,1], [6,1], [4,7], [2,3]], dtype=float),
    # 關閉隨機 UAV 座標時使用；預設 [(0,0), (10,10)]，列數須等於 U。
    fixed_UAV_coord=np.array([[0,0], [10,10]], dtype=float),
    # 風險矩形 (xmin,ymin,xmax,ymax)；預設 [(15,10,35,35)]。
    fixed_risk_region=np.array([[15,10,35,35]], dtype=float),
)

DQN_SETTING = SimpleNamespace(
    hidden_size=256,  # Q 網路隱藏層寬度；預設 256。
    gamma=0.99,  # 未來 reward 折扣率，跨步經驗使用 gamma 的持續步數次方；預設 0.99。
    tau=0.001,  # 目標網路 soft update 的新權重比例；預設 0.001。
    lr=1e-4,  # Adam 學習率；預設 1e-4。
    warmup_steps=500,  # 預填 replay 的步數；greedy/service 表示每個池的最低完整經驗數；預設 500。
    start_eps=1.0,  # epsilon-greedy 起始隨機探索機率；預設 1.0。
    min_eps=0.01,  # epsilon-greedy 最低探索機率；預設 0.01。
    eps_frames=10000,  # 探索率從起始值線性下降至最低值的環境步數；預設 10000。
)

DQN_SERVICE_SETTING = SimpleNamespace(
    hidden_size=256,  # Q 網路隱藏層寬度；預設 256。
    service_ratio=0.1,  # 移動後距離小於等於地圖邊長乘此比例時自動服務；預設 0.1。
    gamma=0.99,  # 未來 reward 折扣率，跨步經驗使用 gamma 的持續步數次方；預設 0.99。
    tau=0.001,  # 目標網路 soft update 的新權重比例；預設 0.001。
    lr=1e-4,  # Adam 學習率；預設 1e-4。
    warmup_steps=500,  # 預填 replay 的步數；greedy/service 表示每個池的最低完整經驗數；預設 500。
    start_eps=1.0,  # epsilon-greedy 起始隨機探索機率；預設 1.0。
    min_eps=0.01,  # epsilon-greedy 最低探索機率；預設 0.01。
    eps_frames=10000,  # 探索率從起始值線性下降至最低值的環境步數；預設 10000。
)

HRL_UPPER_SETTING = SimpleNamespace(
    hidden_size=256,  # 每架 UAV 獨立上層候選評分網路寬度。
    gamma=0.99,  # 折扣模式依 Option 的持續步數折扣。
    tau=0.001,  # 每個上層 target 的 soft update 比例。
    lr=1e-4,  # 每個上層 DQN Adam 學習率。
    warmup_steps=500,  # 每個 UAV 獨立 replay 的最低預填片段數。
    reward_mode="aoi_risk_count",  # 預設僅使用 AoI 降低量。
    risk_penalty_weight=0.0,  # 預設不扣上層風險。
    average_reward_lr=1e-4,  # 平均回報 SMDP 的 rho 更新率。
)
HRL_LOWER_SETTING = SimpleNamespace(
    hidden_size=256,  # 連續策略 actor 與 critic 隱藏層寬度；預設 256。
    algorithm="sac",  # 下層連續控制更新演算法；預設 SAC。
    warmup_steps=500,  # 下層 off-policy replay 每個 UAV 池的最低 transition 數；設為 0 時略過預填；預設 500。
    warmup_mode="random",  # 下層預填動作：random 隨機、demonstration 直線示範；未 pretrain 時強制隨機，A2C 不使用。
    a2c_steps=5,  # A2C 每次 on-policy rollout 的環境步數；預設 5。
    gamma=0.99,  # 下層 reward 折扣率；預設 0.99。
    tau=0.005,  # 下層 target critic soft update 比例；預設 0.005。
    alpha=0.001,  # SAC／DSAC 系列固定熵正則係數（A2C 不使用）；預設 0.001。
    imitation_weight=0.05,  # HRL 聯合訓練起始模仿權重；0 關閉此項，不影響示範初始化；預設 0.05。
    imitation_decay_episodes=0,  # HRL 聯合訓練線性衰減所需已完成回合數；0 表示固定使用起始權重。
    min_imitation_weight=0.0,  # HRL 聯合訓練模仿權重下限；衰減為 0 時忽略。
    actor_lr=1e-4,  # actor Adam 學習率；預設 1e-4。
    critic_lr=1e-4,  # critic Adam 學習率；預設 1e-4。
    log_std_min=-3.0,  # 高斯策略 log 標準差下限；預設 -3.0。
    log_std_max=-0.5,  # 高斯策略 log 標準差上限；預設 -0.5。
    pretrain_batch_size=256,  # 每次示範預訓練抽樣數；預設 256。
    std_min=0.1,  # 分布式 critic 回報標準差下限；預設 0.1。
    td_bound=20.0,  # DSAC 標準差梯度中的 TD 誤差界限；預設 20.0。
    std_rate=0.005,  # DSACv2 回報標準差指數移動平均更新率；預設 0.005。
    policy_delay=2,  # DSAC 系列每幾次 critic 更新才更新 actor 與 target；預設 2。
)
HRL_SETTING = SimpleNamespace(
    service_ratio=0.1,  # 到達/服務距離門檻占地圖邊長比例；預設 0.1。
    upper_fraction=0.1,  # 上層固定回合占比，回合數取 int(總回合 * 比例)；0 略過上層，預設 0.1。
    lower_fraction=0.1,  # 下層固定回合占比，回合數取 int(總回合 * 比例)；0 略過下層，預設 0.1。
    convergence_window=5,  # 收斂診斷視窗長度，至少 2；只記錄、不控制階段切換，預設 5。
    convergence_tolerance=0.02,  # 收斂診斷的相對範圍容許值，不改變固定回合數；預設 0.02。
    convergence_patience=2,  # 收斂診斷需連續穩定的視窗數；預設 2。
    convergence_min_episodes=5,  # 開始收斂診斷前至少訓練的回合數；預設 5。
    progress_weight=1.0,  # 下層接近目標獎勵權重；預設 1.0。
    success_reward=1.0,  # 下層完成目標服務的獎勵；預設 1.0。
    step_penalty=0.05,  # 下層每步時間懲罰；預設 0.05。
    risk_weight=1.0,  # 下層實際風險懲罰的相對權重；預設 1.0。
    risk_penalty_reference=300.0,  # 下層風險懲罰尺度；受罰 300 對應 1 倍 risk_weight。
    cpu_threads=16,  # HRL 執行時 Torch CPU 執行緒數；預設 16。
    lower_batch_size=128,  # 下層 off-policy replay 每次抽樣數；預設 128。
    fragment_steps=32,  # 上層未完成目標每段經驗最多累積步數；預設 32。
    start_eps=1.0,  # 上層 DQN 起始 epsilon-greedy 探索率；預設 1.0。
    min_eps=0.01,  # 上層 DQN 最低 epsilon-greedy 探索率；預設 0.01。
    eps_decisions=2000,  # 上層 DQN 探索率衰減的決策數；預設 2000。
    reach_threshold=0.95,  # 下層收斂診斷的最低導航到達率，不阻止進入聯合訓練；預設 0.95。
    reach_cases=128,  # 每次導航能力驗證案例數；預設 128。
    reach_steps=100,  # 每個導航驗證案例最多步數；預設 100。
    pretrain_batches=400,  # 下層前的一次性離線示範初始化更新批數；0 關閉初始化，不影響 imitation_weight；預設 400。
    navigation_cases=16384,  # 示範預訓練的位置/目標配對數；預設 16384。
    navigation_eval_seed=1,  # 固定導航驗證案例與採樣策略的種子；預設 1。
)
HRL_PPO_LOWER_ALGORITHM_OPTIONS = ("ppo", "a2po")
HRL_PPO_LOWER_ORDER_OPTIONS = ("greedy", "random")
HRL_PPO_UPPER_REWARD_MODE_OPTIONS = (
    "aoi_risk_count", "option_mean_environment", "average_reward_smdp")
HRL_PPO_UPPER_SETTING = SimpleNamespace(
    reward_mode="aoi_risk_count",  # 上層回報模式；預設使用 AoI 降低量。
    risk_penalty_weight=0.0,  # AoI 模式下每個實際受罰事件的上層扣分；預設 0。
    average_reward_lr=1e-4,  # 平均回報 SMDP 的 rho 更新率；預設 1e-4。
    hidden_size=256,  # HRL_PPO 上層 Q 網路隱藏層寬度；預設 256。
    gamma=0.99,  # 上層未來回報折扣率；預設 0.99。
    tau=0.001,  # 上層目標網路 soft update 比例；預設 0.001。
    lr=1e-4,  # 上層 Q 網路 Adam 學習率；預設 1e-4。
    warmup_steps=500,  # 每個上層 Replay 最低預填片段數；至少 TRAIN.batch_size，預填不學習。
    start_eps=1.0,  # 上層起始隨機探索機率；預設 1.0。
    min_eps=0.01,  # 上層最低隨機探索機率；預設 0.01。
    eps_decisions=2000,  # 上層探索率衰減所需的決策數；預設 2000。
)
HRL_PPO_LOWER_SETTING = SimpleNamespace(
    hidden_size=128,  # HRL_PPO 下層 actor 與 critic 隱藏層寬度；預設 128。
    algorithm="a2po",  # 下層更新演算法：ppo 或 a2po；預設 a2po。
    actor_lr=0.0003,  # 下層 actor Adam 學習率；預設 0.0003。
    critic_lr=0.0003,  # 下層 critic Adam 學習率；預設 0.0003。
    gamma=0.99,  # 下層未來回報折扣率；預設 0.99。
    gae_lambda=0.95,  # GAE 優勢估計的 lambda；預設 0.95。
    clip_ratio=0.2,  # PPO policy ratio clipping 範圍；預設 0.2。
    epochs=5,  # 每批 rollout 的更新回合數；預設 5。
    minibatch_size=64,  # 每次 PPO 更新使用的 minibatch 樣本數；預設 64。
    rollout_steps=128,  # 下層每次 on-policy rollout 的環境步數；預設 128。
    entropy_coef=0.001,  # 下層策略 entropy 正則係數；預設 0.001。
    target_kl=0.02,  # PPO 每輪更新的 KL 停止門檻；0 關閉門檻，預設 0.02。
    log_std_min=-3.0,  # 下層高斯策略 log 標準差下限；預設 -3.0。
    log_std_max=0.0,  # 下層高斯策略 log 標準差上限；預設 0.0。
    pretrain_batches=200,  # 一次性離線導航示範初始化更新批數；0 關閉初始化，預設 200。
    pretrain_batch_size=128,  # 每次示範初始化抽取的樣本數；預設 128。
    navigation_cases=4096,  # 建立示範初始化資料的位置與目標配對數；預設 4096。
    order="greedy",  # A2PO agent 更新順序：greedy 或 random；預設 greedy。
    trace_clip=1.0,  # A2PO GAE 軌跡中的策略比率上限；預設 1.0。
)
HRL_PPO_SETTING = SimpleNamespace(
    service_ratio=0.1,  # 到達與服務距離占地圖邊長的比例；預設 0.1。
    navigation_fraction=0.0,  # 示範初始化後預設直接聯合訓練；提高此比例可加入導航暖身。
    fragment_steps=32,  # AoI 模式下每段上層經驗步數上限，不會解除目標鎖定；預設 32。
    progress_weight=1.0,  # 下層接近目標回報權重；預設 1.0。
    success_reward=1.0,  # 下層完成目標服務的回報；預設 1.0。
    step_penalty=0.05,  # 下層每步時間成本；預設 0.05。
    risk_weight=1.0,  # 下層風險懲罰相對權重；預設 1.0。
    risk_penalty_reference=300.0,  # 下層風險懲罰正規化參照值；預設 300.0。
    environment_reward_weight=0.01,  # 下層環境回報權重；預設 0.01。
    cpu_threads=16,  # HRL_PPO 執行時 Torch CPU 執行緒數；預設 16。
)
MA_CQR_SETTING = SimpleNamespace(
    hidden_size=256,  # 分位數 Q 網路隱藏層寬度；預設 256。
    N=8,  # 每個動作估計的分位數數量；預設 8。
    eta=0.15,  # CVaR 選動作採用的最低回報尾端比例，範圍 (0, 1]；1 為全分布平均，預設 0.15。
    gamma=0.99,  # 集中式 QR reward 折扣率；預設 0.99。
    tau=0.001,  # 各 target 網路 soft update 比例；預設 0.001。
    lr=1e-4,  # 集中式 Adam 學習率；預設 1e-4。
    huber_kappa=1.0,  # Quantile Huber loss 二次/線性區域的分界；預設 1.0。
    warmup_steps=500,  # 聯合 replay 預填環境步數，至少 batch_size；預設 500。
    cpu_threads=1,  # Torch CPU 執行緒數；預設 1。
    start_eps=1.0,  # 起始隨機探索機率；預設 1.0。
    min_eps=0.01,  # 最低探索機率；預設 0.01。
    eps_frames=10000,  # 探索率衰減所需環境步數；預設 10000。
)
MA_CQR_RESEARCH_SETTING = SimpleNamespace(
    bootstrap="double",  # 研究消融使用 double 或 target bootstrap；預設 double。
    eval_interval=20,  # 每隔多少訓練回合執行固定驗證案例；預設 20。
    validation_episodes=20,  # 每次驗證的固定案例數；預設 20。
    validation_seed=200000,  # 驗證場景種子起點；預設 200000。
    validation_risk_seed=600000,  # 驗證風險亂數種子起點；預設 600000。
    final_eval_episodes=200,  # 每個檢查點的凍結 holdout 案例數；預設 200。
    final_eval_seed=100000,  # holdout 場景種子起點；預設 100000。
    final_risk_seed=500000,  # holdout 風險亂數種子起點；預設 500000。
    checkpoint_episodes=[],  # 額外保存模型與執行 holdout 的回合里程碑。
    export_replay_episodes=[],  # 將對齊 replay 匯出的訓練回合里程碑。
    export_replay_path="",  # 可選 NPZ 匯出路徑；空字串表示不匯出。
)
MA_CQR_OFFLINE_SETTING = SimpleNamespace(
    cql_alpha=0.0,
    dataset_path="",
)


# 在設定初始化階段，依環境規模推導各方法的狀態與動作維度。
def update_method_sizes():
    state_size = 2 * ENV_SETTING.U + 3 * ENV_SETTING.M
    device_action_size = ENV_SETTING.M + 1
    joint_action_size = 5 * device_action_size
    DQN_SETTING.state_size = state_size
    DQN_SETTING.action_size = joint_action_size
    DQN_SERVICE_SETTING.state_size = state_size
    DQN_SERVICE_SETTING.action_size = device_action_size
    # 各 UAV 以自己的座標、全體 UAV 座標、裝置座標及 AoI 評分 M+1 個候選。
    HRL_UPPER_SETTING.state_size = 2 + 2 * ENV_SETTING.U + 3 * ENV_SETTING.M
    HRL_UPPER_SETTING.action_size = device_action_size
    HRL_LOWER_SETTING.state_size = 6
    HRL_LOWER_SETTING.action_size = 2
    HRL_SETTING.state_size = HRL_UPPER_SETTING.state_size
    HRL_SETTING.action_size = HRL_UPPER_SETTING.action_size
    HRL_PPO_UPPER_SETTING.state_size = HRL_UPPER_SETTING.state_size
    HRL_PPO_UPPER_SETTING.action_size = device_action_size
    risk_region_count = len(FIXED_SETTING.fixed_risk_region)
    HRL_PPO_LOWER_SETTING.state_size = 7 + 2 * ENV_SETTING.U + 4 * risk_region_count
    HRL_PPO_LOWER_SETTING.action_size = 2
    HRL_PPO_LOWER_SETTING.critic_state_size = (
        5 * ENV_SETTING.U + 3 * ENV_SETTING.M + 4 * risk_region_count)
    MA_CQR_SETTING.state_size = state_size
    MA_CQR_SETTING.action_size = joint_action_size


_DERIVED_SETTING_FIELDS = {"state_size", "action_size"}
_METHOD_DERIVED_SETTING_FIELDS = {
    "HRL_PPO_LOWER": {"critic_state_size"},
}


# 只在所屬方法中排除專屬推導欄位，避免改變既有方法的未知欄位驗證。
def _derived_setting_fields(group):
    return _DERIVED_SETTING_FIELDS | _METHOD_DERIVED_SETTING_FIELDS.get(group, set())


# 將執行期設定轉為 validator 可讀的原生資料，不改寫或遮蔽非法數值。
def _setting_value_for_validation(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return [_setting_value_for_validation(item) for item in value]
    if isinstance(value, list):
        return [_setting_value_for_validation(item) for item in value]
    return value


# 驗證目前命名空間的完整設定；所有方法通過後才更新推導尺寸。
def initialize_settings(method=None):
    """Validate current settings for one method or every method at startup."""
    from runtime.configuration import METHODS, default_settings, validate_settings

    if method is None:
        selected_methods = METHODS
    else:
        selected_methods = (method,)
    for selected_method in selected_methods:
        defaults = default_settings(selected_method)
        current = {}
        for group in defaults:
            namespace = globals()[group + "_SETTING"]
            expected_fields = set(defaults[group])
            derived_fields = _derived_setting_fields(group)
            actual_fields = set(vars(namespace)) - {"PATH"} - derived_fields
            missing_fields = expected_fields - actual_fields
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"{group} settings are missing required fields: {missing}")
            fields = {}
            for field, value in vars(namespace).items():
                if field == "PATH" or field in derived_fields:
                    continue
                fields[field] = _setting_value_for_validation(value)
            current[group] = fields
        validate_settings(selected_method, current)

    update_method_sizes()


# 驗證完整或局部外部設定，成功後才套用並更新所有方法尺寸。
def apply_settings(method, settings):
    """Validate and apply settings for an external method invocation."""
    from runtime.configuration import validate_settings

    validated = validate_settings(method, settings)
    for group, fields in validated.items():
        namespace = globals()[group + "_SETTING"]
        for field, value in fields.items():
            if group == "FIXED":
                value = np.asarray(value, dtype=float)
            elif group == "RECORD" and field == "figure_size":
                value = tuple(value)
            setattr(namespace, field, value)
    update_method_sizes()
    return validated


initialize_settings()


# 集中各命令列入口的參數定義與 trace 輸入驗證。
def parse_args(entry, argv=None):
    if entry == "train":
        from runtime.configuration import (METHODS, inactive_setting_paths,
                                               setting_options, validate_experiment)

        parser = argparse.ArgumentParser(description="以共用設定啟動指定訓練方法")
        parser.add_argument("-model", "--model", required=True, choices=METHODS)
        setting_flags = {}
        setting_help = {
            ("ENV", "communication_model"): (
                "通訊模型，預設 snr。SINR 可選同頻／分時擇優、固定功率自適應排程或子頻道配置。"),
            ("ENV", "max_transmit_power"): "SINR 的每裝置峰值發射功率上限（W），預設 1.0。",
            ("ENV", "fixed_transmit_power"): "sinr_adaptive 的固定發射功率（W），預設 0.05。",
            ("ENV", "subchannel_count"): "sinr_subchannels 可用的最大子頻道數，預設 2。",
            ("HRL", "upper_fraction"): (
                "上層固定回合占比；不等待收斂，0 略過上層；"
                "預設 0.1。"),
            ("HRL", "lower_fraction"): (
                "下層固定回合占比；不等待收斂，0 略過下層；"
                "預設 0.1。"),
            ("HRL", "risk_penalty_reference"): (
                "下層風險懲罰參照值；300/300=1、1500/300=5 倍 risk_weight；預設 300。"),
            ("HRL_LOWER", "imitation_weight"): (
                "HRL 聯合訓練的起始模仿權重；示範初始化不受此值影響。"),
            ("HRL_LOWER", "imitation_decay_episodes"): (
                "HRL 聯合訓練模仿權重線性衰減所需的已完成回合數；0 表示固定使用起始權重。"),
            ("HRL_LOWER", "min_imitation_weight"): (
                "HRL 聯合訓練衰減後的模仿權重下限；衰減回合數為 0 時忽略。"),
            ("HRL_LOWER", "algorithm"): (
                "下層連續控制演算法：a2c 為同步 n-step Advantage Actor-Critic；"
                "sac、dsac、dsac_t 使用 replay。"),
            ("HRL_LOWER", "a2c_steps"): (
                "A2C 每次 rollout 收集的環境步數；正整數，預設 5，其他下層演算法不使用。"),
            ("HRL_LOWER", "warmup_mode"): (
                "下層 replay 預填方式：random 隨機動作（預設）、demonstration 直線示範；"
                "pretrain_batches=0 時強制隨機，A2C 不使用。"),
            ("HRL_UPPER", "reward_mode"): (
                "上層回報模式：aoi_risk_count 使用 AoI 降低量扣除實際受罰次數；"
                "option_mean_environment 使用完整 Option 內每步環境回報的算術平均；"
                "average_reward_smdp 使用平均回報 SMDP。"),
            ("HRL_UPPER", "risk_penalty_weight"): (
                "AoI降低量減係數×實際受罰次數；預設0純AoI、1既有rerun；負值反向獎勵受罰。"),
            ("HRL_UPPER", "average_reward_lr"): (
                "平均回報 SMDP 中 rho 的 TD 誤差更新學習率；這是 rho 更新率，不是環境懲罰係數。"),
            ("HRL_PPO_UPPER", "reward_mode"): (
                "上層回報模式：aoi_risk_count 使用 AoI 降低量扣除實際受罰次數；"
                "option_mean_environment 使用完整 Option 內每步環境回報的算術平均；"
                "average_reward_smdp 使用平均回報 SMDP。"),
            ("HRL_PPO_UPPER", "risk_penalty_weight"): (
                "僅 AoI 降低量模式且 FUNCTIONAL.risk 開啟時生效；"
                "須為非負值，以此係數乘上每架 UAV 的實際受罰次數後從上層回報扣除。"),
            ("HRL_PPO_UPPER", "average_reward_lr"): (
                "僅平均回報 SMDP 中更新 rho 的 TD 誤差學習率；不是環境風險懲罰係數。"),
            ("HRL_PPO_UPPER", "warmup_steps"): (
                "每架 UAV 的上層 Replay 最低預填完整片段數；有效值至少為 TRAIN.batch_size，"
                "預填期間只收集經驗，不進行學習。"),
            ("HRL_PPO", "fragment_steps"): (
                "僅 AoI 降低量模式生效：目標尚未完成時到步數上限便建立片段，"
                "但不解除目標鎖定；其他回報模式使用完整 Option。"),
        }
        for setting_name, setting in globals().items():
            if not setting_name.endswith("_SETTING") or not isinstance(setting, SimpleNamespace):
                continue
            if setting_name in {"MA_CQR_RESEARCH_SETTING", "MA_CQR_OFFLINE_SETTING"}:
                continue
            group = setting_name[:-len("_SETTING")]
            derived_fields = _derived_setting_fields(group)
            for field, value in vars(setting).items():
                if field in derived_fields | {"td_loss_scale", "grad_clip"}:
                    continue
                option = f"--{setting_name}.{field}"
                dest = f"_setting_{setting_name}_{field}"
                if setting_name == "RECORD_SETTING" and field == "PATH":
                    parser.add_argument(option, "--output-dir", "--output_dir",
                                        dest="output_dir", type=Path, metavar="PATH")
                    continue
                if isinstance(value, bool) or isinstance(value, (np.ndarray, list, tuple)):
                    value_type = _parse_json_setting
                elif group == "ENV" and field in {"energy_weight", "penalty"}:
                    value_type = float
                elif isinstance(value, int):
                    value_type = int
                elif isinstance(value, float):
                    value_type = float
                elif isinstance(value, Path):
                    value_type = Path
                else:
                    value_type = str
                argument_options = {
                    "dest": dest,
                    "type": value_type,
                    "default": argparse.SUPPRESS,
                }
                choices = setting_options(group, field)
                if choices:
                    argument_options["choices"] = choices
                if (group, field) in setting_help:
                    argument_options["help"] = setting_help[(group, field)]
                parser.add_argument(option, **argument_options)
                setting_flags[dest] = (group, field)
    elif entry == "experiment_worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
    elif entry == "trace_tool":
        parser = argparse.ArgumentParser()
        parser.add_argument("--episode_parquet", type=str)
        parser.add_argument("--step_parquet", type=str)
        parser.add_argument("--episode", type=int, default=1, help="紀錄片段編號，不代表環境終止")
        parser.add_argument("--interval", type=int, default=500, help="自動播放時每幀間隔，單位 ms")
        parser.add_argument("--steps", type=int, default=120, help="未指定 parquet 時，用固定策略產生的步數")
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--save_dir", type=Path, help="將紀錄儲存為 episode.parquet 與 step.parquet")
        parser.add_argument("--snapshot", type=Path, help="輸出最後一幀 PNG，使用無視窗模式")
        parser.add_argument(
            "--display",
            type=str,
            default="all",
            choices=["all", "move-only", "action-only", "static"],
            help="all: 完全顯示, move-only: 包含移動軌跡, action-only: 包含服務動作, static: 僅狀態",
        )
        parser.add_argument("--trajectory", type=int, default=10, help="UAV軌跡保留最近幾個step，預設 10")
    elif entry == "result_output":
        parser = argparse.ArgumentParser(description='從各方法的 Result_Datas 重繪 Results')
        parser.add_argument('--root', type=Path, default=RECORD_SETTING.PATH)
        parser.add_argument('--method', help='指定方法；省略時重繪所有已有結果的方法')
    else:
        raise ValueError(f"Unknown CLI entry: {entry!r}")

    if entry == "train":
        if argv is None:
            raw_argv = list(sys.argv[1:])
        else:
            raw_argv = list(argv)
        options = parser._option_string_actions
        normalized_argv = []
        for token in raw_argv:
            if token == "=" and normalized_argv and normalized_argv[-1] in options:
                continue
            normalized_argv.append(token)
        args = parser.parse_args(normalized_argv)
        supplied = {}
        for dest, (group, field) in setting_flags.items():
            if hasattr(args, dest):
                supplied.setdefault(group, {})[field] = getattr(args, dest)
        try:
            validated = validate_experiment({
                "name": "command-line", "method": args.model, "settings": supplied})
            inactive = inactive_setting_paths(args.model, validated["settings"])
            for group, fields in supplied.items():
                for field in fields:
                    if (group, field) in inactive:
                        raise ValueError(
                            f"設定 {group}.{field} 在目前配置沒有作用：{inactive[(group, field)]}")
        except ValueError as error:
            parser.error(str(error))
        return argparse.Namespace(model=args.model, settings=validated["settings"],
                                  output_dir=args.output_dir)

    args = parser.parse_args(argv)
    if entry == "trace_tool":
        if bool(args.episode_parquet) != bool(args.step_parquet):
            parser.error("--episode_parquet 與 --step_parquet 必須一起提供")
        if args.steps < 0 or args.trajectory < 0 or args.interval <= 0:
            parser.error("steps/trajectory 必須非負，interval 必須大於零")
    return args


def _parse_json_setting(value):
    """解析設定中的 JSON 布林值與陣列，將格式錯誤交給 argparse。"""
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("must be a JSON value") from error




class RNGManager:
    # 分別初始化訓練與評估亂數序列，切回訓練模式。
    def __init__(self):
        import torch
        self.states = {}

        np.random.seed(TRAIN_SETTING.seed)
        random.seed(TRAIN_SETTING.seed)
        torch.manual_seed(TRAIN_SETTING.seed)

        self.states["train"] = self._capture_state()

        np.random.seed(EVAL_SETTING.seed)
        random.seed(EVAL_SETTING.seed)
        torch.manual_seed(EVAL_SETTING.seed)

        self.states["eval"] = self._capture_state()

        self.mode = "eval"

        self.switch_mode("train")

    # 保存 NumPy、Python、Torch 的目前亂數狀態。
    def _capture_state(self):
        import torch
        state = {"np": np.random.get_state(), "random": random.getstate(),
                 "torch": torch.get_rng_state()}
        if TRAIN_SETTING.device == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    # 保存目前亂數狀態，切換到指定模式的序列。
    def switch_mode(self, r_mode):
        import torch
        if r_mode == self.mode:
            return
        if self.states[r_mode] is None:
            return

        self.states[self.mode] = self._capture_state()

        np.random.set_state(self.states[r_mode]["np"])
        random.setstate(self.states[r_mode]["random"])
        torch.set_rng_state(self.states[r_mode]["torch"])
        if TRAIN_SETTING.device == "cuda":
            torch.cuda.set_rng_state_all(self.states[r_mode]["cuda"])

        self.mode = r_mode

    # 取得指定模式保存的亂數狀態。
    def get_state(self, r_mode):
        return self.states[r_mode]

    # 替換指定模式保存的亂數狀態。
    def force_set_state(self, r_mode ,r_states):
        self.states[r_mode] = r_states
