"""Human-readable Traditional Chinese metadata for experiment settings.

This module is presentation-only: it does not load settings, validate values,
or import the GUI, utilities, or training code.
"""

from collections.abc import Iterable


_GROUP_LABELS = {
    "ENV": "環境",
    "TRAIN": "訓練",
    "EVAL": "評估",
    "FUNCTIONAL": "環境功能",
    "FIXED": "固定座標與區域",
    "RECORD": "結果記錄",
    "DQN": "DQN 學習與探索",
    "DQN_SERVICE": "服務導向 DQN",
    "HRL_UPPER": "HRL 上層",
    "HRL_LOWER": "HRL 下層",
    "HRL": "HRL 階段與回報",
    "HRL_PPO_UPPER": "HRL-PPO 上層",
    "HRL_PPO_LOWER": "HRL-PPO 下層",
    "HRL_PPO": "HRL-PPO 階段與回報",
    "MA_CQR": "MA-CQR 合作式分位數",
}

_GROUP_HELP = {
    "ENV": "設定地圖規模、移動限制、通訊成本與風險事件等環境條件。",
    "TRAIN": "設定訓練回合預算、每回合步數、亂數種子、裝置與 replay 共用資源。",
    "EVAL": "設定訓練期間評估的回合數、步數上限、評估間隔與獨立亂數種子。",
    "FUNCTIONAL": "切換隨機座標生成及風險區功能；固定座標資料見「固定座標與區域」。",
    "FIXED": "提供關閉隨機座標時使用的裝置、UAV 座標，以及風險矩形座標。",
    "RECORD": "設定評估回報的平滑視窗與結果圖尺寸。",
    "DQN": "設定一般 DQN 的 Q 網路、折扣與目標網路更新、Replay 預填及探索。",
    "DQN_SERVICE": "設定服務導向 DQN 的網路、距離觸發服務、Replay 與探索。",
    "HRL_UPPER": "設定上層目標選擇器及其 Option 回報模式；回報模式會決定部分係數是否適用。",
    "HRL_LOWER": "設定下層連續導航演算法、Replay 預填、策略與 critic 學習，以及模仿項。",
    "HRL": "設定 HRL 階段回合配額、下層回報、導航診斷及一次性示範初始化。",
    "HRL_PPO_UPPER": "設定 HRL-PPO 上層目標選擇器、Option 回報模式與探索率。",
    "HRL_PPO_LOWER": "設定 PPO／A2PO 下層 actor-critic 更新、rollout 與示範初始化。",
    "HRL_PPO": "設定 HRL-PPO 階段回合比例、AoI 模式的上層片段上限與下層導航回報。",
    "MA_CQR": "設定合作式分位數 Q 網路、CVaR 動作選擇、Quantile Huber 更新與探索。",
}

_FIELD_LABELS = {
    "ENV": {
        "M": "裝置數量",
        "U": "UAV 數量",
        "map_size": "地圖邊長",
        "max_mov": "每步移動上限",
        "energy_weight": "傳輸成本權重",
        "penalty": "風險事件罰款",
        "prob_of_risk": "風險區受罰機率（0–1）",
        "arrival_model": "裝置封包到達模型",
        "arrival_interval": "固定到達週期（時槽）",
        "arrival_probability": "Bernoulli 到達機率（0–1）",
        "arrival_rate": "Poisson 到達率（封包／裝置／步）",
        "channel_gain": "通道增益",
        "altitude": "飛行高度（m）",
        "cell_distance": "座標單位實距（m）",
        "bandwidth": "傳輸頻寬（Hz）",
        "packet_size": "服務封包大小（bit）",
        "noise_power": "雜訊功率（W）",
        "communication_model": "通訊模型",
        "max_transmit_power": "峰值發射功率上限（W）",
        "fixed_transmit_power": "固定發射功率（W）",
        "subchannel_count": "最大子頻道數",
        "min_device_distance": "裝置最低間距",
        "max_coord_attempts": "座標採樣嘗試上限",
    },
    "TRAIN": {
        "device": "訓練運算裝置",
        "max_steps": "每個訓練回合步數上限",
        "episodes": "訓練總回合預算",
        "buffer_size": "每個 Replay 池容量",
        "batch_size": "共用更新批次大小",
        "seed": "訓練亂數種子",
    },
    "EVAL": {
        "max_steps": "每個評估回合步數上限",
        "eval_runs": "每次評估回合數",
        "seed": "評估亂數種子",
        "eval_periods": "評估間隔回合數",
    },
    "FUNCTIONAL": {
        "random_device_coord": "隨機生成裝置座標",
        "random_UAV_coord": "隨機生成 UAV 座標",
        "risk": "啟用風險區與事件罰款",
    },
    "FIXED": {
        "fixed_device_coord": "固定裝置座標",
        "fixed_UAV_coord": "固定 UAV 座標",
        "fixed_risk_region": "固定風險矩形座標",
    },
    "RECORD": {
        "reward_smoothing_window": "評估回報平滑視窗",
        "figure_size": "結果圖寬高",
    },
    "DQN": {
        "hidden_size": "Q 網路隱藏層寬度",
        "gamma": "未來回報折扣率",
        "tau": "目標網路更新比例",
        "lr": "Q 網路學習率",
        "warmup_steps": "Replay 預填量",
        "start_eps": "起始探索率",
        "min_eps": "最低探索率",
        "eps_frames": "探索率衰減步數",
    },
    "DQN_SERVICE": {
        "hidden_size": "Q 網路隱藏層寬度",
        "service_ratio": "自動服務距離比例",
        "gamma": "未來回報折扣率",
        "tau": "目標網路更新比例",
        "lr": "Q 網路學習率",
        "warmup_steps": "Replay 預填量",
        "start_eps": "起始探索率",
        "min_eps": "最低探索率",
        "eps_frames": "探索率衰減步數",
    },
    "HRL_UPPER": {
        "reward_mode": "上層 Option 回報模式",
        "risk_penalty_weight": "上層受罰事件係數",
        "average_reward_lr": "平均回報率更新率",
        "hidden_size": "上層評分網路寬度",
        "gamma": "上層未來回報折扣率",
        "tau": "上層目標網路更新比例",
        "lr": "上層 Q 網路學習率",
        "warmup_steps": "每個上層 Replay 預填片段數",
    },
    "HRL_LOWER": {
        "algorithm": "下層連續控制演算法",
        "warmup_mode": "Replay 預填動作來源",
        "warmup_steps": "每個下層 Replay 預填步數",
        "a2c_steps": "A2C rollout 步數",
        "hidden_size": "Actor 與 Critic 隱藏層寬度",
        "actor_lr": "Actor 學習率",
        "critic_lr": "Critic 學習率",
        "gamma": "下層未來回報折扣率",
        "tau": "Target Critic 更新比例",
        "alpha": "固定熵正則係數",
        "log_std_min": "策略 log 標準差下限",
        "log_std_max": "策略 log 標準差上限",
        "imitation_weight": "訓練中起始模仿權重",
        "imitation_decay_episodes": "模仿權重衰減回合數",
        "min_imitation_weight": "衰減後模仿權重下限",
        "pretrain_batch_size": "示範初始化批次大小",
        "std_min": "分布式 Critic 標準差下限",
        "td_bound": "DSAC 標準差梯度 TD 界限",
        "std_rate": "DSAC-T 標準差更新率",
        "policy_delay": "延遲策略更新間隔",
    },
    "HRL": {
        "upper_fraction": "上層固定回合比例（0–1）",
        "lower_fraction": "下層固定回合比例（0–1）",
        "pretrain_batches": "一次性示範初始化批數",
        "navigation_cases": "示範導航案例數",
        "navigation_eval_seed": "導航驗證亂數種子",
        "service_ratio": "到達與服務距離比例",
        "progress_weight": "接近目標回報權重",
        "success_reward": "完成服務回報",
        "step_penalty": "下層每步時間成本",
        "risk_weight": "下層風險回報權重",
        "risk_penalty_reference": "下層風險懲罰參照值",
        "fragment_steps": "上層片段步數上限",
        "start_eps": "上層起始探索率",
        "min_eps": "上層最低探索率",
        "eps_decisions": "上層探索率衰減決策數",
        "lower_batch_size": "下層 Replay 批次大小",
        "cpu_threads": "HRL CPU 執行緒數",
        "convergence_window": "收斂診斷視窗長度",
        "convergence_tolerance": "收斂診斷相對範圍容許值",
        "convergence_patience": "連續穩定診斷視窗數",
        "convergence_min_episodes": "各階段收斂旗標最低回合數",
        "reach_threshold": "導航診斷最低到達率",
        "reach_cases": "導航驗證案例數",
        "reach_steps": "每個導航案例步數上限",
    },
    "HRL_PPO_UPPER": {
        "reward_mode": "上層 Option 回報模式",
        "risk_penalty_weight": "上層受罰事件係數",
        "average_reward_lr": "平均回報率更新率",
        "hidden_size": "上層 Q 網路隱藏層寬度",
        "gamma": "上層未來回報折扣率",
        "tau": "上層目標網路更新比例",
        "lr": "上層 Q 網路學習率",
        "warmup_steps": "每架 UAV 上層 Replay 最低預填片段數",
        "start_eps": "上層起始探索率",
        "min_eps": "上層最低探索率",
        "eps_decisions": "上層探索率衰減決策數",
    },
    "HRL_PPO_LOWER": {
        "hidden_size": "下層 Actor 與 Critic 隱藏層寬度",
        "algorithm": "下層 PPO 演算法",
        "actor_lr": "下層 Actor 學習率",
        "critic_lr": "下層 Critic 學習率",
        "gamma": "下層未來回報折扣率",
        "gae_lambda": "GAE 優勢估計係數",
        "clip_ratio": "PPO 策略比例裁剪範圍",
        "epochs": "每批 Rollout 更新回合數",
        "minibatch_size": "PPO Minibatch 大小",
        "rollout_steps": "每次 Rollout 環境步數",
        "entropy_coef": "策略熵正則係數",
        "target_kl": "PPO 目標 KL 停止門檻",
        "log_std_min": "策略 log 標準差下限",
        "log_std_max": "策略 log 標準差上限",
        "pretrain_batches": "一次性示範初始化批數",
        "pretrain_batch_size": "示範初始化批次大小",
        "navigation_cases": "示範導航案例數",
        "order": "A2PO Agent 更新順序",
        "trace_clip": "A2PO GAE 軌跡比率上限",
    },
    "HRL_PPO": {
        "service_ratio": "到達與服務距離比例",
        "navigation_fraction": "導航預訓練回合比例",
        "fragment_steps": "上層片段步數上限",
        "progress_weight": "接近目標回報權重",
        "success_reward": "完成服務回報",
        "step_penalty": "下層每步時間成本",
        "risk_weight": "下層風險回報權重",
        "risk_penalty_reference": "下層風險懲罰參照值",
        "environment_reward_weight": "環境回報權重",
        "cpu_threads": "HRL-PPO CPU 執行緒數",
    },
    "MA_CQR": {
        "hidden_size": "分位數網路隱藏層寬度",
        "N": "每個動作的分位數數量",
        "eta": "CVaR 最低回報尾端比例",
        "gamma": "未來回報折扣率",
        "tau": "目標網路更新比例",
        "lr": "聯合網路學習率",
        "huber_kappa": "Quantile Huber 分界值",
        "warmup_steps": "共享 Replay 預填步數",
        "cpu_threads": "CPU 執行緒數",
        "start_eps": "起始探索率",
        "min_eps": "最低探索率",
        "eps_frames": "探索率衰減步數",
    },
}

_FIELD_HELP = {
    "ENV": {
        "M": "環境中的服務裝置總數；也決定裝置座標與 AoI 陣列長度。",
        "U": "同時運作的 UAV／agent 數量；各方法依自身架構分配或共享 agent。",
        "map_size": "正方形地圖的邊長，座標範圍由地圖原點延伸至此值。",
        "max_mov": "環境每步會將 UAV 位移限制在此距離內。",
        "energy_weight": "乘在傳輸功率上的環境成本權重；風險事件罰款另行加入。",
        "penalty": "每次抽中風險事件時加入環境傳輸成本的罰款金額。",
        "prob_of_risk": "UAV 位於風險矩形內時，每步抽中實際受罰事件的機率。",
        "arrival_model": "固定週期模式每隔 n 槽同步產生新資料，n=1 即每槽到達；bernoulli 依每步機率獨立到達；poisson 依 1−exp(−rate) 判斷至少一次到達。固定模式保留 always 設定鍵，相容既有設定。",
        "arrival_interval": "固定週期模式使用的正整數 n；在 reset 後第 n、2n、3n 槽到達，將裝置資料 AoI 更新為 1。UAV AoI 仍需服務才更新；預設 1 保留原有每槽到達行為。",
        "arrival_probability": "bernoulli 模式下，每個裝置每步獨立產生新資料的機率 p；新資料取代舊資料。與 Poisson 比較時，對齊條件為 p=1−exp(−rate)。",
        "arrival_rate": "poisson 模式下，每個裝置每個環境步的 Poisson 到達率；以 1−exp(−rate) 轉成至少一個封包的機率。",
        "channel_gain": "傳輸功率公式使用的線性通道增益。",
        "altitude": "UAV 與地面裝置間的垂直高度，單位為公尺（m），用於傳輸距離計算。",
        "cell_distance": "每一個地圖座標單位換算成的實際水平距離，單位為公尺（m）。",
        "bandwidth": "通訊模型的總頻寬（Hz）。子頻道模式將這份總頻寬分配給各子頻道。",
        "packet_size": "單次服務所需傳送的封包位元數。",
        "noise_power": "總頻寬內的雜訊功率（W）。子頻道模式依各子頻道頻寬比例分配雜訊。",
        "communication_model": "SNR 為獨立鏈路。SINR 可選同頻／分時擇優、固定功率自適應排程，或子頻道配置。",
        "max_transmit_power": "SINR 模式中單一地面裝置發射期間的功率上限。無法在上限與時槽內完成的服務留待後續排程。",
        "fixed_transmit_power": "固定功率自適應排程的裝置發射功率，需小於或等於峰值上限。",
        "subchannel_count": "可選用的等頻寬子頻道數上限。子頻道可在干擾允許時重用。",
        "min_device_distance": "隨機生成裝置時，任兩裝置必須維持的最低地圖座標距離。",
        "max_coord_attempts": "隨機放置裝置或 UAV 時允許的座標採樣嘗試上限。",
    },
    "TRAIN": {
        "device": "訓練網路與 agent tensor 使用的運算裝置；CPU 與 CUDA 選項對應不同運算硬體。",
        "max_steps": "每個訓練 episode 可執行的環境步數上限。",
        "episodes": "整場訓練可用的環境回合總數；HRL 依此預算分配固定階段回合。",
        "buffer_size": "每個 replay pool 可保存的經驗筆數；HRL 各 UAV 的 pool 各自計算。",
        "batch_size": "DQN 系列、MA-CQR 與 HRL 上層每次從 replay 抽取的筆數。",
        "seed": "訓練期間 NumPy、Python 與 Torch 使用的亂數種子。",
    },
    "EVAL": {
        "max_steps": "每個評估 episode 可執行的環境步數上限，與訓練步數上限分開設定。",
        "eval_runs": "每次評估執行的 episode 數量。",
        "seed": "評估所用的亂數種子，與訓練亂數種子分開設定。",
        "eval_periods": "每經過多少訓練 episode 執行一次評估。",
    },
    "FUNCTIONAL": {
        "random_device_coord": "開啟時由環境隨機放置裝置；關閉時使用固定裝置座標。",
        "random_UAV_coord": "開啟時由環境隨機放置 UAV；關閉時使用固定 UAV 座標。",
        "risk": "控制風險矩形與隨機受罰事件；關閉時風險區不參與環境步進。",
    },
    "FIXED": {
        "fixed_device_coord": "隨機裝置座標關閉時使用的 (x, y) 清單。",
        "fixed_UAV_coord": "隨機 UAV 座標關閉時使用的 (x, y) 清單。",
        "fixed_risk_region": "風險功能開啟時使用的矩形座標，每列依序為 xmin、ymin、xmax、ymax。",
    },
    "RECORD": {
        "reward_smoothing_window": "結果中的評估回報移動平均視窗長度；設為 1 時不做平滑。",
        "figure_size": "共用結果圖的寬與高，單位為英吋。",
    },
    "DQN": {
        "hidden_size": "Q 網路隱藏層的寬度。",
        "gamma": "計算未來回報時每步使用的折扣率；跨步經驗按持續步數累積折扣。",
        "tau": "每次 soft update 中，新 Q 網路權重佔目標網路更新的比例。",
        "lr": "Q 網路 Adam optimizer 的學習率。",
        "warmup_steps": "開始學習前 replay 的預填步數。",
        "start_eps": "epsilon-greedy 探索的起始隨機動作機率。",
        "min_eps": "epsilon 衰減後保留的最低隨機動作機率。",
        "eps_frames": "探索率從起始值線性衰減至最低值所經過的環境步數。",
    },
    "DQN_SERVICE": {
        "hidden_size": "Q 網路隱藏層的寬度。",
        "service_ratio": "移動後進入地圖邊長乘此比例的距離範圍時，環境強制服務並解除目標鎖定。",
        "gamma": "計算未來回報時每步使用的折扣率；跨步經驗按持續步數累積折扣。",
        "tau": "每次 soft update 中，新 Q 網路權重佔目標網路更新的比例。",
        "lr": "Q 網路 Adam optimizer 的學習率。",
        "warmup_steps": "開始學習前 replay 的預填步數。",
        "start_eps": "epsilon-greedy 探索的起始隨機動作機率。",
        "min_eps": "epsilon 衰減後保留的最低隨機動作機率。",
        "eps_frames": "探索率從起始值線性衰減至最低值所經過的環境步數。",
    },
    "HRL_UPPER": {
        "reward_mode": "AoI 模式使用 AoI 降低量回報（可另扣實際受罰事件次數）；均值模式取完整 Option 每步團隊環境回報的算術平均；平均回報 SMDP 使用完整 Option 的 R−rho×T 回報項。",
        "risk_penalty_weight": "僅在 AoI 降低量模式中，乘上每個 UAV 實際受罰事件數後從上層回報扣除；不改環境罰款或下層回報。",
        "average_reward_lr": "僅平均回報 SMDP 模式更新每個上層 agent 的 rho；它不是環境風險係數。",
        "hidden_size": "每架 UAV 獨立上層候選評分 Q 網路的隱藏層寬度。",
        "gamma": "前兩種上層回報模式依 Option 持續步數折扣後續價值；平均回報 SMDP 不使用此值。",
        "tau": "上層目標 Q 網路 soft update 使用的新權重比例。",
        "lr": "每架 UAV 的上層 Q 網路 Adam optimizer 學習率。",
        "warmup_steps": "每架 UAV 獨立的上層 replay 開始學習前，至少累積的完整經驗片段數。",
    },
    "HRL_LOWER": {
        "algorithm": "選擇下層更新方式：A2C 使用 on-policy rollout；SAC、DSAC 與 DSAC-T 使用 replay。",
        "warmup_mode": "下層 replay 預填可選隨機動作或直線示範；若 HRL.pretrain_batches=0，預填會強制使用隨機動作，即使選了直線示範；A2C 不使用此設定。",
        "warmup_steps": "每個下層 off-policy replay pool 開始學習前需累積的 transition 數；A2C 不使用此設定。",
        "a2c_steps": "A2C 每次收集的 on-policy rollout 環境步數；其他下層演算法不使用此設定。",
        "hidden_size": "各下層 actor 與 critic 網路隱藏層的寬度。",
        "actor_lr": "連續策略 actor Adam optimizer 的學習率。",
        "critic_lr": "下層價值 critic Adam optimizer 的學習率。",
        "gamma": "下層回報計算使用的每步折扣率。",
        "tau": "SAC／DSAC 系列 target critic soft update 使用的新權重比例。",
        "alpha": "SAC／DSAC 系列 actor loss 的固定熵係數；A2C 不使用。",
        "log_std_min": "高斯策略輸出 log 標準差時使用的下界。",
        "log_std_max": "高斯策略輸出 log 標準差時使用的上界。",
        "imitation_weight": "下層與聯合訓練 actor loss 中模仿項的起始權重；設為 0 不會關閉一次性示範初始化。",
        "imitation_decay_episodes": "聯合階段依已完成回合數線性衰減模仿權重所需的回合數；0 時起始權重保持固定。",
        "min_imitation_weight": "衰減完成後的模仿權重下限；衰減回合數為 0 時不使用。",
        "pretrain_batch_size": "一次性離線示範初始化每次更新抽取的樣本數；由 HRL 的初始化批數控制是否執行。",
        "std_min": "DSAC 分布式回報 critic 所使用的標準差下限；與 actor 的 log 標準差界限不同。",
        "td_bound": "DSAC 調整回報標準差梯度時使用的 TD 誤差界限。",
        "std_rate": "DSAC-T 指數移動平均更新回報標準差的比例。",
        "policy_delay": "僅 DSAC 與 DSAC-T 每完成指定次數 critic 更新後才更新 actor 與 target；A2C 與 SAC 每次 critic 更新都更新 actor。",
    },
    "HRL": {
        "upper_fraction": "從訓練總回合預算分配給上層固定訓練階段的比例；階段回合數由總數乘比例後向下取整。",
        "lower_fraction": "從訓練總回合預算分配給下層固定訓練階段的比例；階段回合數向下取整，剩餘回合用於聯合階段。",
        "pretrain_batches": "上層階段後執行的一次性離線直線導航示範初始化更新批數；0 會略過初始化，不會關閉訓練中的 imitation loss。",
        "navigation_cases": "建立一次性示範初始化資料時使用的位置與目標配對數。",
        "navigation_eval_seed": "固定導航驗證案例及驗證策略採樣所使用的亂數種子；與訓練及一般評估種子分開。",
        "service_ratio": "以地圖邊長乘此比例設定到達／服務距離門檻。",
        "progress_weight": "乘以下層每步朝目標縮短的正規化距離差。",
        "success_reward": "下層完成目標服務時加入的回報。",
        "step_penalty": "下層每個環境步扣除的時間成本；不改上層回報或環境評估回報。",
        "risk_weight": "下層風險事件回報的相對權重；實際扣除尺度另由風險懲罰參照值決定。",
        "risk_penalty_reference": "下層風險扣分的固定正規化分母；例如環境罰款 300、參照值 300 時扣除 1 倍 risk_weight。",
        "fragment_steps": "AoI 風險事件回報模式下，未完成目標時每段上層經驗最多累積的步數；其他兩種回報模式使用完整 Option。",
        "start_eps": "上層 epsilon-greedy 探索的起始隨機動作機率。",
        "min_eps": "上層探索率衰減後保留的最低隨機動作機率。",
        "eps_decisions": "上層探索率從起始值線性衰減至最低值所需的決策數。",
        "lower_batch_size": "每個下層 off-policy replay pool 每次抽樣的 transition 數；A2C 不使用 replay 批次。",
        "cpu_threads": "HRL 執行時 Torch 可使用的 CPU 執行緒數。",
        "convergence_window": "收斂診斷時計算穩定度所用的連續評估視窗長度。",
        "convergence_tolerance": "上層回報或下層到達率診斷中，相對範圍可視為穩定的容許值。",
        "convergence_patience": "診斷旗標要求連續達到穩定條件的視窗數；不控制階段切換。",
        "convergence_min_episodes": "上層或下層各自累積到此回合數後，該階段才具備被標記為收斂的資格；診斷指標在門檻前仍會記錄。",
        "reach_threshold": "下層導航收斂診斷採用的最低到達率；未達標會記錄診斷結果，但不阻止進入聯合階段。",
        "reach_cases": "每次下層導航能力診斷使用的固定起點與目標案例數。",
        "reach_steps": "每個導航診斷案例允許的最大移動步數。",
    },
    "HRL_PPO_UPPER": {
        "reward_mode": "AoI 模式使用 AoI 降低量回報（可另扣實際受罰事件次數）；均值模式取完整 Option 每步團隊環境回報的算術平均；平均回報 SMDP 使用完整 Option 的 R−rho×T 回報項。",
        "risk_penalty_weight": "僅 AoI 降低量模式且 FUNCTIONAL.risk 開啟時生效，須為非負值；乘上每架 UAV 實際受罰事件數後從上層回報扣除，不改環境罰款或下層回報。",
        "average_reward_lr": "僅平均回報 SMDP 模式更新每個上層 agent 的 rho；它不是環境風險係數。",
        "hidden_size": "上層目標選擇 Q 網路的隱藏層寬度。",
        "gamma": "前兩種上層回報模式依 Option 持續步數折扣後續價值；平均回報 SMDP 不使用此值。",
        "tau": "上層目標網路 soft update 中，新網路權重所占比例。",
        "lr": "上層 Q 網路 Adam optimizer 的學習率。",
        "warmup_steps": "每架 UAV 的上層 Replay 最低預填完整片段數；有效值至少為 TRAIN.batch_size，預填期間只收集經驗、不進行學習。",
        "start_eps": "上層 epsilon-greedy 探索的起始隨機動作機率。",
        "min_eps": "上層探索率衰減後保留的最低隨機動作機率。",
        "eps_decisions": "上層探索率由起始值線性衰減至最低值所需的決策數。",
    },
    "HRL_PPO_LOWER": {
        "hidden_size": "下層 actor 與 critic 網路的隱藏層寬度。",
        "algorithm": "選擇下層 PPO 或 A2PO 更新方式。",
        "actor_lr": "連續策略 actor Adam optimizer 的學習率。",
        "critic_lr": "下層價值 critic Adam optimizer 的學習率。",
        "gamma": "下層回報計算使用的每步折扣率。",
        "gae_lambda": "Generalized Advantage Estimation 的 lambda 係數，控制優勢估計的偏差與方差。",
        "clip_ratio": "PPO 更新時限制新舊策略機率比例的裁剪幅度。",
        "epochs": "同一批 rollout 資料重複進行 PPO 更新的回合數。",
        "minibatch_size": "每次 actor-critic 更新抽取的 rollout 樣本數。",
        "rollout_steps": "每次收集 on-policy 資料時的環境步數。",
        "entropy_coef": "乘在策略熵項上的係數；設為 0 可關閉熵正則。",
        "target_kl": "每輪 PPO 更新的 KL 停止門檻；0 表示停用此停止條件。",
        "log_std_min": "高斯策略 log 標準差允許的下界。",
        "log_std_max": "高斯策略 log 標準差允許的上界，且不可小於下界。",
        "pretrain_batches": "導航階段前的一次性示範初始化更新批數；0 會略過初始化。",
        "pretrain_batch_size": "每次示範初始化更新抽取的資料筆數；批數為 0 時不使用。",
        "navigation_cases": "建立示範初始化資料時使用的位置與目標配對數；批數為 0 時不使用。",
        "order": "決定 A2PO 依 greedy 或 random 順序逐一更新 agent；PPO 不使用此欄位。",
        "trace_clip": "A2PO 更新中，前面已更新 agent 的策略比率會影響後續 GAE 軌跡；此值是該比率的上限，PPO 不使用。",
    },
    "HRL_PPO": {
        "service_ratio": "以地圖邊長乘此比例設定 UAV 抵達目標時的服務距離門檻。",
        "navigation_fraction": "從 TRAIN.episodes 分配給導航預訓練階段的比例；乘積向下取整，剩餘回合用於聯合訓練。",
        "fragment_steps": "僅 AoI 降低量模式生效：未完成目標時每段上層經驗最多累積的步數；到達上限會建立片段但不解除目標鎖定，其他兩種回報模式使用完整 Option。",
        "progress_weight": "下層每步朝目標縮短正規化距離時所乘的回報權重。",
        "success_reward": "下層完成目標裝置服務時加入的回報。",
        "step_penalty": "下層每個環境步扣除的時間成本。",
        "risk_weight": "下層風險事件回報的相對權重；FUNCTIONAL.risk 關閉時不適用。",
        "risk_penalty_reference": "下層風險扣分的正規化參照值；FUNCTIONAL.risk 關閉時不適用。",
        "environment_reward_weight": "環境回報在 HRL-PPO 下層訓練回報中的乘數。",
        "cpu_threads": "HRL-PPO 執行時 Torch 可使用的 CPU 執行緒數。",
    },
    "MA_CQR": {
        "hidden_size": "各 agent 分位數 Q 網路隱藏層的寬度。",
        "N": "每個動作預測與學習的回報分位數數量。",
        "eta": "動作選擇採用的最低回報 CVaR 尾端質量比例；1 時以完整分布平均選擇。",
        "gamma": "集中式分位數 TD 目標使用的每步回報折扣率。",
        "tau": "目標分位數網路 soft update 使用的新權重比例。",
        "lr": "所有 agent 分位數網路共用的聯合 optimizer 學習率。",
        "huber_kappa": "Quantile Huber loss 在平方區與線性區之間切換的 TD 誤差界限。",
        "warmup_steps": "開始更新前共享 replay 需累積的環境步數。",
        "cpu_threads": "MA-CQR 訓練可使用的 Torch CPU 執行緒數。",
        "start_eps": "epsilon-greedy 探索的起始隨機動作機率。",
        "min_eps": "epsilon 衰減後保留的最低隨機動作機率。",
        "eps_frames": "探索率從起始值線性衰減至最低值所經過的環境步數。",
    },
}

OPTION_LABELS = {
    ("ENV", "communication_model"): {
        "snr": "SNR｜獨立鏈路",
        "sinr_hybrid": "SINR｜同頻增功率或分時擇優",
        "sinr_adaptive": "SINR｜固定功率自適應排程",
        "sinr_subchannels": "SINR｜子頻道配置",
    },
    ("ENV", "arrival_model"): {
        "always": "固定週期到達",
        "bernoulli": "Bernoulli 到達",
        "poisson": "Poisson 到達",
    },
    ("TRAIN", "device"): {
        "cpu": "CPU",
        "cuda": "CUDA GPU",
    },
    ("HRL_UPPER", "reward_mode"): {
        "aoi_risk_count": "AoI 降低量",
        "option_mean_environment": "Option 平均回報",
        "average_reward_smdp": "平均回報 SMDP",
    },
    ("HRL_PPO_UPPER", "reward_mode"): {
        "aoi_risk_count": "AoI 降低量",
        "option_mean_environment": "Option 平均回報",
        "average_reward_smdp": "平均回報 SMDP",
    },
    ("HRL_LOWER", "warmup_mode"): {
        "random": "隨機動作",
        "demonstration": "直線示範",
    },
    ("HRL_LOWER", "algorithm"): {
        "a2c": "A2C",
        "sac": "SAC",
        "dsac": "DSAC",
        "dsac_t": "DSAC-T",
    },
    ("HRL_PPO_LOWER", "algorithm"): {
        "ppo": "PPO",
        "a2po": "A2PO",
    },
    ("HRL_PPO_LOWER", "order"): {
        "greedy": "貪婪更新順序",
        "random": "隨機更新順序",
    },
}

_FIELD_ORDER = {
    "ENV": (
        "M", "U", "map_size", "max_mov", "altitude", "cell_distance",
        "channel_gain", "bandwidth", "packet_size", "noise_power",
        "communication_model", "max_transmit_power", "fixed_transmit_power", "subchannel_count",
        "energy_weight", "penalty", "prob_of_risk", "arrival_model",
        "arrival_interval", "arrival_probability", "arrival_rate",
        "min_device_distance",
        "max_coord_attempts",
    ),
    "TRAIN": ("device", "episodes", "max_steps", "seed", "buffer_size", "batch_size"),
    "EVAL": ("eval_periods", "eval_runs", "max_steps", "seed"),
    "FUNCTIONAL": ("risk", "random_device_coord", "random_UAV_coord"),
    "FIXED": ("fixed_device_coord", "fixed_UAV_coord", "fixed_risk_region"),
    "RECORD": ("reward_smoothing_window", "figure_size"),
    "DQN": (
        "hidden_size", "lr", "gamma", "tau", "warmup_steps",
        "start_eps", "min_eps", "eps_frames",
    ),
    "DQN_SERVICE": (
        "hidden_size", "service_ratio", "lr", "gamma", "tau",
        "warmup_steps", "start_eps", "min_eps", "eps_frames",
    ),
    "HRL_UPPER": (
        "reward_mode", "risk_penalty_weight", "average_reward_lr",
        "hidden_size", "lr", "gamma", "tau", "warmup_steps",
    ),
    "HRL_LOWER": (
        "algorithm", "warmup_mode", "warmup_steps", "a2c_steps",
        "hidden_size", "actor_lr", "critic_lr", "gamma", "tau", "alpha",
        "log_std_min", "log_std_max", "imitation_weight",
        "imitation_decay_episodes", "min_imitation_weight",
        "pretrain_batch_size", "std_min", "td_bound", "std_rate", "policy_delay",
    ),
    "HRL": (
        "upper_fraction", "lower_fraction", "pretrain_batches", "navigation_cases",
        "navigation_eval_seed", "service_ratio", "progress_weight", "success_reward",
        "step_penalty", "risk_weight", "risk_penalty_reference", "fragment_steps",
        "start_eps", "min_eps", "eps_decisions", "lower_batch_size", "cpu_threads",
        "convergence_window", "convergence_tolerance", "convergence_patience",
        "convergence_min_episodes", "reach_threshold", "reach_cases", "reach_steps",
    ),
    "HRL_PPO_UPPER": (
        "reward_mode", "risk_penalty_weight", "average_reward_lr",
        "hidden_size", "lr", "gamma", "tau", "warmup_steps",
        "start_eps", "min_eps", "eps_decisions",
    ),
    "HRL_PPO_LOWER": (
        "algorithm", "hidden_size", "actor_lr", "critic_lr", "gamma", "gae_lambda",
        "clip_ratio", "epochs", "minibatch_size", "rollout_steps", "entropy_coef",
        "target_kl", "log_std_min", "log_std_max", "pretrain_batches",
        "pretrain_batch_size", "navigation_cases", "order", "trace_clip",
    ),
    "HRL_PPO": (
        "navigation_fraction", "service_ratio", "fragment_steps", "progress_weight",
        "success_reward", "step_penalty", "risk_weight", "risk_penalty_reference",
        "environment_reward_weight", "cpu_threads",
    ),
    "MA_CQR": (
        "hidden_size", "N", "eta", "lr", "huber_kappa", "gamma", "tau",
        "warmup_steps", "cpu_threads", "start_eps", "min_eps", "eps_frames",
    ),
}


def group_label(group: str) -> str:
    """Return a Traditional Chinese title for a setting group."""
    label = _GROUP_LABELS.get(group)
    if label is not None:
        return label
    return f"其他設定（{group.replace('_', ' ')}）"


def group_help(group: str) -> str:
    """Return a concise explanation of a setting group."""
    description = _GROUP_HELP.get(group)
    if description is not None:
        return description
    return f"此群組「{group}」的欄位保留原始設定路徑，方便搜尋與辨識。"


def field_label(path: tuple[str, ...]) -> str:
    """Return a short Traditional Chinese label for a full setting path."""
    if len(path) == 2:
        label = _FIELD_LABELS.get(path[0], {}).get(path[1])
        if label is not None:
            return label
    field = path[-1] if path else "設定"
    return f"其他欄位：{field.replace('_', ' ')}"


def field_help(path: tuple[str, ...]) -> str:
    """Return a concise explanation for a full setting path."""
    if len(path) == 2:
        description = _FIELD_HELP.get(path[0], {}).get(path[1])
        if description is not None:
            return description
    full_path = ".".join(path) if path else "（空路徑）"
    return f"目前沒有此欄位的專屬說明；請依原始設定路徑「{full_path}」辨識。"


def ordered_fields(group: str, fields: Iterable[str]) -> list[str]:
    """Order known settings by purpose, then unknown fields by name."""
    unique_fields = set(fields)
    order = _FIELD_ORDER.get(group, ())
    rank = {field: index for index, field in enumerate(order)}
    known = sorted((field for field in unique_fields if field in rank), key=rank.__getitem__)
    unknown = sorted(
        (field for field in unique_fields if field not in rank),
        key=lambda field: (field.casefold(), field),
    )
    return known + unknown
