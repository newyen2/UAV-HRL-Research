# 命令列操作

以下命令從專案根目錄執行。範例使用專案虛擬環境的 Python。先依[環境說明](ENVIRONMENT.md)安裝依賴。

## 訓練

以 `train.py` 指定方法。可用方法為 `DQN`、`DQN_service`、`HRL`、`HRL_PPO` 與 `MA-CQR`。

```powershell
.\.venv\Scripts\python.exe train.py -model DQN --TRAIN_SETTING.device cpu --TRAIN_SETTING.episodes 100 --DQN_SETTING.lr 0.00005 --output-dir "Outputs/Experiments/dqn-run-01"
```

方法名稱須與上列大小寫相同。設定格式為 `--SETTING_GROUP.field value`，例如 `--TRAIN_SETTING.device cpu`。`--output-dir` 與 `--RECORD_SETTING.PATH` 都可指定輸出根目錄。設定只作用於該次程序，未覆寫的欄位沿用程式預設值。

目前的完整方法與設定欄位以此命令輸出的說明為準。

```powershell
.\.venv\Scripts\python.exe train.py --help
```

常見方法設定範例均指定 CPU，沒有 CUDA 裝置也能執行。

```powershell
.\.venv\Scripts\python.exe train.py -model DQN_service --TRAIN_SETTING.device cpu --DQN_SERVICE_SETTING.service_ratio 0.1
.\.venv\Scripts\python.exe train.py -model HRL --TRAIN_SETTING.device cpu --HRL_LOWER_SETTING.algorithm sac
.\.venv\Scripts\python.exe train.py -model HRL_PPO --TRAIN_SETTING.device cpu --HRL_PPO_LOWER_SETTING.algorithm ppo
.\.venv\Scripts\python.exe train.py -model MA-CQR --TRAIN_SETTING.device cpu --MA_CQR_SETTING.eta 0.15
```

共用設定可用於所有方法，方法專屬設定只能用於對應方法。程式會在載入訓練方法前檢查欄位、型別、範圍與適用性。不適用的設定會回報錯誤。布林值使用 JSON 格式的 `true` 或 `false`。陣列使用 JSON，PowerShell 中可用單引號包住，例如 `'[[0,0],[10,10]]'`。

直接執行 `DQN.py`、`HRL.py` 等方法檔會使用預設設定，不解析命令列覆寫。需要調整設定時請使用 `train.py`。

## 資料到達與通訊

用 `ENV_SETTING.arrival_model` 選擇資料到達模式。

| 值 | 設定方式 |
| --- | --- |
| `always` | 固定週期到達，由 `ENV_SETTING.arrival_interval` 設定時槽間隔。預設間隔為 1。 |
| `bernoulli` | 每個裝置每步依 `ENV_SETTING.arrival_probability` 抽樣。 |
| `poisson` | 每個裝置每步使用 `ENV_SETTING.arrival_rate` 作為 Poisson 到達率。 |

通訊模式由 `ENV_SETTING.communication_model` 選擇。預設 `snr` 使用獨立鏈路 SNR。`sinr_hybrid` 依同頻或分時排程擇優，`sinr_adaptive` 使用固定發射功率自適應排程，`sinr_subchannels` 使用子頻道配置。SINR 可搭配 `ENV_SETTING.max_transmit_power`、`ENV_SETTING.fixed_transmit_power` 與 `ENV_SETTING.subchannel_count` 調整。

例如使用 CPU、Poisson 到達並選擇 SINR 子頻道模式。

```powershell
.\.venv\Scripts\python.exe train.py -model DQN --TRAIN_SETTING.device cpu --ENV_SETTING.arrival_model poisson --ENV_SETTING.arrival_rate 0.2 --ENV_SETTING.communication_model sinr_subchannels
```

## 執行單場 JSON 設定

`gui.experiment_worker` 需要單場設定檔與該場的輸出資料夾。

```powershell
.\.venv\Scripts\python.exe -m gui.experiment_worker --config docs/experiment.example.json --output-dir Outputs/Experiments/dqn-cli
```

單場設定包含 `name`、`method` 與 `settings`。`settings` 按群組列出要覆寫的欄位，未提供的欄位使用目前預設值。worker 會驗證整份設定，再開始訓練。上例是格式範例，會執行訓練並建立輸出資料。結果在 `Outputs/Experiments/dqn-cli/results/` 下，狀態記錄在 `Outputs/Experiments/dqn-cli/status.json`。

此 worker 接受單場設定，不接受包含多場 `experiments` 清單的 GUI 佇列 JSON。GUI 佇列匯入方式見[實驗指南](EXPERIMENTS.md#json-匯入與設定範本)。

## 回放既有軌跡

用 `gui.trace_tool` 開啟前面範例產生的 `episode.parquet` 與 `step.parquet`。

```powershell
.\.venv\Scripts\python.exe -m gui.trace_tool --episode_parquet "Outputs/Experiments/dqn-run-01/DQN/Stored_Datas/episode.parquet" --step_parquet "Outputs/Experiments/dqn-run-01/DQN/Stored_Datas/step.parquet" --episode 1
```

兩個 parquet 參數必須一起提供。GUI 也可在「結果」頁選取結果後按「觀看所選結果軌跡」，或在「回放」頁載入這兩個檔案。資料夾結構及欄位說明見[輸出格式](OUTPUT_FORMAT.md)。
