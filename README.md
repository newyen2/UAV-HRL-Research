# UAV-HRL-Research

本專案支援 `DQN`、`DQN_service`、`HRL`、`HRL_PPO` 與 `MA-CQR` 五種 UAV 訓練方法。可用圖形介面建立佇列，也可用 `train.py` 從命令列訓練。

## 開始使用

建立 Python 環境後，從專案根目錄執行下列命令啟動 GUI。[環境說明](docs/ENVIRONMENT.md)

```powershell
.\.venv\Scripts\python.exe -m gui
```

GUI 的「實驗」頁可編輯設定、安排佇列及監看進度。「回放」、「結果」及「散布圖」頁可查看已保存的實驗。[實驗指南](docs/EXPERIMENTS.md)

下列命令透過 `train.py` 在 CPU 上執行 DQN。

```powershell
.\.venv\Scripts\python.exe train.py -model DQN --TRAIN_SETTING.device cpu --TRAIN_SETTING.episodes 100 --DQN_SETTING.lr 0.00005 --output-dir "Outputs/Experiments/dqn-run-01"
```

完整選項以目前程式輸出的 `train.py --help` 為準。常用格式與單場 JSON worker、軌跡回放指令見[命令列指南](docs/CLI.md)。

## 實驗設定與輸出

資料到達預設為每個時槽固定到達。可改用固定週期、Bernoulli 或 Poisson 模式。通訊預設使用 SNR，也可選擇三種 SINR 排程模式。GUI 與命令列都能設定這些選項。

結果預設寫入 `Outputs/Experiments/`。GUI 會為每場建立獨立資料夾。直接使用 CLI 時，結果寫入輸出根目錄下的方法資料夾。若要保留多次 CLI 執行結果，請為每次執行指定不同的 `--output-dir`。

## 操作手冊

- [環境安裝與檢查](docs/ENVIRONMENT.md)
- [GUI 佇列、範本與回放](docs/EXPERIMENTS.md)
- [CLI 訓練與設定](docs/CLI.md)
- [輸出檔案與指標解讀](docs/OUTPUT_FORMAT.md)
- [文件索引](docs/README.md)
