# 實驗輸出與指標

結果預設寫入 `Outputs/Experiments/`。GUI 佇列會為每場建立獨立資料夾。直接使用 CLI 時，方法結果寫入所選輸出根目錄下。

## GUI 場次的資料夾

```text
Outputs/Experiments/<場次資料夾>/
├── config.json
├── status.json
├── training.log
├── progress.jsonl
└── results/
    └── <方法>/
        ├── Result_Datas/
        ├── Results/
        └── Stored_Datas/
```

## CLI 結果的資料夾

```text
<輸出根目錄>/<方法>/
├── Result_Datas/
├── Results/
└── Stored_Datas/
```

`<方法>` 為 `DQN`、`DQN_service`、`HRL`、`HRL_PPO` 或 `MA-CQR`。CLI 在相同輸出根目錄與方法下重跑時會寫入同一組檔名。要保留不同執行結果，請為每次執行指定不同的 `--output-dir`。GUI 會自行建立不同場次目錄。

## 指標 CSV 與圖表

| 檔案 | 內容 |
| --- | --- |
| `Result_Datas/Result_<方法>.csv` | 每個訓練回合的記錄。`reward` 是環境累積回報，`mean_aoi` 是該回合的平均 AoI。HRL 的上層學習回報另存於 `learning_reward`。 |
| `Result_Datas/Reward.csv` | 訓練回合的環境回報，排除暖身、評估等非訓練階段。 |
| `Result_Datas/Eval_Episodes_<方法>.csv` | 每個評估回合的原始指標。 |
| `Result_Datas/AoI.csv` | 各訓練評估點跨評估回合平均的 `eval_aoi`。 |
| `Result_Datas/Power.csv` | 跨評估回合平均的加權傳輸成本及累積功率。 |
| `Result_Datas/Violations.csv` | 跨評估回合平均的風險區步數與實際風險懲罰。 |
| `Result_Datas/eval_reward.csv` | 由評估 AoI、加權傳輸成本及實際風險懲罰組成的評估回報。 |
| `Results/*.png` | 由同名 CSV 繪製的指標圖表。 |

非評估回合的 `eval_*` 欄位可能為空。`eval_reward` 的計算為 `-(eval_aoi × eval_max_steps + eval_weighted_power + eval_violation_penalty)`。`eval_reward.png` 顯示移動平均，`eval_reward.csv` 同時保存原始值與平滑值。訓練 `Reward.png` 反映環境回報，不能與評估 `eval_reward.png` 混為一談。

`eval_aoi` 是評估期間對裝置及時槽平均後的 AoI。`eval_weighted_power` 是環境回報使用的加權傳輸成本。`eval_violation_count` 計算評估步驟中位於風險區的 UAV 數量累計後再跨評估回合平均。`eval_violation_penalty` 則記錄實際發生的風險懲罰。

結果頁和散布圖可讀取這些已保存的數值。更改目前訓練設定不會重算歷史結果。

## 軌跡與模型

`Stored_Datas/episode.parquet` 保存每回合的初始位置與場景資訊。`Stored_Datas/step.parquet` 保存各環境步的位置、動作、AoI 及回報。回放使用兩個檔案成對載入。

軌跡中的 `device_AOI` 表示 UAV 已收到資料的 AoI，`source_AOI` 表示裝置目前資料的 AoI。使用 GUI 回放頁可切換兩種檢視。含 SINR 通訊紀錄的軌跡也可能包含 `communication_model`、`communication_scheme`、送達狀態及傳輸功率等欄位。

`Stored_Datas/trace_schema.json` 說明軌跡欄位。`Stored_Datas/model.pt` 保存該方法提供的模型狀態與設定快照，實際欄位依方法而異。GUI 和 CLI 訓練入口會建立新訓練，沒有載入模型以接續訓練的選項。

GUI 場次的 `config.json` 保存該場設定，`status.json` 記錄狀態，`training.log` 保存訓練輸出。失敗或停止的場次可能沒有完整 CSV、圖表或模型檔，請先查看狀態與日誌。
