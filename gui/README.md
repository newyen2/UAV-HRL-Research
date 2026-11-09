# 實驗 GUI

在專案根目錄執行下列命令啟動介面。

```powershell
.\.venv\Scripts\python.exe -m gui
```

介面提供實驗設定與佇列、軌跡回放、結果瀏覽及散布圖。[實驗操作指南](../docs/EXPERIMENTS.md) · [命令列指南](../docs/CLI.md) · [環境說明](../docs/ENVIRONMENT.md)

## 設定範本

「設定範本」視窗可將選取的表單欄位儲存為 JSON。首次儲存會建立 `Outputs/Presets/`。若要使用既有範本，將範本 JSON 放入該目錄，再開啟視窗並按「套用所選範本」。未列出的欄位會回到目標方法的預設值。

## 匯入與結果

「載入 JSON…」支援[佇列範例](experiments.example.json)和[單場設定範例](../docs/experiment.example.json)所示的兩種格式。結果頁可讀取已保存的資料，並將目前支援的方法設定帶回表單供編輯。
