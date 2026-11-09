# Python 環境

以下指令從專案根目錄在 Windows PowerShell 執行。專案依賴列於 [`requirements.txt`](../requirements.txt)，GUI 需要 Python 的 Tkinter 元件。

## 建立環境

使用 Python 3.11 建立專案虛擬環境。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

沒有可用 CUDA 裝置時，安裝專案依賴並檢查套件狀態。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

若使用 CUDA 12.8，先安裝 PyTorch 2.11.0 對應的官方 wheel，再安裝其餘專案依賴。

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

沒有可用 CUDA 裝置時，訓練設定請選 `cpu`。

## 確認環境

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m tkinter
```

第一行顯示 PyTorch 版本與 CUDA 可用狀態。第二行會開啟 Tkinter 測試視窗，關閉視窗即可完成檢查。

GUI 和命令列訓練都從專案根目錄使用同一個直譯器。

```powershell
.\.venv\Scripts\python.exe -m gui
.\.venv\Scripts\python.exe train.py --help
```

不要在專案外的工作目錄執行相對路徑命令。GUI 啟動後，訓練工作會沿用啟動 GUI 的 Python 環境。
