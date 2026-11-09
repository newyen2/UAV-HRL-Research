"""歷史 training.log 的唯讀快照。"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from runtime import fileIO as saving
from gui.gui_style import MONO_FONT, configure_gui_style


_MAX_TAIL_BYTES = 256 * 1024
_MAX_TAIL_LINES = 1000


def read_training_log_tail(path):
    """讀取日誌尾端的有限快照，回傳 (文字, 是否截斷)。"""
    return saving.read_log_tail(
        path, max_bytes=_MAX_TAIL_BYTES, max_lines=_MAX_TAIL_LINES)


def show_training_log(parent, path):
    """先讀取快照，再開啟可複製的唯讀視窗；讀取錯誤交由呼叫端處理。"""
    snapshot, truncated = read_training_log_tail(path)
    source_path = str(Path(path).resolve())
    configure_gui_style(parent._root())

    window = tk.Toplevel(parent)
    window.title("歷史訓練日誌快照")
    window.geometry("1000x650")
    window.columnconfigure(0, weight=1)
    window.rowconfigure(2, weight=1)

    ttk.Label(
        window,
        text=f"來源：{source_path}",
        anchor="w",
        wraplength=980,
        justify="left",
        padding=(8, 8, 8, 4),
    ).grid(row=0, column=0, sticky="ew")

    if truncated:
        snapshot_note = "唯讀快照：尾端最多 256 KiB、最多 1000 行；內容已截斷。"
    else:
        snapshot_note = "唯讀快照：尾端最多 256 KiB、最多 1000 行；內容未截斷。"
    ttk.Label(window, text=snapshot_note, anchor="w", padding=(8, 2)).grid(
        row=1, column=0, sticky="ew")

    text_frame = ttk.Frame(window, padding=(8, 4))
    text_frame.grid(row=2, column=0, sticky="nsew")
    text_frame.columnconfigure(0, weight=1)
    text_frame.rowconfigure(0, weight=1)
    log_text = tk.Text(text_frame, wrap="none", state="normal", font=MONO_FONT)
    vertical_scroll = ttk.Scrollbar(
        text_frame, orient="vertical", command=log_text.yview)
    horizontal_scroll = ttk.Scrollbar(
        text_frame, orient="horizontal", command=log_text.xview)
    log_text.configure(
        yscrollcommand=vertical_scroll.set,
        xscrollcommand=horizontal_scroll.set,
    )
    log_text.grid(row=0, column=0, sticky="nsew")
    vertical_scroll.grid(row=0, column=1, sticky="ns")
    horizontal_scroll.grid(row=1, column=0, sticky="ew")
    log_text.insert("1.0", snapshot)
    log_text.configure(state="disabled")

    status_var = tk.StringVar(master=window)
    if snapshot == "":
        status_var.set("訓練日誌為空；複製時會複製空內容。")
    else:
        status_var.set("此視窗顯示開啟時的快照，不會自動更新。")

    def copy_snapshot():
        try:
            window.clipboard_clear()
            window.clipboard_append(snapshot)
        except tk.TclError as error:
            status_var.set(f"無法複製快照：{error}")
            return
        if snapshot == "":
            status_var.set("已複製空白快照內容。")
        else:
            status_var.set("已複製目前顯示的快照內容。")

    controls = ttk.Frame(window, padding=(8, 4, 8, 8))
    controls.grid(row=3, column=0, sticky="ew")
    controls.columnconfigure(0, weight=1)
    ttk.Label(controls, textvariable=status_var, anchor="w").grid(
        row=0, column=0, sticky="ew")
    copy_button = ttk.Button(controls, text="複製顯示內容", command=copy_snapshot)
    copy_button.grid(row=0, column=1, padx=(8, 4))
    close_button = ttk.Button(controls, text="關閉", command=window.destroy)
    close_button.grid(row=0, column=2)

    return window
