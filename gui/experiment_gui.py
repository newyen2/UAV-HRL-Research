"""Traditional Chinese Tkinter front end for queued RL experiments.

The GUI deliberately imports the experiment backend only when the window is
created.  Training modules are never imported here.
"""

from __future__ import annotations

import copy
from bisect import bisect_left, bisect_right
import json
import math
import os
import queue
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any
from gui.gui_format import format_number, format_edit_value
from gui.gui_style import HEADING_FONT, SMALL_FONT, SMALL_BOLD_FONT, MONO_FONT, configure_gui_style
from gui.experiment_settings import SettingsEditor
from gui.setting_labels import field_label
from gui.experiment_replay import ReplayPanel
from gui.experiment_results import ResultsPanel, result_config_path
from gui.experiment_scatter import ScatterPanel
from runtime import fileIO as saving


METHOD_LABELS = {
    "DQN": "DQN",
    "DQN_service": "DQN（服務導向）",
    "HRL": "HRL（各 UAV 獨立候選評分）",
    "HRL_PPO": "HRL-PPO（上層 DQN＋下層 PPO／A2PO）",
    "MA-CQR": "MA-CQR（合作式分位數）",
}
_COMMON_SETTING_GROUPS = {"ENV", "TRAIN", "EVAL", "FUNCTIONAL", "FIXED", "RECORD"}
MAX_LOG_LINES = 1500
MAX_LOG_CHARS = 256 * 1024
MAX_LOG_LINE_CHARS = 4096
MAX_LOG_HIGHLIGHTS = 2000
MAX_CURVE_POINTS = 300


# 載入實驗排程後端，缺少檔案時回報原因。
def _load_core():
    """Load the experiment API without importing any trainer in this module."""
    try:
        from gui import experiment_core
    except ImportError as exc:
        raise RuntimeError(
            "找不到 gui/experiment_core.py。請確認實驗管理後端檔案存在。"
        ) from exc
    return experiment_core


# 將執行秒數格式化為時、分、秒。
def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# 進度計數只接受有限的整數；無效輸出不能中斷 GUI 事件處理。
def _progress_counter(value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value) or value < minimum or int(value) != value:
            return None
    except (ValueError, OverflowError):
        return None
    return int(value)


class ExperimentGUI:
    # 初始化實驗佇列、表單與回放介面。
    def __init__(self, root: tk.Tk):
        self.root = root
        configure_gui_style(self.root)
        self.core = _load_core()
        from gui.preset_config import PRESET_DIRECTORY

        self.runner = self.core.QueueRunner(Path(__file__).resolve().parent.parent / "Outputs" / "Experiments")
        self.queue_items: list[dict[str, Any]] = []
        self.queue_iid_rows: dict[str, dict[str, Any]] = {}
        self.queue_iid_display: dict[str, tuple[str, tuple[str, str]]] = {}
        self._refreshing_queue = False
        self.run_row_indices: list[int] = []
        self.session_row_indices: list[int] = []
        self.deferred_run_indices: list[int] = []
        self.queue_active = False
        self.queue_stopping = False
        self.selected_edit_index: int | None = None
        self.preset_directory = PRESET_DIRECTORY
        self.preset_dialog = None
        self.closing = False
        self.run_started_at: float | None = None
        self.experiment_started_at: float | None = None
        self.active_output_dir: str | None = None
        self.active_episode_count = 0
        self.active_max_steps = 0
        self.curves = {"reward": [], "aoi": []}
        self.log_lines: list[str] = []
        # Python offsets map to Tk indexes through bounded line starts, so index
        # conversion only walks the current line instead of the whole Text widget.
        self._log_line_starts = [0]
        self._log_content_length = 0
        self._log_search_match_count = 0
        self._log_search_highlights: list[tuple[int, int]] = []
        self._log_search_match_ranges: list[tuple[int, int]] = []
        self._log_search_match_ranges_valid = False
        self._log_search_current_offset: int | None = None
        self._log_search_last_query = ""
        self._log_search_has_current = False

        self.root.title("強化學習實驗與軌跡回放")
        self.root.geometry("1180x820")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._select_method()
        self._poll_after_id = None
        self.root.bind("<Destroy>", self._on_root_destroy, add="+")
        self._poll_events()

    # 建立設定頁、佇列、進度、曲線與日誌控制元件。
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        ttk.Style(self.root).configure("Main.TNotebook.Tab", padding=(18, 7))
        self.main_notebook = ttk.Notebook(self.root, style="Main.TNotebook")
        self.main_notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.experiment_tab = ttk.Frame(self.main_notebook)
        self.replay_tab = ttk.Frame(self.main_notebook)
        self.results_tab = ttk.Frame(self.main_notebook)
        self.scatter_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.experiment_tab, text="實驗")
        self.main_notebook.add(self.replay_tab, text="回放")
        self.main_notebook.add(self.results_tab, text="結果")
        self.main_notebook.add(self.scatter_tab, text="散布圖")
        self.experiment_tab.columnconfigure(0, weight=1)
        self.experiment_tab.rowconfigure(1, weight=1)
        self.replay_panel = ReplayPanel(self.replay_tab)
        self.replay_panel.pack(fill="both", expand=True)
        self.results_panel = ResultsPanel(
            self.results_tab, self.runner.output_root, on_replay=self._view_result_trace,
            on_load_settings=self._load_result_settings, on_renamed=self._result_renamed)
        self.results_panel.pack(fill="both", expand=True)
        self.scatter_panel = ScatterPanel(
            self.scatter_tab, self.runner.output_root, on_renamed=self._result_renamed)
        self.scatter_panel.pack(fill="both", expand=True)
        self.main_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        header = ttk.Frame(self.experiment_tab, padding=(12, 10, 12, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="輸出資料夾：").grid(row=0, column=0, sticky="w")
        self.output_var = tk.StringVar(value=str(self.runner.output_root))
        self.output_entry = ttk.Entry(header, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self.choose_output_button = ttk.Button(header, text="選擇…", command=self._choose_output)
        self.choose_output_button.grid(row=0, column=2, padx=(0, 5))
        self.open_output_button = ttk.Button(header, text="開啟輸出資料夾", command=self._open_output)
        self.open_output_button.grid(row=0, column=3)

        main = ttk.Panedwindow(self.experiment_tab, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=5)
        main.add(right, weight=6)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        # 小視窗時讓佇列分攤較多壓縮，保留新增預估列後的監看區高度。
        right.rowconfigure(0, weight=4, minsize=190)
        right.rowconfigure(1, weight=2, minsize=130)
        right.rowconfigure(2, weight=2, minsize=180)

        form_head = ttk.Frame(left)
        form_head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(form_head, text="實驗設定", font=HEADING_FONT).pack(side="left")
        self.method_var = tk.StringVar(value="DQN")
        self.method_box = ttk.Combobox(form_head, textvariable=self.method_var,
                                       values=list(METHOD_LABELS), state="readonly", width=17)
        self.method_box.pack(side="right")
        self.method_box.bind("<<ComboboxSelected>>", lambda _e: self._select_method())

        self.settings_editor = SettingsEditor(left, self.core)
        self.settings_editor.grid(row=1, column=0, sticky="nsew")
        # 範本與方法切換共用同一份欄位字典；editor 重建時只清空內容。
        self.field_vars = self.settings_editor.field_vars
        self.name_var = tk.StringVar()
        name_row = ttk.Frame(left)
        name_row.grid(row=2, column=0, sticky="ew", pady=(7, 3))
        name_row.columnconfigure(1, weight=1)
        ttk.Label(name_row, text="實驗名稱：").grid(row=0, column=0, sticky="w")
        ttk.Entry(name_row, textvariable=self.name_var).grid(row=0, column=1, sticky="ew")
        form_submit = ttk.Frame(left)
        form_submit.grid(row=3, column=0, sticky="ew", pady=(3, 0))
        self.form_action = ttk.Button(
            form_submit, text="加入佇列", command=self._add_to_queue,
            style="Accent.TButton")
        self.form_action.pack(side="left", fill="x", expand=True)
        self.cancel_edit_button = ttk.Button(
            form_submit, text="取消編輯", command=self._cancel_edit, state="disabled")
        self.cancel_edit_button.pack(side="left", padx=(5, 0))
        form_tools = ttk.Frame(left)
        form_tools.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.validate_button = ttk.Button(form_tools, text="檢查設定", command=self._validate_form)
        self.validate_button.pack(side="left", fill="x", expand=True)
        self.reset_button = ttk.Button(form_tools, text="回復預設", command=self._reset_form)
        self.reset_button.pack(side="left", fill="x", expand=True, padx=4)
        self.advanced_tools_var = tk.BooleanVar(value=False)
        self.advanced_tools_toggle = ttk.Checkbutton(
            left, text="進階工具（批次、範本、匯出）",
            variable=self.advanced_tools_var, command=self._toggle_advanced_tools)
        self.advanced_tools_toggle.grid(row=5, column=0, sticky="w", pady=(5, 0))
        self.advanced_tools = ttk.Frame(left)
        self.advanced_tools.columnconfigure((0, 1), weight=1)
        self.sweep_button = ttk.Button(
            self.advanced_tools, text="參數批次…", command=self._add_parameter_sweep)
        self.sweep_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.grid_button = ttk.Button(
            self.advanced_tools, text="多欄組合…", command=self._add_parameter_grid)
        self.grid_button.grid(row=0, column=1, sticky="ew")
        self.export_form_button = ttk.Button(
            self.advanced_tools, text="匯出表單 JSON…", command=self._export_form_settings)
        self.export_form_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        self.preset_button = ttk.Button(
            self.advanced_tools, text="設定範本…", command=self._open_preset_manager)
        self.preset_button.grid(row=1, column=1, sticky="ew", pady=(4, 0))

        queue_frame = ttk.LabelFrame(right, text="實驗佇列", padding=8)
        queue_frame.grid(row=0, column=0, sticky="nsew")
        queue_frame.rowconfigure(1, weight=1)
        queue_frame.columnconfigure(0, weight=1)
        self.queue_count_var = tk.StringVar(value="等待 0｜排程 0｜執行 0｜完成 0｜失敗 0｜取消 0")
        ttk.Label(queue_frame, textvariable=self.queue_count_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.queue_tree = ttk.Treeview(
            queue_frame, columns=("method", "status"), show="tree headings",
            selectmode="browse", height=3)
        self.queue_tree.heading("#0", text="# / 實驗名稱")
        self.queue_tree.heading("method", text="方法")
        self.queue_tree.heading("status", text="狀態")
        self.queue_tree.column("#0", width=200, stretch=True, minwidth=100)
        self.queue_tree.column("method", width=105, stretch=False)
        self.queue_tree.column("status", width=100, stretch=False)
        for tag, background, foreground in (
            ("queue_running", "#e8f0fe", "#174ea6"),
            ("queue_completed", "#e8f5e9", "#25613a"),
            ("queue_failed", "#fce8e6", "#9c2b23"),
            ("queue_cancelled", "#f1f3f4", "#5f6368"),
        ):
            self.queue_tree.tag_configure(
                tag, background=background, foreground=foreground)
        self.queue_tree.grid(row=1, column=0, sticky="nsew")
        qscroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.queue_tree.yview)
        qscroll.grid(row=1, column=1, sticky="ns")
        self.queue_tree.configure(yscrollcommand=qscroll.set)
        self.queue_tree.bind("<<TreeviewSelect>>", self._on_queue_selection_changed)
        # 僅在佇列取得焦點時使用，輸入欄不會誤觸編輯或移除。
        for sequence in ("<Return>", "<Delete>", "<Control-d>", "<Control-D>",
                         "<Control-c>", "<Control-C>"):
            self.queue_tree.bind(sequence, self._on_queue_key)
        controls = ttk.Frame(queue_frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        self.mutation_buttons = []
        edit_row = ttk.Frame(controls)
        edit_row.pack(fill="x")
        data_row = ttk.Frame(controls)
        data_row.pack(fill="x", pady=(3, 0))
        edit_actions = (
            ("複製", self._duplicate_selected),
            ("編輯", self._edit_selected),
            ("移除", self._remove_selected),
            ("↑", lambda: self._move_selected(-1)),
            ("↓", lambda: self._move_selected(1)),
            ("移除已結束", self._remove_finished),
        )
        for index, (text, command) in enumerate(edit_actions):
            if index < 3:
                parent = edit_row
            else:
                parent = data_row
            button = ttk.Button(parent, text=text, command=command)
            if text in ("↑", "↓"):
                button.configure(width=3)
            elif text == "移除已結束":
                button.configure(width=10)
            else:
                button.configure(width=6)
            button.pack(side="left", padx=(0, 4))
            self.mutation_buttons.append(button)
        self.save_queue_button = ttk.Button(
            edit_row, text="儲存佇列…", command=self._save_queue)
        self.save_queue_button.pack(side="right")
        self.load_queue_button = ttk.Button(
            edit_row, text="載入 JSON…", command=self._load_queue)
        self.load_queue_button.pack(side="right", padx=(0, 4))
        self.open_selected_output_button = ttk.Button(
            data_row, text="所選實驗資料夾", command=self._open_selected_output,
            state="disabled")
        self.open_selected_output_button.pack(side="right", padx=(4, 0))
        self.view_selected_results_button = ttk.Button(
            data_row, text="所選實驗結果", command=self._view_selected_results,
            state="disabled")
        self.view_selected_results_button.pack(side="right", padx=(4, 0))

        live = ttk.LabelFrame(right, text="執行狀態", padding=5)
        live.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        live.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="待命")
        self.stage_var = tk.StringVar(value="階段：尚未開始")
        self.progress_var = tk.StringVar(value="尚未開始")
        self.elapsed_var = tk.StringVar(value="經過時間：00:00")
        self.eta_var = tk.StringVar(value="本場預估剩餘：尚未開始")
        ttk.Label(live, textvariable=self.status_var, font=SMALL_BOLD_FONT).grid(
            row=0, column=0, sticky="w")
        self.stage_label = ttk.Label(
            live, textvariable=self.stage_var, style="Status.TLabel", wraplength=240)
        self.stage_label.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(live, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        self.progress_label = ttk.Label(live, textvariable=self.progress_var, wraplength=500)
        self.progress_label.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        ttk.Label(live, textvariable=self.elapsed_var, style="Muted.TLabel").grid(
            row=3, column=0, sticky="w")
        self.eta_label = ttk.Label(live, textvariable=self.eta_var, wraplength=500)
        self.eta_label.grid(row=3, column=1, columnspan=2, sticky="ew")

        def resize_live_labels(event):
            width = max(140, event.width - 20)
            side_width = max(120, int(event.width * 0.66) - 12)
            self.stage_label.configure(wraplength=side_width)
            self.progress_label.configure(wraplength=width)
            self.eta_label.configure(wraplength=side_width)

        live.bind("<Configure>", resize_live_labels)
        actions = ttk.Frame(live)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        self.start_button = ttk.Button(
            actions, text="開始執行", command=self._start, style="Accent.TButton")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions, text="停止佇列", command=self._stop,
            state="disabled", style="Danger.TButton")
        self.stop_button.pack(side="left", padx=6)
        self.open_job_button = ttk.Button(actions, text="開啟目前實驗資料夾", command=self._open_job_output, state="disabled")
        self.open_job_button.pack(side="right")
        self.view_results_button = ttk.Button(actions, text="查看結果", command=self._view_job_results)
        self.view_results_button.pack(side="right", padx=6)

        # 紀錄與曲線共用可伸縮區域，小視窗仍留得下佇列與執行操作。
        self.monitor_notebook = ttk.Notebook(right)
        self.monitor_notebook.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        curves_frame = ttk.Frame(self.monitor_notebook, padding=6)
        self.live_curves_tab = curves_frame
        curves_frame.rowconfigure(0, weight=1)
        self.reward_canvas = self._make_plot(curves_frame, "評估 Reward（回合累積平均）", 0)
        self.aoi_canvas = self._make_plot(curves_frame, "評估 AoI（評估回合平均）", 1)
        curve_actions = ttk.Frame(curves_frame)
        curve_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.export_curve_button = ttk.Button(
            curve_actions, text="匯出顯示曲線 CSV", command=self._export_live_curves)
        self.export_curve_button.pack(side="right")
        log_frame = ttk.Frame(self.monitor_notebook, padding=6)
        self.monitor_notebook.add(log_frame, text="執行紀錄")
        self.monitor_notebook.add(curves_frame, text="即時曲線")
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill="x", pady=(0, 4))
        self.log_search_var = tk.StringVar(value="")
        self.log_match_count_var = tk.StringVar(value="0 個命中")
        self.log_follow_var = tk.BooleanVar(value=True)
        ttk.Label(log_toolbar, text="搜尋").pack(side="left", padx=(0, 4))
        self.log_search_entry = ttk.Entry(log_toolbar, textvariable=self.log_search_var, width=15)
        self.log_search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.log_match_count_label = ttk.Label(
            log_toolbar, textvariable=self.log_match_count_var)
        self.log_match_count_label.pack(side="left", padx=(0, 4))
        self.log_clear_search_button = ttk.Button(
            log_toolbar, text="清除", width=5, command=self._clear_log_search)
        self.log_clear_search_button.pack(side="left", padx=(0, 4))
        # 搜尋與檔案操作分兩列，小視窗仍能看到所有按鈕。
        log_actions = ttk.Frame(log_frame)
        log_actions.pack(fill="x", pady=(0, 4))
        self.log_follow_button = ttk.Checkbutton(
            log_actions, text="跟隨最新", variable=self.log_follow_var,
            command=self._on_log_follow)
        self.log_follow_button.pack(side="left", padx=(0, 4))
        self.log_previous_button = ttk.Button(
            log_actions, text="←", width=3, command=lambda: self._navigate_log_match(-1),
            state="disabled")
        self.log_previous_button.pack(side="left", padx=(0, 2))
        self.log_next_button = ttk.Button(
            log_actions, text="→", width=3, command=lambda: self._navigate_log_match(1),
            state="disabled")
        self.log_next_button.pack(side="left", padx=(0, 4))
        self.log_copy_button = ttk.Button(log_actions, text="複製", width=6, command=self._copy_log)
        self.log_copy_button.pack(side="left", padx=(0, 4))
        self.log_save_button = ttk.Button(log_actions, text="另存…", width=6, command=self._save_log)
        self.log_save_button.pack(side="left")
        self.log = tk.Text(log_frame, height=3, width=55, wrap="word", state="disabled", font=MONO_FONT)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.tag_configure("log_search_match", background="#fff1a8", foreground="#202020")
        self.log.tag_configure(
            "log_search_current", background="#286090", foreground="white", underline=True)
        self.log.tag_raise("log_search_current")
        # 快捷鍵只綁在搜尋欄與日誌文字，避免影響其他輸入框。
        for widget in (self.log_search_entry, self.log):
            widget.bind("<F3>", self._on_log_next_key, add="+")
            widget.bind("<Shift-F3>", self._on_log_previous_key, add="+")
        self.log_search_var.trace_add("write", self._refresh_log_search)

    # 收合或展開次要的批次、範本與匯出工具。
    def _toggle_advanced_tools(self):
        if self.advanced_tools_var.get():
            self.advanced_tools.grid(
                row=6, column=0, sticky="ew", pady=(3, 0))
        else:
            self.advanced_tools.grid_remove()

    # 建立顯示訓練曲線的 canvas。
    def _make_plot(self, parent, title, row):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        if row == 0:
            padx = (0, 0)
            curve_key = "reward"
        else:
            padx = (4, 0)
            curve_key = "aoi"
        box.grid(row=0, column=row, sticky="nsew", padx=padx)
        parent.columnconfigure(row, weight=1)
        canvas = tk.Canvas(box, width=260, height=60, background="white", highlightthickness=1, highlightbackground="#cccccc")
        canvas.pack(fill="both", expand=True)
        canvas.bind("<Configure>", lambda _e, c=canvas, k=curve_key: self._draw_curve(c, k))
        return canvas

    # 切換方法時保留通用草稿；明確載入設定時則取代整份表單。
    def _select_method(self, settings: dict | None = None):
        method = self.method_var.get()
        common_values = {}
        try:
            if settings is None:
                settings = self.core.default_settings(method)
                # 保留原始輸入，避免驗證或正規化掉隱藏及尚未填完的欄位。
                common_values = {path: variable.get()
                                 for path, (variable, _kind, _default) in self.field_vars.items()
                                 if path[0] in _COMMON_SETTING_GROUPS}
        except Exception as exc:
            messagebox.showerror("無法載入設定", str(exc), parent=self.root)
            return
        self._render_settings(settings)
        for path, value in common_values.items():
            if path in self.field_vars:
                self.field_vars[path][0].set(value)
        if common_values:
            self._apply_settings_filter()
        self.name_var.set("")

    # 設定表單自行管理分頁、搜尋、適用性與輸入狀態。
    def _render_settings(self, settings: dict):
        self.settings_editor.render(self.method_var.get(), settings)

    def _apply_settings_filter(self, *_args):
        self.settings_editor.apply_filter()

    # 批次軸仍使用原始設定路徑，避免顯示名稱改變資料格式。
    def _setting_paths(self, settings=None):
        return self.settings_editor.active_paths(settings)

    # 局部範本只讀取選定欄位，不驗證其餘未完成草稿。
    def _read_setting_field(self, path):
        return self.settings_editor.read_field(path)

    def _collect_settings(self):
        try:
            return self.settings_editor.collect_settings()
        except ValueError:
            self.main_notebook.select(self.experiment_tab)
            raise

    # 獨立視窗管理可重複使用的局部設定，不驗證未選取的表單草稿。
    def _open_preset_manager(self):
        if self.preset_dialog is not None and self.preset_dialog.winfo_exists():
            self.preset_dialog.lift()
            self.preset_dialog.focus_set()
            return
        from gui.experiment_presets import PresetDialog

        self.preset_dialog = PresetDialog(self, directory=self.preset_directory)
        self.preset_dialog.grab_set()

    # 先驗證預設值與局部覆寫，再更新表單；佇列仍須手動加入或套用編輯。
    def _apply_preset(self, preset):
        if not self._queue_editable():
            return False
        try:
            from gui.preset_config import resolve_preset

            spec = resolve_preset(preset, self.method_var.get())
        except (ValueError, TypeError, OverflowError) as error:
            messagebox.showerror("無法套用設定範本", str(error), parent=self.root)
            return False
        name = self.name_var.get()
        self.method_var.set(spec["method"])
        self._select_method(settings=spec["settings"])
        self.name_var.set(name)
        self.main_notebook.select(self.experiment_tab)
        self._append_log(f"已套用設定範本：{spec['name']}（{spec['method']}）。")
        return True

    # 驗證目前表單，取得可加入佇列的實驗快照。
    def _snapshot_from_form(self):
        name = self.name_var.get().strip()
        if not name:
            name = f"{self.method_var.get()} 實驗 {len(self.queue_items) + 1}"
        spec = {"name": name, "method": self.method_var.get(), "settings": self._collect_settings()}
        try:
            return self.core.validate_experiment(spec)
        except ValueError as error:
            # 核心驗證仍是唯一判斷來源；介面只將可定位的欄位帶回眼前。
            message = str(error)
            mentioned = [path for path in self.field_vars if ".".join(path) in message]
            if mentioned:
                path = min(mentioned, key=lambda item: message.index(".".join(item)))
                self.main_notebook.select(self.experiment_tab)
                self.settings_editor.focus_field(path)
                raise ValueError(
                    f"「{field_label(path)}」（{'.'.join(path)}）設定無效。\n{message}"
                ) from error
            raise

    # 驗證並匯出目前表單快照，不加入或修改佇列。
    def _export_form_settings(self):
        try:
            spec = self._snapshot_from_form()
        except Exception as exc:
            messagebox.showerror("設定有誤", str(exc), parent=self.root)
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="匯出表單設定 JSON",
            defaultextension=".json",
            filetypes=(("JSON 檔案", "*.json"), ("所有檔案", "*.*")),
        )
        if not filename:
            return
        try:
            saving.write_json(filename, spec, strict=True, atomic=True)
        except (OSError, ValueError) as exc:
            messagebox.showerror("匯出失敗", f"無法寫入設定 JSON：{exc}", parent=self.root)
            return
        self._append_log(f"已匯出表單設定 JSON：{filename}")

    # 加入新實驗，或將表單內容套用到編輯中的項目。
    def _add_to_queue(self):
        if not self._queue_editable():
            return
        try:
            spec = self._snapshot_from_form()
        except Exception as exc:
            messagebox.showerror("設定有誤", str(exc), parent=self.root)
            return
        if self.selected_edit_index is None:
            self.queue_items.append({"spec": spec, "status": "等待中", "output_dir": None})
        else:
            if not 0 <= self.selected_edit_index < len(self.queue_items):
                self._cancel_edit()
                messagebox.showerror("編輯已失效", "佇列項目已變更，請重新選取要編輯的實驗。", parent=self.root)
                return
            self.queue_items[self.selected_edit_index] = {"spec": spec, "status": "等待中", "output_dir": None}
        self._cancel_edit()
        self._refresh_queue()
        self.name_var.set("")

    # 檢查目前表單，不加入佇列或啟動訓練。
    def _validate_form(self):
        try:
            self._snapshot_from_form()
        except Exception as error:
            messagebox.showerror("設定有誤", str(error), parent=self.root)
            return
        messagebox.showinfo("設定檢查", "設定有效，可以加入佇列。", parent=self.root)

    # 回復目前方法的預設設定，保留名稱與編輯中的項目。
    def _reset_form(self):
        if not self._queue_editable():
            return
        name = self.name_var.get()
        self._select_method(settings=self.core.default_settings(self.method_var.get()))
        self.name_var.set(name)

    # 以目前表單為基準，指定一個參數及多個值。
    def _add_parameter_sweep(self):
        if not self._queue_editable():
            return
        try:
            spec = self._snapshot_from_form()
        except Exception as error:
            messagebox.showerror("設定有誤", str(error), parent=self.root)
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("建立參數批次實驗")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        paths = self._setting_paths()
        if "TRAIN.seed" in paths:
            initial_path = "TRAIN.seed"
        else:
            initial_path = paths[0]
        display_paths = {
            f"{field_label(tuple(path.split('.')))}（{path}）": path for path in paths
        }
        initial_label = next(label for label, path in display_paths.items() if path == initial_path)
        field_var = tk.StringVar(master=dialog, value=initial_label)
        values_var = tk.StringVar(master=dialog, value="[0, 1, 2]")
        ttk.Label(body, text="參數欄位").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(body, textvariable=field_var, values=list(display_paths), state="readonly", width=62).grid(
            row=0, column=1, sticky="ew")
        ttk.Label(body, text="參數值清單").grid(row=1, column=0, sticky="w", pady=8)
        values_entry = ttk.Entry(body, textvariable=values_var, width=48)
        values_entry.grid(row=1, column=1, sticky="ew", pady=8)
        ttk.Label(body, text="JSON 清單，例如 [0, 1, 2] 或 [0.0001, 0.001]；一次最多 100 個實驗。").grid(
            row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(body, text="每個值使用同一份基準設定，完成檢查後一起加入待執行佇列。").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 12))

        def submit():
            try:
                values = json.loads(values_var.get())
                field_path = display_paths.get(field_var.get(), field_var.get())
                count = self._queue_sweep(spec, field_path, values)
            except (TypeError, ValueError) as error:
                messagebox.showerror("批次設定有誤", str(error), parent=dialog)
                return
            if count:
                self.status_var.set(f"已加入 {count} 個批次實驗")
                dialog.destroy()

        ttk.Button(body, text="取消", command=dialog.destroy).grid(row=4, column=0, sticky="w")
        ttk.Button(body, text="加入佇列", command=submit).grid(row=4, column=1, sticky="e")
        values_entry.bind("<Return>", lambda _event: submit())
        values_entry.focus_set()
        dialog.grab_set()

    # 以目前表單為基準，輸入多個完整欄位路徑及其測試值。
    def _add_parameter_grid(self):
        if not self._queue_editable():
            return
        try:
            spec = self._snapshot_from_form()
        except Exception as error:
            messagebox.showerror("設定有誤", str(error), parent=self.root)
            return

        initial_axes = {"TRAIN.seed": [0, 1]}
        common_groups = {"ENV", "TRAIN", "EVAL", "FUNCTIONAL", "FIXED", "RECORD"}
        active_paths = set(self._setting_paths(spec["settings"]))
        for path in self.field_vars:
            if (".".join(path) in active_paths and len(path) == 2
                    and path[0] not in common_groups):
                if path[1] in {"lr", "actor_lr"}:
                    initial_axes[".".join(path)] = [spec["settings"][path[0]][path[1]]]
                    break

        dialog = tk.Toplevel(self.root)
        dialog.title("建立多欄參數組合")
        dialog.transient(self.root)
        dialog.geometry("760x560")
        dialog.minsize(680, 500)
        dialog.resizable(True, True)
        body = ttk.Frame(dialog, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(4, weight=1)

        paths = self._setting_paths(spec["settings"])
        display_paths = {
            f"{field_label(tuple(path.split('.')))}（{path}）": path for path in paths
        }
        path_labels = {path: label for label, path in display_paths.items()}
        initial_path = next(iter(initial_axes))
        initial_label = path_labels.get(initial_path, initial_path)
        field_var = tk.StringVar(master=dialog, value=initial_label)
        initial_values = initial_axes[initial_path]
        values_var = tk.StringVar(
            master=dialog,
            value=json.dumps(initial_values, ensure_ascii=False),
        )

        ttk.Label(body, text="欄位").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        field_box = ttk.Combobox(
            body,
            textvariable=field_var,
            values=list(display_paths),
            state="readonly",
            width=48,
        )
        field_box.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(body, text="候選值（JSON清單）").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        values_entry = ttk.Entry(body, textvariable=values_var, width=34)
        values_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))

        def show_error(message):
            messagebox.showerror("多欄組合有誤", message, parent=dialog)

        def read_axes_for_edit():
            try:
                axes = json.loads(axes_text.get("1.0", "end-1c"))
            except (TypeError, ValueError) as error:
                show_error(f"下方 JSON 目前無效，已保留草稿。\n{error}")
                return None
            if not isinstance(axes, dict):
                show_error("下方 JSON 必須是以欄位路徑為鍵的物件；已保留草稿。")
                return None
            return axes

        def format_axes(axes):
            lines = []
            factors = []
            invalid_lists = False
            if not axes:
                lines.append("尚無參數軸；請在上方選擇欄位並加入候選值。")
                return lines

            for field_path, candidates in axes.items():
                label = field_label(tuple(field_path.split(".")))
                if isinstance(candidates, list):
                    count_label = f"{len(candidates)} 個候選值"
                    factors.append(len(candidates))
                    if not candidates:
                        invalid_lists = True
                else:
                    count_label = "候選值須為 JSON 清單"
                    invalid_lists = True
                lines.append(f"{label}（{field_path}）：{count_label}")

            if invalid_lists:
                lines.append("候選清單不可空，且必須使用 JSON list。")
            else:
                combination_count = 1
                for factor in factors:
                    combination_count *= factor
                product_text = " × ".join(str(factor) for factor in factors)
                summary = f"{product_text} = {combination_count} 組"
                if combination_count > 100:
                    summary += "；超過 100 組上限，不會加入佇列。"
                else:
                    summary += "（所有組合）"
                lines.append(summary)
            return lines

        def update_preview(_event=None):
            try:
                axes = json.loads(axes_text.get("1.0", "end-1c"))
            except (TypeError, ValueError):
                lines = ["JSON 格式無效；修正後即可預覽組合數。"]
            else:
                if isinstance(axes, dict):
                    lines = format_axes(axes)
                else:
                    lines = ["JSON 須為以欄位路徑為鍵的物件。"]
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", "\n".join(lines))
            preview_text.configure(state="disabled")

        def add_or_update_axis():
            axes = read_axes_for_edit()
            if axes is None:
                return
            try:
                candidates = json.loads(values_var.get())
            except (TypeError, ValueError) as error:
                show_error(f"候選值必須是 JSON 清單。\n{error}")
                return
            if not isinstance(candidates, list) or not candidates:
                show_error("候選值必須是非空 JSON 清單，例如 [0, 1]。")
                return

            field_path = display_paths.get(field_var.get())
            if field_path is None:
                show_error("請先從欄位清單中選擇一個欄位。")
                return
            axes[field_path] = candidates
            try:
                merged_json = json.dumps(axes, ensure_ascii=False, indent=2, allow_nan=False)
            except (TypeError, ValueError) as error:
                show_error(f"候選值無法保存為 JSON；已保留草稿。\n{error}")
                return
            axes_text.delete("1.0", "end")
            axes_text.insert("1.0", merged_json)
            axes_text.edit_modified(False)
            update_preview()

        def load_selected_axis(_event=None):
            try:
                axes = json.loads(axes_text.get("1.0", "end-1c"))
            except (TypeError, ValueError):
                return
            if not isinstance(axes, dict):
                return
            field_path = display_paths.get(field_var.get())
            if field_path in axes:
                values_var.set(json.dumps(axes[field_path], ensure_ascii=False))
            else:
                values_var.set("[]")

        field_box.bind("<<ComboboxSelected>>", load_selected_axis)
        ttk.Button(body, text="加入／更新欄位", command=add_or_update_axis).grid(
            row=1, column=2, sticky="e", padx=(8, 0), pady=(0, 6))
        ttk.Label(
            body,
            text="值須為非空清單，例如 [0, 1] 或 [1e-12, 2e-12]。",
            font=SMALL_FONT,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(
            body,
            text="參數組合 JSON（可直接編輯；最多 100 組）",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))
        axes_frame = ttk.Frame(body)
        axes_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")
        axes_frame.rowconfigure(0, weight=1)
        axes_frame.columnconfigure(0, weight=1)
        axes_text = tk.Text(axes_frame, width=70, height=8, wrap="none", font=MONO_FONT)
        axes_text.grid(row=0, column=0, sticky="nsew")
        axes_scroll_y = ttk.Scrollbar(axes_frame, orient="vertical", command=axes_text.yview)
        axes_scroll_y.grid(row=0, column=1, sticky="ns")
        axes_scroll_x = ttk.Scrollbar(axes_frame, orient="horizontal", command=axes_text.xview)
        axes_scroll_x.grid(row=1, column=0, sticky="ew")
        axes_text.configure(yscrollcommand=axes_scroll_y.set, xscrollcommand=axes_scroll_x.set)
        axes_text.insert("1.0", json.dumps(initial_axes, ensure_ascii=False, indent=2))
        axes_text.edit_modified(False)

        def on_axes_modified(_event=None):
            if axes_text.edit_modified():
                axes_text.edit_modified(False)
                update_preview()

        axes_text.bind("<<Modified>>", on_axes_modified)

        ttk.Label(body, text="組合預覽", font=SMALL_BOLD_FONT).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(6, 2))
        preview_frame = ttk.Frame(body)
        preview_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        preview_frame.columnconfigure(0, weight=1)
        preview_text = tk.Text(
            preview_frame,
            height=4,
            wrap="word",
            state="disabled",
            takefocus=False,
            font=SMALL_FONT,
        )
        preview_text.grid(row=0, column=0, sticky="ew")
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
        preview_scroll.grid(row=0, column=1, sticky="ns")
        preview_text.configure(yscrollcommand=preview_scroll.set)
        ttk.Label(
            body,
            text="以所有軸的候選值組合計算，不是逐項配對；預覽不會加入佇列。",
            font=SMALL_FONT,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 0))
        update_preview()

        buttons = ttk.Frame(body)
        buttons.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left")

        def submit():
            try:
                axes = json.loads(axes_text.get("1.0", "end-1c"))
                count = self._queue_parameter_grid(spec, axes)
            except (TypeError, ValueError) as error:
                messagebox.showerror("多欄組合有誤", str(error), parent=dialog)
                return
            if count:
                self.status_var.set(f"已加入 {count} 個多欄組合實驗")
                dialog.destroy()

        ttk.Button(buttons, text="建立佇列", command=submit).pack(side="right")
        axes_text.focus_set()
        dialog.grab_set()

    # 將已完整驗證的實驗一次加入待執行列。
    def _append_queue_items(self, experiments):
        first_index = len(self.queue_items)
        self.queue_items.extend({"spec": item, "status": "等待中", "output_dir": None}
                                for item in experiments)
        self._cancel_edit()
        self._refresh_queue()
        self.queue_tree.selection_set(str(first_index))
        self.queue_tree.focus(str(first_index))
        self.queue_tree.see(str(first_index))
        self._update_selected_output_buttons()
        return len(experiments)

    # 全部欄位組合驗證成功後一次加入待執行列。
    def _queue_parameter_grid(self, spec, axes):
        if not self._queue_editable():
            return 0
        experiments = self.core.expand_parameter_grid(spec, axes)
        return self._append_queue_items(experiments)

    # 全部實驗驗證完成才更新佇列，失敗時保留原有項目與編輯狀態。
    def _queue_sweep(self, spec, field_path, values):
        if not self._queue_editable():
            return 0
        experiments = self.core.expand_parameter_sweep(spec, field_path, values)
        return self._append_queue_items(experiments)

    # 結束項目編輯，恢復加入佇列模式。
    def _cancel_edit(self):
        self.selected_edit_index = None
        self.form_action.configure(text="加入佇列")
        self.cancel_edit_button.configure(state="disabled")

    # 取得目前選取的佇列項目索引。
    def _selected_index(self):
        selection = self.queue_tree.selection()
        if not selection:
            return None
        try:
            return int(selection[0])
        except (TypeError, ValueError):
            return None

    # 複製選取項目的設定，新增為待執行實驗。
    def _duplicate_selected(self):
        if not self._queue_editable():
            return
        index = self._selected_index()
        if index is None:
            return
        self._cancel_edit()
        clone = copy.deepcopy(self.queue_items[index])
        # 預留副本標記，避免長名稱超過設定驗證的 120 字上限。
        copy_suffix = " (副本)"
        clone["spec"]["name"] = clone["spec"]["name"][:120 - len(copy_suffix)] + copy_suffix
        clone["status"] = "等待中"
        clone["output_dir"] = None
        self.queue_items.insert(index + 1, clone)
        self._refresh_queue()
        new_index = index + 1
        self.queue_tree.selection_set(str(new_index))
        self.queue_tree.focus(str(new_index))
        self.queue_tree.see(str(new_index))
        self._update_selected_output_buttons()

    # 將選取項目的設定載入表單供修改。
    def _edit_selected(self):
        if not self._queue_editable():
            return
        index = self._selected_index()
        if index is None:
            return
        spec = copy.deepcopy(self.queue_items[index]["spec"])
        self.method_var.set(spec["method"])
        self._select_method(settings=spec["settings"])
        self.name_var.set(spec["name"])
        self.selected_edit_index = index
        self.form_action.configure(text="套用編輯")
        self.cancel_edit_button.configure(state="normal")

    # 移除選取項目並處理目前編輯狀態。
    def _remove_selected(self):
        if not self._queue_editable():
            return
        index = self._selected_index()
        if index is not None:
            self._cancel_edit()
            del self.queue_items[index]
            self._refresh_queue()
            if self.queue_items:
                selected_index = min(index, len(self.queue_items) - 1)
                self.queue_tree.selection_set(str(selected_index))
                self.queue_tree.focus(str(selected_index))
                self.queue_tree.see(str(selected_index))
                self._update_selected_output_buttons()

    # 清理終態列，保留待執行項目與仍在編輯的表單。
    def _remove_finished(self):
        if not self._queue_editable():
            return
        edit_row = None
        if self.selected_edit_index is not None:
            edit_row = self.queue_items[self.selected_edit_index]
        remaining = [row for row in self.queue_items
                     if row["status"] not in {"完成", "失敗", "已取消／略過"}]
        removed_count = len(self.queue_items) - len(remaining)
        if not removed_count:
            self._append_log("佇列沒有已結束項目。")
            return
        self.queue_items = remaining
        if edit_row is not None:
            # 列號會移動，以原列物件找回編輯中的項目。
            self.selected_edit_index = None
            for index, row in enumerate(remaining):
                if row is edit_row:
                    self.selected_edit_index = index
                    break
            if self.selected_edit_index is None:
                self._cancel_edit()
        self._refresh_queue()
        self._append_log(f"已從佇列移除 {removed_count} 個已結束項目。")

    # 將選取項目向前或向後移動。
    def _move_selected(self, offset):
        if not self._queue_editable():
            return
        index = self._selected_index()
        if index is not None:
            target = index + offset
        else:
            target = -1
        if index is None or target < 0 or target >= len(self.queue_items):
            return
        self._cancel_edit()
        self.queue_items[index], self.queue_items[target] = self.queue_items[target], self.queue_items[index]
        self._refresh_queue()
        self.queue_tree.selection_set(str(target))
        self.queue_tree.focus(str(target))
        self.queue_tree.see(str(target))
        self._update_selected_output_buttons()

    # 判斷目前是否允許修改實驗佇列。
    def _queue_editable(self):
        if self.queue_active or self.runner.is_running:
            return False
        return True

    # 依列物件身分增量更新佇列，保留仍存在項目的選取、焦點與捲動位置。
    def _refresh_queue(self):
        old_rows = self.queue_iid_rows
        old_iids = self.queue_tree.get_children()
        old_order = [old_rows.get(iid) for iid in old_iids]
        structure_changed = len(old_order) != len(self.queue_items)
        if not structure_changed:
            for index, row in enumerate(self.queue_items):
                if old_order[index] is not row:
                    structure_changed = True
                    break

        selected_ids = []
        for iid in self.queue_tree.selection():
            row = old_rows.get(iid)
            if row is not None:
                selected_ids.append(id(row))
        focus_row = old_rows.get(self.queue_tree.focus())
        top_row = None
        old_yview = None
        if structure_changed:
            old_yview = self.queue_tree.yview()
            top_y = None
            for iid in old_iids:
                row = old_rows.get(iid)
                bbox = self.queue_tree.bbox(iid)
                if row is None or not bbox:
                    continue
                y = int(bbox[1])
                if y >= 0 and (top_y is None or y < top_y):
                    top_row = row
                    top_y = y

        new_rows = {}
        new_display = {}
        new_iid_by_identity = {}
        existing_iids = set(old_iids)
        self._refreshing_queue = True
        try:
            for index, row in enumerate(self.queue_items):
                iid = str(index)
                spec = row["spec"]
                text = f"{index + 1}. {spec['name']}"
                values = (METHOD_LABELS.get(spec["method"], spec["method"]), row["status"])
                status_tags = {
                    "執行中": "queue_running",
                    "完成": "queue_completed",
                    "失敗": "queue_failed",
                    "已取消／略過": "queue_cancelled",
                }
                tags = (status_tags[row["status"]],) if row["status"] in status_tags else ()
                if iid in existing_iids:
                    if self.queue_iid_display.get(iid) != (text, values):
                        self.queue_tree.item(iid, text=text, values=values, tags=tags)
                    else:
                        self.queue_tree.item(iid, tags=tags)
                else:
                    self.queue_tree.insert(
                        "", "end", iid=iid, text=text, values=values, tags=tags)
                new_rows[iid] = row
                new_display[iid] = (text, values)
                new_iid_by_identity[id(row)] = iid

            desired_iids = set(new_rows)
            for iid in self.queue_tree.get_children():
                if iid not in desired_iids:
                    self.queue_tree.delete(iid)
            if structure_changed:
                for index, iid in enumerate(new_rows):
                    if self.queue_tree.index(iid) != index:
                        self.queue_tree.move(iid, "", index)
            self.queue_iid_rows = new_rows
            self.queue_iid_display = new_display

            if structure_changed:
                current_selection = self.queue_tree.selection()
                if current_selection:
                    self.queue_tree.selection_remove(*current_selection)
                selected_iids = []
                for row_id in selected_ids:
                    iid = new_iid_by_identity.get(row_id)
                    if iid is not None:
                        selected_iids.append(iid)
                if selected_iids:
                    self.queue_tree.selection_set(*selected_iids)

                if focus_row is not None:
                    focus_iid = new_iid_by_identity.get(id(focus_row))
                    if focus_iid is not None:
                        self.queue_tree.focus(focus_iid)
                    else:
                        self.queue_tree.focus("")
                else:
                    self.queue_tree.focus("")

                if top_row is not None:
                    top_iid = new_iid_by_identity.get(id(top_row))
                else:
                    top_iid = None
                if top_iid is not None:
                    top_fraction = int(top_iid) / max(1, len(new_rows))
                    self.queue_tree.yview_moveto(top_fraction)
                    self.queue_tree.see(top_iid)
                elif old_yview:
                    self.queue_tree.yview_moveto(old_yview[0])
        finally:
            self._refreshing_queue = False

        counts = {
            "等待中": 0,
            "排程中": 0,
            "執行中": 0,
            "完成": 0,
            "失敗": 0,
            "已取消／略過": 0,
        }
        for row in self.queue_items:
            status = row.get("status")
            if status in counts:
                counts[status] += 1
        self.queue_count_var.set(
            f"等待 {counts['等待中']}｜排程 {counts['排程中']}｜執行 {counts['執行中']}｜"
            f"完成 {counts['完成']}｜失敗 {counts['失敗']}｜取消 {counts['已取消／略過']}"
        )

        running = self.queue_active or self.runner.is_running
        # 執行中鎖住方法切換與佇列編輯；閒置時方法選單只允許選取。
        if running:
            widget_state = "disabled"
            method_state = "disabled"
        else:
            widget_state = "normal"
            method_state = "readonly"
        for widget in (self.choose_output_button, self.form_action,
                       self.sweep_button, self.grid_button, self.reset_button):
            widget.configure(state=widget_state)
        self.load_queue_button.configure(state="normal")
        has_waiting = any(row["status"] == "等待中" for row in self.queue_items)
        if self.queue_stopping or self.closing or (running and not has_waiting):
            self.start_button.configure(state="disabled")
        else:
            self.start_button.configure(state="normal")
        if running:
            self.start_button.configure(text="排入後續")
        else:
            self.start_button.configure(text="開始執行")
        self.method_box.configure(state=method_state)
        self.output_entry.configure(state=widget_state)
        for widget in self.mutation_buttons:
            widget.configure(state=widget_state)
        if running or self.selected_edit_index is None:
            self.cancel_edit_button.configure(state="disabled")
        else:
            self.cancel_edit_button.configure(state="normal")
        self._update_selected_output_buttons()

    # 使用佇列列物件，而非目前活動實驗，取得所選輸出資料夾。
    def _selected_output_row(self):
        for iid in self.queue_tree.selection():
            row = self.queue_iid_rows.get(iid)
            if row is not None and row.get("output_dir"):
                return row
        return None

    # 只在目前選取的列有輸出路徑時開放檢視按鈕。
    def _update_selected_output_buttons(self):
        row = self._selected_output_row()
        if row is not None:
            state = "normal"
        else:
            state = "disabled"
        self.open_selected_output_button.configure(state=state)
        self.view_selected_results_button.configure(state=state)

    # 選取改變時同步所選實驗的檢視按鈕。
    def _on_queue_selection_changed(self, _event=None):
        if self._refreshing_queue:
            return
        self._update_selected_output_buttons()

    def _on_queue_key(self, event):
        # 沿用按鈕的執行中檢查，快捷鍵也不會修改正在排程的佇列。
        if event.keysym == "Return":
            self._edit_selected()
        elif event.keysym == "Delete":
            self._remove_selected()
        elif event.keysym in ("d", "D"):
            self._duplicate_selected()
        elif event.keysym in ("c", "C"):
            self._copy_selected_settings()
        return "break"

    def _copy_selected_settings(self):
        if self.closing:
            return
        index = self._selected_index()
        if index is None:
            self._append_log("請先選取要複製設定的實驗。")
            return
        # 讀取已加入佇列的完整設定，未套用的表單內容不會混入。
        spec = self.queue_items[index]["spec"]
        try:
            text = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except (tk.TclError, TypeError, ValueError) as error:
            messagebox.showerror("無法複製設定", str(error), parent=self.root)
            return
        self._append_log(f"已複製實驗設定：{spec['name']}")

    # 將目前佇列與輸出路徑保存為 JSON。
    def _save_queue(self):
        if self.closing:
            return
        output_root = self.output_var.get()
        if not output_root.strip() or "\x00" in output_root:
            messagebox.showerror("無法儲存佇列", "輸出資料夾不可空白，且不可包含空字元。",
                                 parent=self.root)
            return
        # 另存只讀取快照，執行中也可使用；背景事件不會改動這份設定。
        data = {"version": 1, "output_root": output_root,
                "experiments": [copy.deepcopy(item["spec"]) for item in self.queue_items]}
        filename = filedialog.asksaveasfilename(parent=self.root, title="儲存佇列",
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            saving.write_json(filename, data, atomic=True)
        except OSError as exc:
            messagebox.showerror("儲存失敗", str(exc), parent=self.root)

    # 驗證並載入單場設定或完整佇列 JSON。
    def _load_queue(self):
        if self.closing:
            return
        filename = filedialog.askopenfilename(parent=self.root, title="載入佇列或單場 config.json",
            filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")])
        if not filename:
            return
        try:
            data = saving.read_json(Path(filename), encoding="utf-8-sig")
            single_config = isinstance(data, dict) and "method" in data and "experiments" not in data
            if not single_config and isinstance(data, dict) and "version" in data:
                version = data["version"]
                if type(version) is not int or version != 1:
                    raise ValueError(f"不支援的佇列版本 {version!r}；目前僅支援整數版本 1。")
            # 單場設定包成一列；佇列物件則從 experiments 欄位取清單。
            if single_config:
                specs = [data]
            elif isinstance(data, dict):
                specs = data.get("experiments", data)
            else:
                specs = data
            if not isinstance(specs, list):
                raise ValueError("JSON 必須是單場設定（name、method、settings），或包含 experiments 清單的佇列。")
            loaded_output_root = None
            if not single_config and isinstance(data, dict) and "output_root" in data:
                loaded_output_root = data["output_root"]
                if (not isinstance(loaded_output_root, str)
                        or not loaded_output_root.strip() or "\x00" in loaded_output_root):
                    raise ValueError("output_root 必須是非空的資料夾路徑字串，且不可含空字元。")
            normalized = [self.core.validate_experiment(copy.deepcopy(spec)) for spec in specs]
            if self.closing:
                return
            running = self.queue_active or self.runner.is_running
            rows = [{"spec": spec, "status": "等待中", "output_dir": None} for spec in normalized]
            self.queue_items.extend(rows)
            if loaded_output_root is not None and not running:
                self.output_var.set(loaded_output_root)
            self._refresh_queue()
            if single_config and not running:
                selected_index = len(self.queue_items) - 1
                self.queue_tree.selection_set(str(selected_index))
                self.queue_tree.focus(str(selected_index))
                self._update_selected_output_buttons()
                self._edit_selected()
            self._append_log(f"已追加匯入 {len(rows)} 場實驗，狀態為等待中。")
            if running and loaded_output_root is not None:
                self._append_log("執行中保留目前輸出資料夾，未套用 JSON 的 output_root。")
        except Exception as exc:
            messagebox.showerror("載入失敗", str(exc), parent=self.root)

    # 驗證歷史設定後帶回表單；新實驗仍由使用者加入佇列。
    def _load_result_settings(self, result):
        if not self._queue_editable():
            self.results_panel.status_var.set("執行中無法載入設定，請等佇列結束後再操作。")
            return False
        try:
            run_dir = result.get("run_dir")
            if run_dir is None:
                raise ValueError("此結果沒有保存設定的實驗資料夾。")
            config_path = result_config_path(result)
            if config_path is None:
                raise ValueError("此結果沒有保存設定檔。")
            data = saving.read_json(config_path, encoding="utf-8-sig")
            if isinstance(data, dict) and data.get("method") in self.core.RETIRED_METHODS:
                raise ValueError(
                    f"方法 {data['method']} 已退役；可查看歷史結果，不能載入為新訓練。")
            spec = self.core.validate_experiment(data)
            if spec["method"] != result.get("method"):
                raise ValueError("保存設定的方法與所選結果不一致。")
        except (OSError, ValueError, TypeError, OverflowError) as error:
            self.results_panel.status_var.set(f"設定未載入：{error}")
            return False
        # 先完成驗證才替換表單，失敗時保留尚未完成的輸入與編輯狀態。
        self._cancel_edit()
        self.method_var.set(spec["method"])
        self._select_method(spec["settings"])
        self.name_var.set(spec["name"])
        self.main_notebook.select(self.experiment_tab)
        if result.get("comparison_context"):
            self._append_log(
                f"已載入比較來源設定：{config_path}。新實驗會使用目前正式 HRL 的"
                "獨立下層與固定階段配額；收斂與導航指標只作診斷。")
        else:
            self._append_log(f"已載入保存設定：{config_path}。可修改後加入佇列。")
        return True

    # 開啟資料夾選擇器，更新輸出根目錄。
    def _choose_output(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=self.output_var.get(), title="選擇輸出資料夾")
        if path:
            self.output_var.set(path)

    # 使用作業系統的預設方式開啟指定資料夾。
    def _open_path(self, path):
        if not path:
            return
        try:
            target = Path(path).expanduser().resolve()
            if not target.is_dir():
                raise FileNotFoundError(f"資料夾不存在或不是資料夾：{target}")
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("無法開啟資料夾", str(exc), parent=self.root)

    # 開啟目前設定的輸出根目錄。
    def _open_output(self):
        self._open_path(self.output_var.get())

    # 開啟目前實驗的輸出資料夾。
    def _open_job_output(self):
        self._open_path(self.active_output_dir)

    # 開啟所選佇列項目的輸出資料夾。
    def _open_selected_output(self):
        row = self._selected_output_row()
        if row is not None:
            self._open_path(row.get("output_dir"))

    # 顯示目前實驗的結果；尚未開始時查詢目前設定的輸出資料夾。
    def _view_job_results(self):
        if self.active_output_dir:
            path = self.active_output_dir
        else:
            path = self.output_var.get()
        self.results_panel.set_root(path)
        self.results_panel.refresh()
        self.main_notebook.select(self.results_tab)

    # 查詢所選佇列項目的結果，不改變目前執行中的輸出路徑。
    def _view_selected_results(self):
        row = self._selected_output_row()
        if row is None:
            return
        self.results_panel.set_root(row["output_dir"])
        self.results_panel.refresh()
        self.main_notebook.select(self.results_tab)

    # 結果頁選取的軌跡交給原有回放面板，保留背景載入流程。
    def _view_result_trace(self, episode_path, step_path):
        self.replay_panel.load_paths(episode_path, step_path)
        self.main_notebook.select(self.replay_tab)

    # 閒置時啟動待執行項目；執行中則將等待項目接在目前排程後。
    def _start(self, indices=None):
        if self.closing or self.queue_stopping:
            return
        if indices is None:
            pending_indices = [
                index for index, row in enumerate(self.queue_items)
                if row["status"] == "等待中"
            ]
        else:
            pending_indices = list(indices)
        if not pending_indices:
            messagebox.showinfo("沒有待執行實驗", "請先將至少一個實驗加入佇列。", parent=self.root)
            return
        try:
            specs = [copy.deepcopy(self.queue_items[index]["spec"])
                     for index in pending_indices]
            if self.queue_active or self.runner.is_running:
                accepted = self.runner.append(specs)
                if not self.session_row_indices:
                    self.session_row_indices = list(self.run_row_indices)
                self.session_row_indices.extend(pending_indices)
                if accepted:
                    self.run_row_indices.extend(pending_indices)
                else:
                    # 後端剛完成尾端時，由 queue_done 接手啟動，避免遺失使用者的排程。
                    self.deferred_run_indices.extend(pending_indices)
                for index in pending_indices:
                    self.queue_items[index]["status"] = "排程中"
                self._append_log(f"已將 {len(pending_indices)} 場等待實驗排入後續。")
                self._refresh_queue()
                return
            output_root = self.output_var.get()
            if not output_root.strip() or "\x00" in output_root:
                raise ValueError("輸出資料夾不可空白，且不可包含空字元。")
            root = Path(output_root).expanduser().resolve()
            saving.make_directory(root, parents=True, exist_ok=True)
            self.runner.output_root = root
            self.runner.start(specs)
            self.run_row_indices = pending_indices
        except Exception as exc:
            messagebox.showerror("無法開始", str(exc), parent=self.root)
            return
        self.queue_active = True
        self.queue_stopping = False
        for index in pending_indices:
            self.queue_items[index]["status"] = "排程中"
        if self.run_started_at is None:
            self.run_started_at = time.monotonic()
            self.session_row_indices = list(pending_indices)
        self.experiment_started_at = None
        self.eta_var.set("本場預估剩餘：等待開始")
        self.stage_var.set("階段：等待開始")
        self.curves = {"reward": [], "aoi": []}
        self._draw_curve(self.reward_canvas, "reward")
        self._draw_curve(self.aoi_canvas, "aoi")
        self.status_var.set("執行中")
        self.progress_var.set("準備開始…")
        self.progress.configure(value=0)
        self.stop_button.configure(state="normal")
        self.open_job_button.configure(state="normal")
        self._refresh_queue()

    # 請求停止佇列並更新停止狀態。
    def _stop(self):
        if self.queue_active or self.runner.is_running:
            self.queue_stopping = True
            self.runner.stop()
            self.experiment_started_at = None
            self.eta_var.set("本場預估剩餘：已停止估算")
            self.stage_var.set("階段：停止中")
            self.status_var.set("正在停止，已排程的後續實驗將略過…")
            self.stop_button.configure(state="disabled")
            self._refresh_queue()

    # 定期處理背景事件，限制每次處理量以維持介面回應。
    def _poll_events(self):
        self._poll_after_id = None
        # 同時限制筆數與耗時；慢事件也要讓出 Tk 執行緒處理輸入及重繪。
        deadline = time.monotonic() + 0.016
        for _ in range(400):
            try:
                event = self.runner.events.get_nowait()
            except queue.Empty:
                break
            if not isinstance(event, dict):
                self._append_log("略過無效事件：事件內容必須是物件。")
            else:
                try:
                    self._handle_event(event)
                except (TypeError, ValueError, OverflowError) as error:
                    # 繼續接收後續完成／失敗事件，避免單筆異常輸出鎖住整個佇列。
                    self._append_log(f"略過無效事件：{type(error).__name__}: {error}")
            if time.monotonic() >= deadline:
                break
        if self.run_started_at is not None and self.runner.is_running:
            self.elapsed_var.set(f"經過時間：{_format_elapsed(time.monotonic() - self.run_started_at)}")
        self._refresh_eta()
        if self.closing and not self.runner.is_running:
            self.replay_panel.close()
            self.results_panel.close()
            self.scatter_panel.close()
            self.root.destroy()
            return
        if self.runner.events.empty():
            delay = 100
        else:
            delay = 10
        self._poll_after_id = self.root.after(delay, self._poll_events)

    # 視窗關閉前取消尚未執行的事件輪詢，避免 Tcl 呼叫已釋放的 callback。
    def _cancel_poll_events(self):
        if self._poll_after_id is None:
            return
        try:
            self.root.after_cancel(self._poll_after_id)
        except tk.TclError:
            pass
        self._poll_after_id = None

    # 清理由外部直接銷毀主視窗時尚未執行的輪詢 callback。
    def _on_root_destroy(self, event):
        if event.widget is self.root:
            self._cancel_poll_events()

    # 依事件種類更新實驗狀態、日誌與進度。
    def _handle_event(self, event):
        kind = event.get("type", event.get("event", ""))
        index = event.get("index")
        # Worker 的 index 從本次批次的 0 開始，換成完整佇列中的列號。
        queue_index = index
        if isinstance(index, int) and 0 <= index < len(self.run_row_indices):
            queue_index = self.run_row_indices[index]
        row = None
        if isinstance(queue_index, int) and 0 <= queue_index < len(self.queue_items):
            row = self.queue_items[queue_index]
        if kind == "started":
            self.experiment_started_at = time.monotonic()
            self.active_episode_count = 0
            self.active_max_steps = 0
            if row:
                row["status"] = "執行中"
                row["output_dir"] = event.get("output_dir")
                self.active_episode_count, self.active_max_steps = self._progress_limits(row["spec"]["settings"])
            self.eta_var.set("本場預估剩餘：估算中")
            self.stage_var.set("階段：啟動中")
            self.curves = {"reward": [], "aoi": []}
            self._draw_curve(self.reward_canvas, "reward")
            self._draw_curve(self.aoi_canvas, "aoi")
            self.active_output_dir = event.get("output_dir")
            self.progress.configure(value=0)
            fallback_name = ""
            if row:
                fallback_name = row["spec"]["name"]
            self.progress_var.set(f"開始：{event.get('name', fallback_name)}")
            self.status_var.set("執行中")
        elif kind == "progress":
            self._update_progress(event)
        elif kind == "log":
            self._append_log(str(event.get("text", "")))
        elif kind == "finished":
            status = self._status_label(str(event.get("status", "完成")))
            self.experiment_started_at = None
            if status == "完成":
                self.eta_var.set("本場預估剩餘：00:00（已完成）")
            else:
                self.eta_var.set(f"本場預估剩餘：{status}")
            self.stage_var.set(f"階段：{status}")
            if row:
                row["status"] = status
                row["output_dir"] = event.get("output_dir") or row.get("output_dir")
            self.active_output_dir = event.get("output_dir") or self.active_output_dir
            if row:
                experiment_name = row["spec"]["name"]
            else:
                experiment_name = index
            self._append_log(f"實驗 {experiment_name}：{status}")
            if status == "失敗":
                self.status_var.set(f"實驗失敗：{experiment_name}")
        elif kind == "queue_done":
            had_active_experiment = self.experiment_started_at is not None
            self.experiment_started_at = None
            self.queue_active = False
            terminal_error = event.get("error")
            stopped = bool(event.get("stopped") or event.get("cancelled")
                           or self.queue_stopping or self.closing)
            self.queue_stopping = False
            deferred_indices = self.deferred_run_indices
            self.deferred_run_indices = []
            if deferred_indices and not stopped and not terminal_error:
                for index in deferred_indices:
                    self.queue_items[index]["status"] = "等待中"
                self._start(indices=deferred_indices)
                if not self.queue_active:
                    self.run_started_at = None
                    self.status_var.set("後續實驗未啟動，保留等待中")
                    self.stop_button.configure(state="disabled")
                    self._refresh_queue()
                return
            if terminal_error:
                self.eta_var.set("本場預估剩餘：已中止")
            elif stopped:
                self.eta_var.set("本場預估剩餘：已停止")
            elif had_active_experiment:
                self.eta_var.set("本場預估剩餘：已結束")
            if stopped or terminal_error:
                for i in self.session_row_indices or self.run_row_indices:
                    if self.queue_items[i]["status"] in {"排程中", "執行中"}:
                        # 尚未開始的實驗是略過；失敗只屬於實際出錯的那一場。
                        if terminal_error and self.queue_items[i]["status"] == "執行中":
                            self.queue_items[i]["status"] = "失敗"
                        else:
                            self.queue_items[i]["status"] = "已取消／略過"
            if terminal_error:
                summary = f"佇列錯誤：{terminal_error}"
            elif stopped:
                summary = "已停止"
            else:
                failures = sum(
                    self.queue_items[i]["status"] == "失敗"
                    for i in self.session_row_indices or self.run_row_indices
                )
                if failures:
                    summary = f"佇列結束（{failures} 場失敗）"
                else:
                    summary = "全部完成"
            self.status_var.set(summary)
            self.stage_var.set(f"狀態：{summary}")
            self.stop_button.configure(state="disabled")
            self.start_button.configure(state="normal")
            self.run_started_at = None
            self.progress_var.set(self.status_var.get())
        else:
            if event.get("text"):
                self._append_log(str(event["text"]))
        # 逐步進度與日誌不改佇列狀態，避免每筆輸出都重設操作元件。
        if kind in {"started", "finished", "queue_done"}:
            self._refresh_queue()

    # 將後端狀態名稱轉成中文顯示文字。
    def _status_label(self, status):
        lower = status.lower()
        if "cancel" in lower or "skip" in lower or "stop" in lower or "略過" in status:
            return "已取消／略過"
        if "error" in lower or "fail" in lower or "錯誤" in status:
            return "失敗"
        return "完成"

    # 更新階段、訓練進度與曲線，分開處理評估事件。
    def _update_progress(self, event):
        subtype = str(event.get("event", "")).lower()
        episode = _progress_counter(event.get("episode"), 0)
        step = _progress_counter(event.get("step"), 0)
        total_steps = _progress_counter(event.get("total_steps"), 1)
        stage = event.get("stage") or event.get("phase")
        if stage is None:
            if subtype.startswith("evaluation") or subtype.startswith("eval_"):
                stage = "evaluation"
            else:
                stage = "training"
        stage_names = {
            "warmup": "暖身", "evaluation": "評估", "eval": "評估",
            "navigation": "導航預訓練", "navigation_pretrain": "離線示範初始化",
            "navigation_check": "導航檢查", "saving": "儲存",
            "training": "訓練", "upper": "上層策略訓練", "lower": "下層策略訓練",
            "joint": "聯合訓練",
        }
        stage = stage_names.get(str(stage).lower(), stage)
        pct = None
        phase_value = str(event.get("phase", "")).lower()
        is_eval = (
            subtype.startswith("evaluation")
            or subtype.startswith("eval_")
            or phase_value in {"evaluation", "eval"}
            or str(stage).lower() == "評估"
        )
        self.stage_var.set(f"階段：{stage}")
        if not is_eval and total_steps and step is not None:
            planned = self.active_episode_count * self.active_max_steps
            # 缺少有效總訓練步數時，改以目前回合的步數估算進度。
            if planned and episode is not None and episode > 0:
                completed = (episode - 1) * self.active_max_steps + step
                pct = 100.0 * (min(completed, planned) / planned)
            elif not planned:
                pct = 100.0 * (min(step, total_steps) / total_steps)
        elif (not is_eval and subtype == "episode_end" and episode is not None
              and episode > 0 and self.active_episode_count):
            pct = 100.0 * (min(episode, self.active_episode_count) / self.active_episode_count)
        if not is_eval and pct is not None:
            self.progress.configure(value=pct)
        bits = [str(stage)]
        if episode is not None:
            episode_text = f"回合 {episode}"
            if self.active_episode_count:
                episode_text += f" / {self.active_episode_count}"
            bits.append(episode_text)
        phase = event.get("phase")
        phase_names = {"upper": "上層", "lower": "下層", "joint": "聯合", "navigation": "導航",
                       "pretrain": "離線示範初始化", "evaluation": "評估", "eval": "評估"}
        if phase and str(phase).lower() not in {"train", "training"}:
            phase_text = phase_names.get(str(phase).lower(), str(phase))
            if phase_text != stage:
                bits.append(phase_text)
        if step is not None:
            if total_steps is not None:
                total_steps_text = f" / {total_steps}"
            else:
                total_steps_text = ""
            bits.append(f"步數 {step}" + total_steps_text)
        for label, keys in (
            ("累積環境 Reward", ("reward", "mean_reward")),
            ("回合平均 AoI", ("mean_aoi", "aoi")),
        ):
            val = next((event.get(key) for key in keys if event.get(key) is not None), None)
            if val is not None:
                try:
                    if isinstance(val, bool):
                        continue
                    metric = float(val)
                    if not math.isfinite(metric):
                        continue
                    bits.append(f"{label} {format_number(metric)}")
                except (TypeError, ValueError, OverflowError):
                    pass
        if subtype == "evaluation_end":
            for label, key, curve_key, canvas in (
                ("評估 Reward", "eval_reward", "reward", self.reward_canvas),
                ("評估 AoI", "eval_aoi", "aoi", self.aoi_canvas),
            ):
                value = event.get(key)
                if value is None or isinstance(value, bool):
                    continue
                try:
                    metric = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(metric):
                    continue
                bits.append(f"{label} {format_number(metric)}")
                if episode is not None and episode > 0:
                    self.curves[curve_key].append((float(episode), metric))
                    self.curves[curve_key] = self.curves[curve_key][-MAX_CURVE_POINTS:]
                    self._draw_curve(canvas, curve_key)
        self.progress_var.set("　·　".join(bits))
        self._refresh_eta()

    # 預估總時間 = 本場經過時間 / 進度比例；剩餘時間再扣除已經過時間。
    def _refresh_eta(self, now=None):
        if self.experiment_started_at is None:
            return
        if now is None:
            now = time.monotonic()
        fraction = float(self.progress["value"]) / 100.0
        if not math.isfinite(fraction) or fraction <= 0:
            text = "估算中"
        else:
            elapsed = max(0.0, now - self.experiment_started_at)
            seconds = max(0.0, elapsed / fraction - elapsed)
            text = f"約 {_format_elapsed(math.ceil(seconds))}"
        self.eta_var.set(f"本場預估剩餘：{text}")

    # 讀取總訓練回合數與每回合步數。
    @staticmethod
    def _progress_limits(settings):
        training = settings["TRAIN"]
        return int(training["episodes"]), int(training["max_steps"])

    # 在 canvas 繪製已保存的 Reward 或 AoI 曲線。
    def _draw_curve(self, canvas, key):
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        points = [point for point in self.curves[key]
                  if math.isfinite(point[0]) and math.isfinite(point[1])]
        w, h = max(80, canvas.winfo_width()), max(55, canvas.winfo_height())
        margin, left = 20, 80
        if not points:
            canvas.create_line(left, h - margin, w - 5, h - margin, fill="#b8b8b8")
            canvas.create_line(left, 5, left, h - margin, fill="#b8b8b8")
            canvas.create_text(w / 2, h / 2, text="等待回合資料…", fill="#777777")
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if math.isclose(ymin, ymax):
            pad = max(1.0, abs(ymin) * 0.05)
            padded_min, padded_max = ymin - pad, ymax + pad
            if math.isfinite(padded_min) and math.isfinite(padded_max):
                ymin, ymax = padded_min, padded_max
        ymax_text = format_number(ymax)
        ymin_text = format_number(ymin)
        tick_width = max(
            int(canvas.tk.call("font", "measure", SMALL_FONT, ymax_text)),
            int(canvas.tk.call("font", "measure", SMALL_FONT, ymin_text)),
        )
        left = max(left, tick_width + 12)
        canvas.create_line(left, h - margin, w - 5, h - margin, fill="#b8b8b8")
        canvas.create_line(left, 5, left, h - margin, fill="#b8b8b8")
        canvas.create_text(left - 4, 9, text=ymax_text, anchor="e", fill="#555555", font=SMALL_FONT)
        canvas.create_text(left - 4, h - margin, text=ymin_text, anchor="e", fill="#555555", font=SMALL_FONT)
        canvas.create_text(left, h - margin + 12, text=str(int(xmin)), anchor="w", fill="#555555", font=SMALL_FONT)
        if xmax != xmin:
            canvas.create_text(w - 8, h - margin + 12, text=str(int(xmax)), anchor="e", fill="#555555", font=SMALL_FONT)
        # 先縮放再計算差值，避免兩個有限極值相減後溢位成無限大。
        x_scale = max(1.0, abs(xmin), abs(xmax))
        y_scale = max(1.0, abs(ymin), abs(ymax))
        scaled_xmin, scaled_xmax = xmin / x_scale, xmax / x_scale
        scaled_ymin, scaled_ymax = ymin / y_scale, ymax / y_scale
        coords = []
        for x, y in points:
            if scaled_xmin == scaled_xmax:
                x_fraction = 0.0
            else:
                x_fraction = (x / x_scale - scaled_xmin) / (scaled_xmax - scaled_xmin)
            if scaled_ymin == scaled_ymax:
                y_fraction = 0.5
            else:
                y_fraction = (y / y_scale - scaled_ymin) / (scaled_ymax - scaled_ymin)
            px = left + x_fraction * max(1, w - left - 8)
            py = h - margin - y_fraction * (h - margin - 10)
            coords.extend((px, py))
        if len(coords) >= 4:
            canvas.create_line(*coords, fill="#2468a2", width=2, smooth=False)
        elif len(coords) == 2:
            canvas.create_oval(coords[0] - 2, coords[1] - 2, coords[0] + 2, coords[1] + 2, fill="#2468a2", outline="")

    # 匯出目前保留的有限曲線點，不重排或修改圖表資料。
    def _export_live_curves(self):
        rows = []
        for metric, curve_key in (("eval_reward", "reward"), ("eval_aoi", "aoi")):
            for point in self.curves[curve_key]:
                try:
                    episode = float(point[0])
                    value = float(point[1])
                except (TypeError, ValueError, OverflowError, IndexError):
                    continue
                if not math.isfinite(episode) or not math.isfinite(value):
                    continue
                rows.append((metric, episode, value))
        if not rows:
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="匯出顯示曲線 CSV",
            defaultextension=".csv",
            filetypes=(("CSV 檔案", "*.csv"), ("所有檔案", "*.*")),
        )
        if not filename:
            return
        try:
            saving.write_csv(filename, ("metric", "episode", "value"), rows)
        except OSError as exc:
            messagebox.showerror("匯出失敗", f"無法寫入曲線 CSV：{exc}", parent=self.root)
            return
        self._append_log(f"已匯出顯示曲線 CSV：{filename}（{len(rows)} 個點）")

    # 依實際行數及字元數保留日誌尾端，完整原始輸出仍由後端保存。
    def _append_log(self, text):
        if not text:
            return
        keep_view = not self.log_follow_var.get()
        if keep_view:
            self.log.mark_set("log_view_anchor", self.log.index("@0,0"))
            self.log.mark_gravity("log_view_anchor", "left")

        # 先限制單筆輸出，避免多行錯誤或超長行繞過畫面保留上限。
        appended_text = (text + "\n")[-MAX_LOG_CHARS:]
        appended_lines = appended_text[:-1].split("\n")[-MAX_LOG_LINES:]
        for index, line in enumerate(appended_lines):
            if len(line) > MAX_LOG_LINE_CHARS:
                # Tk 的自動換行處理超長單行時成本很高，畫面只保留有標記的尾端。
                marker = "[此行過長，僅顯示尾端] …"
                tail_length = MAX_LOG_LINE_CHARS - len(marker)
                appended_lines[index] = marker + line[-tail_length:]
        appended_text = "\n".join(appended_lines) + "\n"
        removed_count = max(0, len(self.log_lines) + len(appended_lines) - MAX_LOG_LINES)
        removed_length = sum(len(line) + 1 for line in self.log_lines[:removed_count])
        retained_length = self._log_content_length - removed_length + len(appended_text)
        while retained_length > MAX_LOG_CHARS:
            line_length = len(self.log_lines[removed_count]) + 1
            removed_length += line_length
            retained_length -= line_length
            removed_count += 1
        removed_lines = self.log_lines[:removed_count]
        self.log_lines = self.log_lines[removed_count:] + appended_lines

        # The final entry is the offset after the trailing newline. Reuse the
        # retained prefix and extend it for appended lines.
        self._log_line_starts = [
            offset - removed_length
            for offset in self._log_line_starts[removed_count:]
        ]
        for line in appended_lines:
            self._log_line_starts.append(
                self._log_line_starts[-1] + len(line) + 1)

        self.log.configure(state="normal")
        try:
            if removed_lines:
                delete_end = f"{removed_count + 1}.0"
                if self._log_search_has_current:
                    current_ranges = self.log.tag_ranges("log_search_current")
                    if not current_ranges or self.log.compare(current_ranges[0], "<", delete_end):
                        # 目前命中若被裁掉，就移除焦點標記，避免它落到新首行。
                        self._clear_log_current_match()
                self.log.delete("1.0", delete_end)
                self._log_content_length -= removed_length
                if self._log_search_current_offset is not None:
                    self._log_search_current_offset -= removed_length
                shifted_highlights = []
                for start_offset, end_offset in self._log_search_highlights:
                    if start_offset < removed_length:
                        continue
                    shifted_highlights.append((
                        start_offset - removed_length,
                        end_offset - removed_length,
                    ))
                self._log_search_highlights = shifted_highlights
            append_start = self._log_content_length
            self.log.insert("end-1c", appended_text)
            self._log_content_length += len(appended_text)
            if keep_view:
                self.log.see("log_view_anchor")
            else:
                self.log.see("end")
        finally:
            self.log.configure(state="disabled")

        query = self.log_search_var.get()
        if not query:
            self._log_search_match_count = 0
            self._log_search_match_ranges = []
            self._log_search_match_ranges_valid = True
            self._update_log_search_controls()
        elif "\n" in query or len(appended_lines) > 1:
            # 跨行查詢或多行紀錄可能改變相鄰行的命中，完整重算確保標記正確。
            self._refresh_log_search()
        else:
            removed_matches = 0
            for removed_line in removed_lines:
                removed_matches += len(self._find_log_match_ranges(removed_line, query))
            added_ranges = self._find_log_match_ranges(appended_lines[0], query)
            self._log_search_match_count += len(added_ranges) - removed_matches
            # Incremental count and visible tags stay cheap; F3 builds the full
            # list once on demand and reuses it until the next append.
            self._log_search_match_ranges_valid = False

            retained_added_ranges = added_ranges[-MAX_LOG_HIGHLIGHTS:]
            added_highlights = [
                (append_start + match_start, append_start + match_end)
                for match_start, match_end in retained_added_ranges
            ]
            overflow = max(
                0,
                len(self._log_search_highlights) + len(added_highlights)
                - MAX_LOG_HIGHLIGHTS,
            )
            if overflow:
                evicted_highlights = self._log_search_highlights[:overflow]
                self._log_search_highlights = self._log_search_highlights[overflow:]
                self._remove_log_tag_ranges("log_search_match", evicted_highlights)

            self._add_log_tag_ranges("log_search_match", added_highlights)
            self._log_search_highlights.extend(added_highlights)
            self._update_log_search_controls()

    # 重新勾選跟隨時立即顯示末尾，不必等待下一筆訓練日誌。
    def _on_log_follow(self):
        if self.log_follow_var.get():
            self.log.yview_moveto(1.0)

    # F3 導覽下一筆搜尋命中。
    def _on_log_next_key(self, _event=None):
        self._navigate_log_match(1)
        return "break"

    # Shift-F3 導覽上一筆搜尋命中。
    def _on_log_previous_key(self, _event=None):
        self._navigate_log_match(-1)
        return "break"

    # 清除目前命中標記，搜尋結果標記仍保留。
    def _clear_log_current_match(self):
        self.log.tag_remove("log_search_current", "1.0", "end")
        self.log.mark_unset("log_search_current_start")
        self._log_search_has_current = False
        self._log_search_current_offset = None

    # 將 Python 字元位移轉為短範圍 Tk index，避免從 1.0 掃描整份日誌。
    def _log_index_for_offset(self, offset):
        offset = min(max(offset, 0), self._log_content_length)
        line_index = bisect_right(self._log_line_starts, offset) - 1
        line_start = self._log_line_starts[line_index]
        return f"{line_index + 1}.0 + {offset - line_start} chars"

    # 將多個標記區間合併為一次 Tk call，減少大量命中時的跨語言呼叫。
    def _add_log_tag_ranges(self, tag_name, ranges):
        if not ranges:
            return
        indices = []
        for start_offset, end_offset in ranges:
            indices.extend((
                self._log_index_for_offset(start_offset),
                self._log_index_for_offset(end_offset),
            ))
        self.log.tag_add(tag_name, *indices)

    def _remove_log_tag_ranges(self, tag_name, ranges):
        # Text.tag_remove 只接受一對索引；各索引仍從所在行開始換算。
        for start_offset, end_offset in ranges:
            self.log.tag_remove(
                tag_name,
                self._log_index_for_offset(start_offset),
                self._log_index_for_offset(end_offset),
            )

    # 顯示完整命中數，並在超過黃色標記上限時說明只標示最近命中。
    def _update_log_search_controls(self):
        if self._log_search_match_count > MAX_LOG_HIGHLIGHTS:
            self.log_match_count_var.set(
                f"{self._log_search_match_count} 個命中（標記最近 {MAX_LOG_HIGHLIGHTS}）"
            )
        else:
            self.log_match_count_var.set(f"{self._log_search_match_count} 個命中")

        if self._log_search_match_count > 0:
            state = "normal"
        else:
            state = "disabled"
        self.log_previous_button.configure(state=state)
        self.log_next_button.configure(state=state)

    # 依字面命中導覽，邊界處環繞並保留作用中的命中位置。
    def _navigate_log_match(self, direction):
        # F3 仍可觸發；沒有命中時不必重掃整份日誌。
        if self._log_search_match_count == 0:
            self._clear_log_current_match()
            self._update_log_search_controls()
            return
        if self._log_search_match_ranges_valid:
            matches = self._log_search_match_ranges
        else:
            content = self.log.get("1.0", "end-1c")
            query = self.log_search_var.get()
            # 保留全部 Python 位移供導覽；Tk index 只在稍後轉換目標命中。
            matches = self._find_log_match_ranges(content, query)
            self._log_search_match_ranges = matches
            self._log_search_match_ranges_valid = True
            self._log_search_match_count = len(matches)
        if not matches:
            self._clear_log_current_match()
            self._update_log_search_controls()
            return

        current_index = None
        if self._log_search_has_current:
            current_offset = self._log_search_current_offset
            if current_offset is None:
                # Preserve correctness if a mark was restored outside navigation.
                current_offset = len(
                    self.log.get("1.0", "log_search_current_start"))
            match_index = bisect_left(matches, (current_offset, -1))
            if (match_index < len(matches)
                    and matches[match_index][0] == current_offset):
                current_index = match_index
        if current_index is None:
            if direction < 0:
                target_index = len(matches) - 1
            else:
                target_index = 0
        else:
            if direction < 0:
                target_index = (current_index - 1) % len(matches)
            else:
                target_index = (current_index + 1) % len(matches)

        start_offset, end_offset = matches[target_index]
        start = self.log.index(self._log_index_for_offset(start_offset))
        end = self.log.index(self._log_index_for_offset(end_offset))
        self.log.tag_remove("log_search_current", "1.0", "end")
        self.log.mark_set("log_search_current_start", start)
        self.log.mark_gravity("log_search_current_start", "left")
        self.log.tag_add("log_search_current", start, end)
        self._log_search_has_current = True
        self._log_search_current_offset = start_offset
        self.log_follow_var.set(False)
        self.log.see(start)
        self._update_log_search_controls()

    # 回傳字面、不重疊的大小寫折疊命中範圍，供完整與增量標記共用。
    def _find_log_match_ranges(self, content, query):
        folded_content = content.casefold()
        folded_query = query.casefold()
        if not folded_query:
            return []

        # With one folded code point per input code point, positions map directly.
        if len(folded_content) == len(content):
            match_ranges = []
            search_from = 0
            while True:
                match_start = folded_content.find(folded_query, search_from)
                if match_start < 0:
                    break
                match_end = match_start + len(folded_query)
                match_ranges.append((match_start, match_end))
                search_from = match_end
            return match_ranges

        folded_parts = []
        folded_to_original = []
        for original_offset, character in enumerate(content):
            folded_character = character.casefold()
            folded_parts.append(folded_character)
            folded_to_original.extend([original_offset] * len(folded_character))

        folded_content = "".join(folded_parts)
        match_ranges = []
        search_from = 0
        while folded_query:
            match_start = folded_content.find(folded_query, search_from)
            if match_start < 0:
                break
            match_end = match_start + len(folded_query)
            original_start = folded_to_original[match_start]
            original_end = folded_to_original[match_end - 1] + 1
            match_ranges.append((original_start, original_end))
            search_from = match_end
            # 從下一個原文字元開始，避免停在 casefold 展開的中途。
            while (search_from < len(folded_to_original)
                   and folded_to_original[search_from] < original_end):
                search_from += 1
        return match_ranges

    # 以字面子字串搜尋保留中的完整日誌，並標記每個不重疊命中。
    def _refresh_log_search(self, *_args):
        query = self.log_search_var.get()
        if query != self._log_search_last_query:
            self._clear_log_current_match()
            self._log_search_last_query = query
        self.log.tag_remove("log_search_match", "1.0", "end")
        self._log_search_highlights = []
        if not query:
            self._log_search_match_count = 0
            self._log_search_match_ranges = []
            self._log_search_match_ranges_valid = True
            self._update_log_search_controls()
            return

        content = self.log.get("1.0", "end-1c")
        match_ranges = self._find_log_match_ranges(content, query)
        self._log_search_match_ranges = match_ranges
        self._log_search_match_ranges_valid = True
        self._log_search_match_count = len(match_ranges)
        self._log_search_highlights = match_ranges[-MAX_LOG_HIGHLIGHTS:]
        self._add_log_tag_ranges("log_search_match", self._log_search_highlights)
        self._update_log_search_controls()

    # 清除搜尋文字與標記。
    def _clear_log_search(self):
        self.log_search_var.set("")
        self.log_search_entry.focus_set()

    # 目前畫面保留的完整紀錄複製到系統剪貼簿。
    def _copy_log(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.log.get("1.0", "end-1c"))
        except tk.TclError as error:
            messagebox.showerror("無法複製紀錄", f"無法寫入剪貼簿：{error}", parent=self.root)

    # 將目前保留的完整紀錄另存為 UTF-8 文字檔。
    def _save_log(self):
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="另存執行紀錄",
            defaultextension=".log",
            filetypes=(("Log 檔案", "*.log"), ("文字檔案", "*.txt"), ("所有檔案", "*.*")),
        )
        if not filename:
            return
        try:
            saving.write_text(filename, self.log.get("1.0", "end-1c"), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("無法儲存紀錄", f"無法寫入指定檔案：{exc}", parent=self.root)

    # 處理關閉請求，停止佇列並清理回放資源。
    def _on_close(self):
        if self.runner.is_running:
            if not messagebox.askyesno("實驗仍在執行", "停止目前實驗並關閉視窗？", parent=self.root):
                return
            self.closing = True
            self.replay_panel.close()
            self.results_panel.close()
            self.scatter_panel.close()
            self.runner.stop()
            self.status_var.set("正在停止並安全關閉…")
            return
        self.replay_panel.close()
        self.results_panel.close()
        self.scatter_panel.close()
        self._cancel_poll_events()
        self.root.destroy()

    # 重新讀取顯示名稱，並讓兩頁尚未完成的舊掃描失效。
    def _result_renamed(self, result):
        if self.closing:
            return
        self.results_panel.refresh()
        self.scatter_panel.refresh()

    # 切換離開回放頁時暫停播放。
    def _on_tab_changed(self, event=None):
        if self.main_notebook.select() != str(self.replay_tab):
            self.replay_panel.pause()
        if self.main_notebook.select() == str(self.scatter_tab):
            self.scatter_panel.ensure_loaded()


# 建立 Tk 主視窗並啟動實驗介面事件迴圈。
def main():
    root = tk.Tk()
    try:
        ExperimentGUI(root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("無法啟動實驗管理器", str(exc), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
