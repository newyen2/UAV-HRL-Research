"""Tk page for comparing experiment summaries as a scatter plot."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from gui import experiment_results as result_data
from gui.gui_style import SMALL_FONT, SMALL_BOLD_FONT, configure_gui_style


_METRIC_NAMES = tuple(result_data._METRICS)
_LABEL_MODES = ("全部顯示", "僅所選", "不顯示")
_LABEL_WIDTH = 140
_COLORS = result_data._COLORS


def _path_identity(path):
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = candidate.absolute()
    return str(candidate).casefold()


def _metric_values(result):
    """Cache the nine table summaries once for axis changes and tree rows."""
    values = {}
    for label, metric in result_data._METRICS.items():
        value = result_data._summary_value(result, metric)
        if value is not None and not math.isfinite(value):
            value = None
        values[label] = value
    return values


def _axis_range(values):
    """Return a padded range in scaled coordinates, safe for extreme floats."""
    scale = max((abs(value) for value in values), default=1.0)
    if scale == 0.0:
        scale = 1.0
    scaled = [value / scale for value in values]
    low = min(scaled)
    high = max(scaled)
    if low == high:
        padding = max(abs(low) * 0.05, 0.05)
    else:
        padding = (high - low) * 0.08
    return scale, low - padding, high + padding


def _axis_position(value, axis_range):
    scale, low, high = axis_range
    # Scale first so subtracting opposite-sign finite extremes cannot overflow.
    return ((value / scale) - low) / (high - low)


def _tick_text(value_scaled, scale):
    value = value_scaled * scale
    if math.isfinite(value):
        return result_data._format_number(value)
    return f"{value_scaled:.3g} × {scale:.3g}"


class ScatterPanel(ttk.Frame):
    """Browse output folders and show each experiment as one scatter point."""

    def __init__(self, parent, output_root, *, on_renamed=None):
        super().__init__(parent)
        configure_gui_style(self._root())
        self.output_root = Path(output_root).expanduser()
        self.on_renamed = on_renamed
        self._generation = 0
        self._closed = False
        self._events = queue.Queue()
        self._poll_after_id = None
        self._configure_after_id = None
        self._scan_pending = False
        self._pending_root_identity = None
        self._loaded_root_identity = None
        self._has_loaded = False
        self._updating_path = False
        self.results = []
        self._results_by_iid = {}
        self._iid_by_identity = {}
        self._identity_by_iid = {}
        self._summary_by_iid = {}
        self._canvas_item_to_iid = {}
        self._point_coordinates = {}
        self._selected_identity = None
        self._hidden_identities = set()
        self._load_issues = 0
        self._load_error = None

        self.path_var = tk.StringVar(master=self, value=str(self.output_root))
        self.x_metric_var = tk.StringVar(master=self, value="訓練 Reward")
        self.y_metric_var = tk.StringVar(master=self, value="評估 AoI")
        self.label_mode_var = tk.StringVar(master=self, value=_LABEL_MODES[0])
        self.x_reverse_var = tk.BooleanVar(master=self, value=False)
        self.y_reverse_var = tk.BooleanVar(master=self, value=False)
        self.metric_description_var = tk.StringVar(master=self)
        self.show_metric_help_var = tk.BooleanVar(master=self, value=False)
        self.status_var = tk.StringVar(
            master=self, value="尚未讀取實驗結果；切換至此頁時會載入。")
        self._path_trace_id = self.path_var.trace_add("write", self._path_entry_changed)
        self._build_ui()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="結果資料夾：").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.path_var).grid(
            row=0, column=1, sticky="ew", padx=5)
        ttk.Button(toolbar, text="選擇資料夾…", command=self._choose_root).grid(
            row=0, column=2, padx=(0, 5))
        ttk.Button(toolbar, text="重新整理", command=self.refresh).grid(
            row=0, column=3)

        controls = ttk.Frame(self, padding=(8, 2, 8, 5))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        ttk.Label(controls, text="X 軸：").grid(row=0, column=0, sticky="w")
        self.x_metric_box = ttk.Combobox(
            controls, textvariable=self.x_metric_var, values=_METRIC_NAMES,
            state="readonly", width=20)
        self.x_metric_box.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.x_metric_box.bind("<<ComboboxSelected>>", self._axis_changed)
        ttk.Label(controls, text="Y 軸：").grid(row=0, column=2, sticky="w")
        self.y_metric_box = ttk.Combobox(
            controls, textvariable=self.y_metric_var, values=_METRIC_NAMES,
            state="readonly", width=20)
        self.y_metric_box.grid(row=0, column=3, sticky="w", padx=(4, 0))
        self.y_metric_box.bind("<<ComboboxSelected>>", self._axis_changed)
        self.metric_description = ttk.Label(
            controls, textvariable=self.metric_description_var, anchor="w",
            justify="left", wraplength=900, style="Muted.TLabel")
        self.metric_description.grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.metric_description.grid_remove()
        self.metric_description.bind(
            "<Configure>",
            lambda event: self.metric_description.configure(
                wraplength=max(160, event.width - 8)),
        )
        ttk.Label(
            controls,
            text="累積 Wall time 取最新值；其餘取末 20 筆有效記錄的平均（不足 20 筆則取全部）。",
            anchor="w", style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(3, 0))

        direction_controls = ttk.Frame(controls)
        direction_controls.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Label(direction_controls, text="點名：").pack(side="left")
        self.label_mode_box = ttk.Combobox(
            direction_controls, textvariable=self.label_mode_var, values=_LABEL_MODES,
            state="readonly", width=10)
        self.label_mode_box.pack(side="left", padx=(4, 10))
        self.label_mode_box.bind("<<ComboboxSelected>>", self._label_mode_changed)
        ttk.Separator(direction_controls, orient="vertical").pack(
            side="left", fill="y", padx=(0, 5))
        self.x_reverse_check = ttk.Checkbutton(
            direction_controls, text="X 反向（高值在左）",
            variable=self.x_reverse_var, command=self._axis_direction_changed,
        )
        self.x_reverse_check.pack(side="left", padx=(8, 0))
        self.y_reverse_check = ttk.Checkbutton(
            direction_controls, text="Y 反向（高值在下）",
            variable=self.y_reverse_var, command=self._axis_direction_changed,
        )
        self.y_reverse_check.pack(side="left", padx=(8, 0))
        self.metric_help_toggle = ttk.Checkbutton(
            direction_controls, text="指標說明", variable=self.show_metric_help_var,
            command=self._toggle_metric_help)
        self.metric_help_toggle.pack(side="right", padx=(10, 0))

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 4))
        # 清單可水平捲動；避免用所有欄寬作初始寬度，把圖表擠得太窄。
        list_frame = ttk.Frame(panes, padding=(0, 0, 4, 0), width=330)
        list_frame.grid_propagate(False)
        chart_frame = ttk.Frame(panes, padding=(4, 0, 0, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(2, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        panes.add(list_frame, weight=3)
        panes.add(chart_frame, weight=5)

        actions = ttk.Frame(list_frame)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(actions, text="實驗清單（選取一項以標示）：").pack(side="left")
        self.rename_button = ttk.Button(
            actions, text="重新命名…", command=self._rename_selected,
            state="disabled")
        self.rename_button.pack(side="right")

        visibility_actions = ttk.Frame(list_frame)
        visibility_actions.grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.select_all_button = ttk.Button(
            visibility_actions, text="全選", command=lambda: self._set_all_visible(True))
        self.select_all_button.pack(side="left")
        self.deselect_all_button = ttk.Button(
            visibility_actions, text="全取消", command=lambda: self._set_all_visible(False))
        self.deselect_all_button.pack(side="left", padx=(5, 0))

        table_frame = ttk.Frame(list_frame)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("visible", "name", "method", "x", "y")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse")
        headers = {
            "name": ("實驗名稱", 205),
            "method": ("方法", 80),
            "x": ("X", 95),
            "y": ("Y", 95),
        }
        for column, (title, width) in headers.items():
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, minwidth=60, anchor="w")
        self.tree.column("x", anchor="e")
        self.tree.column("y", anchor="e")
        self.tree.heading("visible", text="顯示")
        self.tree.column("visible", width=46, minwidth=46, stretch=False, anchor="center")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Button-1>", self._tree_clicked)
        self.tree.bind("<space>", self._toggle_focused_visibility)

        self.canvas = tk.Canvas(
            chart_frame, background="white", highlightthickness=1,
            highlightbackground="#cccccc", width=520, height=440)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.canvas.bind("<Button-1>", self._canvas_clicked)

        self.status_label = ttk.Label(
            self, textvariable=self.status_var, anchor="w", padding=(10, 3),
            style="Muted.TLabel", justify="left")
        self.status_label.grid(row=3, column=0, sticky="ew")
        self.status_label.bind("<Configure>", lambda event: self.status_label.configure(
            wraplength=max(160, event.width - 24)))
        self._update_metric_description()

    def _toggle_metric_help(self):
        if self.show_metric_help_var.get():
            self.metric_description.grid()
        else:
            self.metric_description.grid_remove()

    def _path_entry_changed(self, *_args):
        if self._closed or self._updating_path:
            return
        try:
            path_text = self.path_var.get().strip()
            if path_text:
                identity = _path_identity(path_text)
            else:
                identity = ""
        except (OSError, ValueError):
            identity = self.path_var.get().strip().casefold()
        current_identity = self._pending_root_identity
        if current_identity is None:
            current_identity = self._loaded_root_identity
        if identity == current_identity:
            return
        self._generation += 1
        self._scan_pending = False
        self._pending_root_identity = None
        if identity != self._loaded_root_identity:
            self._has_loaded = False
            self._clear_loaded_results()
            self._draw_scatter()
        self.status_var.set("結果資料夾已變更，請按重新整理載入。")

    def set_root(self, path):
        """Set the input folder without starting a scan."""
        if self._closed:
            return
        new_root = Path(path).expanduser()
        new_identity = _path_identity(new_root)
        old_identity = _path_identity(self.output_root)
        self.output_root = new_root
        if self.path_var.get() != str(new_root):
            self._updating_path = True
            try:
                self.path_var.set(str(new_root))
            finally:
                self._updating_path = False
        if new_identity != old_identity:
            self._generation += 1
            self._scan_pending = False
            self._pending_root_identity = None
        if new_identity != self._loaded_root_identity:
            self._has_loaded = False
            self._clear_loaded_results()
            self.status_var.set("結果資料夾已變更；切換至此頁或按重新整理載入。")
            self._draw_scatter()

    def ensure_loaded(self):
        """Load the current folder on first use or after changing folders."""
        if self._closed:
            return
        entry = self.path_var.get().strip()
        if entry:
            root = Path(entry).expanduser()
        else:
            root = self.output_root
        identity = _path_identity(root)
        if (not self._scan_pending
                and (not self._has_loaded or self._loaded_root_identity != identity)):
            self.refresh()

    def _choose_root(self):
        if self._closed:
            return
        try:
            initial = str(Path(self.path_var.get()).expanduser())
        except (OSError, ValueError):
            initial = str(self.output_root)
        selected = filedialog.askdirectory(
            parent=self.winfo_toplevel(), initialdir=initial,
            title="選擇結果資料夾")
        if selected:
            self.set_root(selected)
            self.refresh()

    def refresh(self):
        if self._closed:
            return
        path_text = self.path_var.get().strip()
        if path_text:
            root = Path(path_text).expanduser()
        else:
            root = self.output_root
        self.output_root = root
        root_identity = _path_identity(root)
        self._generation += 1
        generation = self._generation
        self._scan_pending = True
        self._pending_root_identity = root_identity
        if root_identity != self._loaded_root_identity:
            self._has_loaded = False
            self._clear_loaded_results()
            self._draw_scatter()
        self.status_var.set("正在讀取歷史結果…")
        if self._poll_after_id is None:
            self._poll_after_id = self.after(80, self._poll_events)
        events = self._events

        def work():
            try:
                results, issues = result_data.load_results(root)
                error = None
            except Exception as caught:
                results, issues = [], 1
                error = str(caught)
                if not error:
                    error = type(caught).__name__
            events.put((generation, root_identity, results, issues, error))

        try:
            threading.Thread(
                target=work, name="experiment-scatter-scan", daemon=True).start()
        except Exception as error:
            self._scan_pending = False
            self._pending_root_identity = None
            self.status_var.set(f"無法啟動結果讀取：{error}")

    def _poll_events(self):
        if self._closed:
            return
        while True:
            try:
                generation, root_identity, results, issues, error = self._events.get_nowait()
            except queue.Empty:
                break
            if self._closed or generation != self._generation:
                continue
            visible = self.path_var.get().strip()
            if visible:
                visible_root = Path(visible).expanduser()
            else:
                visible_root = self.output_root
            if _path_identity(visible_root) != root_identity:
                continue
            if root_identity != self._pending_root_identity:
                continue
            self._install_results(results, issues, error)
        if not self._closed:
            try:
                self._poll_after_id = self.after(80, self._poll_events)
            except tk.TclError:
                self._poll_after_id = None

    def _install_results(self, results, issues, error):
        selected_identity = self._selected_identity_from_tree()
        self.results = list(results)
        self._results_by_iid.clear()
        self._summary_by_iid.clear()
        self._iid_by_identity.clear()
        self._identity_by_iid.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for index, result in enumerate(self.results):
            identity = _path_identity(result["path"])
            iid = str(index)
            self._results_by_iid[iid] = result
            self._iid_by_identity[identity] = iid
            self._identity_by_iid[iid] = identity
            self._summary_by_iid[iid] = _metric_values(result)
        # 用來源路徑保存勾選狀態，重新命名或重整後即使列號改變也不會勾回。
        self._hidden_identities.intersection_update(self._iid_by_identity)
        self._loaded_root_identity = self._pending_root_identity
        self._pending_root_identity = None
        self._scan_pending = False
        self._has_loaded = True
        self._selected_identity = selected_identity
        self._load_issues = issues
        self._load_error = error
        self._render_table(restore_identity=selected_identity)
        self._draw_scatter()
        self._update_plot_status()

    def _update_plot_status(self):
        if self._scan_pending:
            self.status_var.set("正在讀取歷史結果…")
            return
        if self._load_error:
            self.status_var.set(f"讀取失敗：{self._load_error}")
            return
        valid_count = len(self._valid_points())
        checked_count = len(self.results) - len(self._hidden_identities)
        skipped = checked_count - valid_count
        if not self.results:
            if self._load_issues:
                self.status_var.set("沒有可用結果；部分檔案或資料夾無法讀取。")
            else:
                self.status_var.set("此資料夾沒有可用的結果 CSV。")
            return
        status = (f"已讀取 {len(self.results)} 項實驗；勾選 {checked_count} 項，"
                  f"散布圖顯示 {valid_count} 點。")
        if skipped:
            status += f" {skipped} 項因 X 或 Y 指標缺少有限資料而略過。"
        if self._load_issues:
            status += f" 另略過 {self._load_issues} 個無效或無法讀取的項目。"
        self.status_var.set(status)

    def _clear_loaded_results(self):
        self.results = []
        self._results_by_iid.clear()
        self._summary_by_iid.clear()
        self._iid_by_identity.clear()
        self._identity_by_iid.clear()
        self._selected_identity = None
        self._hidden_identities.clear()
        self._load_issues = 0
        self._load_error = None
        if hasattr(self, "tree"):
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            self._update_rename_button()

    def _selected_identity_from_tree(self):
        if hasattr(self, "tree"):
            selection = self.tree.selection()
        else:
            selection = ()
        if selection:
            return self._identity_by_iid.get(selection[0])
        else:
            return self._selected_identity

    def _render_table(self, restore_identity=None):
        x_name = self.x_metric_var.get()
        y_name = self.y_metric_var.get()
        for iid, result in self._results_by_iid.items():
            values = self._summary_by_iid[iid]
            if self._identity_by_iid[iid] in self._hidden_identities:
                checked_text = "☐"
            else:
                checked_text = "☑"
            row_values = (
                checked_text,
                result_data._result_display_name(result),
                result.get("method", ""),
                result_data._format_number(values.get(x_name)),
                result_data._format_number(values.get(y_name)),
            )
            if self.tree.exists(iid):
                self.tree.item(iid, values=row_values)
            else:
                self.tree.insert("", "end", iid=iid, values=row_values)
        if restore_identity in self._iid_by_identity:
            iid = self._iid_by_identity[restore_identity]
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self._selected_identity = restore_identity
        else:
            self._selected_identity = None
        self._update_rename_button()

    def _axis_changed(self, _event=None):
        if self._closed:
            return
        self._update_metric_description()
        selected_identity = self._selected_identity_from_tree()
        self._render_table(restore_identity=selected_identity)
        self._draw_scatter()
        self._update_plot_status()

    def _axis_direction_changed(self):
        if self._closed:
            return
        self._draw_scatter()

    def _label_mode_changed(self, _event=None):
        if self._closed:
            return
        self._draw_scatter()

    def _tree_clicked(self, event):
        if (self._closed or self.tree.identify_region(event.x, event.y) != "cell"
                or self.tree.identify_column(event.x) != "#1"):
            return
        iid = self.tree.identify_row(event.y)
        if iid in self._results_by_iid:
            self._toggle_visibility(iid)
            return "break"

    def _toggle_focused_visibility(self, _event=None):
        self._toggle_visibility(self.tree.focus())
        return "break"

    def _toggle_visibility(self, iid):
        if self._closed or iid not in self._results_by_iid:
            return
        identity = self._identity_by_iid[iid]
        if identity in self._hidden_identities:
            self._hidden_identities.remove(identity)
            checked_text = "☑"
        else:
            self._hidden_identities.add(identity)
            checked_text = "☐"
        self.tree.set(iid, "visible", checked_text)
        self._draw_scatter()
        self._update_plot_status()

    def _set_all_visible(self, visible):
        if self._closed:
            return
        if visible:
            self._hidden_identities.clear()
        else:
            self._hidden_identities.update(self._identity_by_iid.values())
        self._render_table(restore_identity=self._selected_identity_from_tree())
        self._draw_scatter()
        self._update_plot_status()

    def _update_metric_description(self):
        x_name = self.x_metric_var.get()
        y_name = self.y_metric_var.get()
        x_description = result_data._METRIC_DESCRIPTIONS.get(x_name, "")
        y_description = result_data._METRIC_DESCRIPTIONS.get(y_name, "")
        self.metric_description_var.set(
            f"X — {x_name}：{x_description}\nY — {y_name}：{y_description}")

    def _selection_changed(self, _event=None):
        selection = self.tree.selection()
        if selection:
            self._selected_identity = self._identity_by_iid.get(selection[0])
        else:
            self._selected_identity = None
        self._update_rename_button()
        self._draw_scatter()

    def _update_rename_button(self):
        if not hasattr(self, "rename_button"):
            return
        enabled = len(self.tree.selection()) == 1 and not self._closed
        if enabled:
            state = "normal"
        else:
            state = "disabled"
        self.rename_button.configure(state=state)

    def _valid_points(self):
        x_name = self.x_metric_var.get()
        y_name = self.y_metric_var.get()
        points = []
        for iid, result in self._results_by_iid.items():
            if self._identity_by_iid[iid] in self._hidden_identities:
                continue
            values = self._summary_by_iid[iid]
            x_value = values.get(x_name)
            y_value = values.get(y_name)
            if x_value is None or y_value is None:
                continue
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            points.append((iid, result, x_value, y_value))
        return points

    def _schedule_redraw(self, _event=None):
        if self._closed or self._configure_after_id is not None:
            return
        try:
            self._configure_after_id = self.after(40, self._redraw_after_configure)
        except tk.TclError:
            self._configure_after_id = None

    def _redraw_after_configure(self):
        self._configure_after_id = None
        if not self._closed:
            self._draw_scatter()

    def _draw_scatter(self):
        if self._closed or not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        self._canvas_item_to_iid.clear()
        self._point_coordinates.clear()
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        points = self._valid_points()
        if not points:
            if self.results and len(self._hidden_identities) == len(self.results):
                message = "尚未勾選任何實驗，請勾選清單項目或按「全選」。"
            else:
                message = "目前沒有同時具備有限 X、Y 數值的實驗。"
            self.canvas.create_text(
                width / 2, height / 2, text=message, fill="#666666",
                width=max(120, width - 40), justify="center")
            return

        x_name = self.x_metric_var.get()
        y_name = self.y_metric_var.get()
        x_values = [point[2] for point in points]
        y_values = [point[3] for point in points]
        x_range = _axis_range(x_values)
        y_range = _axis_range(y_values)
        axis_left = 72.0
        axis_right = max(axis_left + 120.0, width - 24.0)
        axis_top = 38.0
        label_half = min(_LABEL_WIDTH / 2, max(30.0, (axis_right - axis_left - 80.0) / 2))
        data_left = axis_left + label_half
        data_right = axis_right - label_half
        if data_right <= data_left:
            data_left = axis_left + 20.0
            data_right = axis_right - 20.0

        label_height = max(
            self._measure_label_height(result)
            for _iid, result, _x, _y in points)
        label_height += 4.0
        point_bottom = height - label_height - 82.0
        if point_bottom <= axis_top + 24.0:
            point_bottom = axis_top + 24.0
        axis_bottom = point_bottom

        for tick_index in range(5):
            fraction = tick_index / 4
            x = data_left + fraction * (data_right - data_left)
            y = axis_top + fraction * (axis_bottom - axis_top)
            if self.x_reverse_var.get():
                x_tick_value = x_range[2] * (1 - fraction) + x_range[1] * fraction
            else:
                x_tick_value = x_range[1] * (1 - fraction) + x_range[2] * fraction
            if self.y_reverse_var.get():
                y_tick_value = y_range[1] * (1 - fraction) + y_range[2] * fraction
            else:
                y_tick_value = y_range[2] * (1 - fraction) + y_range[1] * fraction
            self.canvas.create_line(
                x, axis_top, x, axis_bottom, fill="#e7e7e7", tags=("grid",))
            self.canvas.create_line(
                data_left, y, data_right, y, fill="#e7e7e7", tags=("grid",))
            self.canvas.create_text(
                x, axis_bottom + label_height + 10,
                text=_tick_text(x_tick_value, x_range[0]),
                anchor="n", fill="#555555", font=SMALL_FONT,
                tags=("tick", f"x-tick-{tick_index}"))
            self.canvas.create_text(
                data_left - 8, y, text=_tick_text(y_tick_value, y_range[0]),
                anchor="e", fill="#555555", font=SMALL_FONT,
                tags=("tick", f"y-tick-{tick_index}"))
        self.canvas.create_line(
            data_left, axis_bottom, data_right, axis_bottom,
            fill="#555555", width=1.2, tags=("axis",))
        self.canvas.create_line(
            data_left, axis_top, data_left, axis_bottom,
            fill="#555555", width=1.2, tags=("axis",))
        self.canvas.create_text(
            (data_left + data_right) / 2, height - 17,
            text=x_name, fill="#222222", font=SMALL_BOLD_FONT,
            tags=("axis-title",))
        self.canvas.create_text(
            16, (axis_top + axis_bottom) / 2, text=y_name, angle=90,
            fill="#222222", font=SMALL_BOLD_FONT,
            tags=("axis-title",))

        label_mode = self.label_mode_var.get()
        for iid, result, x_value, y_value in points:
            x_fraction = _axis_position(x_value, x_range)
            y_fraction = _axis_position(y_value, y_range)
            if self.x_reverse_var.get():
                x_fraction = 1.0 - x_fraction
            if self.y_reverse_var.get():
                # Canvas Y grows downward, so this puts high values at the bottom.
                y_fraction = 1.0 - y_fraction
            x = data_left + x_fraction * (data_right - data_left)
            y = axis_top + (1.0 - y_fraction) * (axis_bottom - axis_top)
            self._point_coordinates[iid] = (x, y)
            selected = self._identity_by_iid.get(iid) == self._selected_identity
            # 配色依原清單順序，隱藏其他實驗時不改變仍顯示點的顏色。
            color = _COLORS[self.tree.index(iid) % len(_COLORS)]
            if selected:
                radius = 7.0
                outline = "#171717"
                marker_width = 2.5
                label_font = SMALL_BOLD_FONT
            else:
                radius = 4.8
                outline = color
                marker_width = 1.0
                label_font = SMALL_FONT
            point_item = self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=color, outline=outline, width=marker_width,
                tags=("scatter-point", f"result:{iid}"),
            )
            self._canvas_item_to_iid[point_item] = iid
            show_label = label_mode == "全部顯示"
            if label_mode == "僅所選":
                show_label = self._identity_by_iid.get(iid) == self._selected_identity
            if show_label:
                display_name = result_data._result_display_name(result)
                label_item = self.canvas.create_text(
                    x, y + 9, text=display_name, anchor="n", width=_LABEL_WIDTH,
                    justify="center", fill="#222222",
                    font=label_font,
                    tags=("scatter-label", f"result:{iid}"),
                )
                self._canvas_item_to_iid[label_item] = iid

        # Keep the selected point above coincident dots and labels.
        selected_iid = self._iid_by_identity.get(self._selected_identity)
        if selected_iid in self._point_coordinates:
            for item_id, item_iid in self._canvas_item_to_iid.items():
                if item_iid == selected_iid:
                    self.canvas.tag_raise(item_id)

    def _measure_label_height(self, result):
        """Measure Tk's actual wrapping so long labels fit below low points."""
        item_id = self.canvas.create_text(
            0, 0, text=result_data._result_display_name(result), anchor="n",
            width=_LABEL_WIDTH, justify="center",
            font=SMALL_BOLD_FONT, tags=("measure-label",),
        )
        bounds = self.canvas.bbox(item_id)
        self.canvas.delete(item_id)
        if bounds is None:
            return 15.0
        return max(15.0, float(bounds[3] - bounds[1]))

    def _canvas_clicked(self, event):
        if self._closed:
            return
        overlap = self.canvas.find_overlapping(
            event.x - 4, event.y - 4, event.x + 4, event.y + 4)
        for item_id in reversed(overlap):
            iid = self._canvas_item_to_iid.get(item_id)
            if iid is not None and iid in self._results_by_iid:
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)
                self._selected_identity = self._identity_by_iid.get(iid)
                self._update_rename_button()
                self._draw_scatter()
                return

    def _rename_selected(self):
        selection = self.tree.selection()
        if self._closed or len(selection) != 1:
            return
        iid = selection[0]
        result = self._results_by_iid.get(iid)
        if result is None:
            return
        current_name = str(result.get("name", ""))
        proposed = simpledialog.askstring(
            "重新命名實驗", "新的實驗名稱：", initialvalue=current_name,
            parent=self.winfo_toplevel())
        if proposed is None:
            return
        new_name = proposed.strip()
        if not new_name:
            messagebox.showerror(
                "無法重新命名", "實驗名稱不能為空。", parent=self.winfo_toplevel())
            return
        if new_name == current_name:
            return
        try:
            saved_name = result_data.rename_result(result, new_name)
        except Exception as error:
            self.status_var.set(f"重新命名失敗：{error}")
            messagebox.showerror(
                "重新命名失敗", str(error), parent=self.winfo_toplevel())
            return

        # Invalidate any scan that began before the rename could be persisted.
        self._generation += 1
        self._scan_pending = False
        self._pending_root_identity = None
        if isinstance(saved_name, str) and saved_name:
            new_name = saved_name
        result["name"] = new_name
        identity = self._identity_by_iid.get(iid)
        self._selected_identity = identity
        self._render_table(restore_identity=identity)
        self._draw_scatter()
        self.status_var.set(f"已重新命名為「{new_name}」。")
        if self.on_renamed is not None:
            self.on_renamed(result)
        else:
            self.refresh()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._scan_pending = False
        self._pending_root_identity = None
        for attribute in ("_poll_after_id", "_configure_after_id"):
            after_id = getattr(self, attribute)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
                setattr(self, attribute, None)
        try:
            self.path_var.trace_remove("write", self._path_trace_id)
        except (tk.TclError, AttributeError):
            pass
        try:
            self.canvas.unbind("<Configure>")
            self.canvas.unbind("<Button-1>")
        except (tk.TclError, AttributeError):
            pass

    def _on_destroy(self, event):
        if event.widget is self:
            self.close()

