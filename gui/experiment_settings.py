"""Reusable, presentation-only editor for experiment settings."""
from __future__ import annotations

import copy
import json
import math
import tkinter as tk
from collections.abc import Mapping, Sequence
from tkinter import ttk
from typing import Any

from gui.gui_format import format_edit_value
from gui.gui_style import BODY_FONT, SMALL_FONT, SMALL_BOLD_FONT, configure_gui_style
from gui.setting_labels import (
    OPTION_LABELS,
    field_help,
    field_label,
    group_help,
    group_label,
    ordered_fields,
)


_COMMON_GROUPS = frozenset({"TRAIN", "ENV", "FUNCTIONAL", "FIXED", "EVAL", "RECORD"})
_TAB_GROUPS = {
    "訓練": ("TRAIN",),
    "環境": ("ENV", "FUNCTIONAL", "FIXED"),
    "方法": (),
    "評估與輸出": ("EVAL", "RECORD"),
}
_CONTROLLER_PATHS = {
    ("ENV", "arrival_model"),
    ("ENV", "communication_model"),
    ("HRL_UPPER", "reward_mode"),
    ("HRL_PPO_UPPER", "reward_mode"),
    ("HRL_LOWER", "algorithm"),
    ("HRL", "pretrain_batches"),
    ("HRL_PPO_LOWER", "algorithm"),
    ("HRL_PPO_LOWER", "pretrain_batches"),
    ("FUNCTIONAL", "risk"),
    ("FUNCTIONAL", "random_device_coord"),
    ("FUNCTIONAL", "random_UAV_coord"),
}
_BUDGET_PATHS = {
    ("TRAIN", "episodes"), ("HRL", "upper_fraction"), ("HRL", "lower_fraction"),
    ("HRL_PPO", "navigation_fraction"),
}
_PRESERVED_INACTIVE_PATHS = {
    ("HRL_UPPER", "risk_penalty_weight"),
    ("HRL", "fragment_steps"),
    ("HRL_UPPER", "gamma"),
    ("HRL_UPPER", "average_reward_lr"),
    ("HRL_PPO_UPPER", "risk_penalty_weight"),
    ("HRL_PPO", "fragment_steps"),
    ("HRL_PPO_UPPER", "gamma"),
    ("HRL_PPO_UPPER", "average_reward_lr"),
}


class SettingsEditor(ttk.Frame):
    """A four-tab editor that keeps raw drafts while filtering rows in place."""

    def __init__(self, parent, core):
        super().__init__(parent)
        configure_gui_style(self._root())
        self.core = core
        self.method = "DQN"
        self.field_vars: dict[tuple[str, str], tuple[tk.Variable, str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self._row_by_path: dict[tuple[str, str], dict[str, Any]] = {}
        self._tab_canvases: dict[str, tk.Canvas] = {}
        self._tab_bodies: dict[str, ttk.Frame] = {}
        self._tab_rows: dict[str, list[dict[str, Any]]] = {}
        self._tab_empty_labels: dict[str, ttk.Label] = {}
        self._group_records: dict[str, list[dict[str, Any]]] = {}
        self._project_defaults: dict = {}
        self._selected_path: tuple[str, str] | None = None
        self._inactive_paths: dict[tuple[str, str], str] = {}
        self._field_traces = []
        self._pending_scroll = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self, padding=(0, 0, 0, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="搜尋").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.search_var = tk.StringVar(master=self, value="")
        self.search_entry = ttk.Entry(toolbar, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.clear_button = ttk.Button(toolbar, text="清除", width=5, command=self._clear_search)
        self.clear_button.grid(row=0, column=2, padx=(5, 0))
        self.show_inactive_var = tk.BooleanVar(master=self, value=False)
        self.show_inactive_button = ttk.Checkbutton(
            toolbar, text="顯示不適用欄位", variable=self.show_inactive_var,
            command=self.apply_filter)
        self.show_inactive_button.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.match_count_var = tk.StringVar(master=self, value="顯示 0 個｜適用 0 個｜隱藏 0 個")
        ttk.Label(toolbar, textvariable=self.match_count_var).grid(
            row=1, column=2, sticky="e", pady=(3, 0))
        self._search_trace = self.search_var.trace_add("write", self.apply_filter)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.help_frame = ttk.LabelFrame(self, text="欄位說明", padding=(7, 4))
        self.help_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        self.help_frame.columnconfigure(0, weight=1)
        self.help_title_var = tk.StringVar(master=self, value="選取任一欄位查看用途與預設值。")
        self.help_title = ttk.Label(
            self.help_frame, textvariable=self.help_title_var,
            font=SMALL_BOLD_FONT, anchor="w")
        self.help_title.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.help_text_var = tk.StringVar(master=self, value="")
        # 預設座標可能很長；固定高度保留設定頁空間，完整文字仍可捲動閱讀。
        self.help_text = tk.Text(
            self.help_frame, height=3, width=1, wrap="word", state="disabled",
            font=BODY_FONT, borderwidth=0, highlightthickness=0)
        self.help_text.grid(row=1, column=0, sticky="nsew")
        help_scrollbar = ttk.Scrollbar(
            self.help_frame, orient="vertical", command=self.help_text.yview)
        help_scrollbar.grid(row=1, column=1, sticky="ns")
        self.help_text.configure(yscrollcommand=help_scrollbar.set)
        self._shown_help_text = ""
        self._help_trace = self.help_text_var.trace_add("write", self._refresh_help_text)
        self.help_frame.bind("<Configure>", self._resize_help)

    def _clear_field_traces(self):
        for variable, trace_id in self._field_traces:
            variable.trace_remove("write", trace_id)
        self._field_traces.clear()

    def _cancel_pending_scroll(self):
        if self._pending_scroll is not None:
            self.after_cancel(self._pending_scroll)
            self._pending_scroll = None

    def destroy(self):
        # Tcl callback 會持有 editor；銷毀前解除自建 trace 與延後捲動。
        self._cancel_pending_scroll()
        self._clear_field_traces()
        for variable, trace_id in ((self.search_var, self._search_trace),
                                   (self.help_text_var, self._help_trace)):
            if trace_id is not None:
                variable.trace_remove("write", trace_id)
        self._search_trace = None
        self._help_trace = None
        super().destroy()

    def _refresh_help_text(self, *_args):
        text = self.help_text_var.get()
        if text == self._shown_help_text:
            return
        self.help_text.configure(state="normal")
        self.help_text.delete("1.0", "end")
        self.help_text.insert("1.0", text)
        self.help_text.configure(state="disabled")
        self.help_text.yview_moveto(0)
        self._shown_help_text = text

    # 使用正式預設值作為型別原型，並以呼叫端的快照覆寫目前顯示值。
    def render(self, method: str, settings: dict) -> None:
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        self.method = method
        self._project_defaults = copy.deepcopy(self.core.default_settings(method))
        display_settings = copy.deepcopy(self._project_defaults)
        source_settings = copy.deepcopy(dict(settings))
        for group, values in source_settings.items():
            if (group in display_settings and isinstance(display_settings[group], dict)
                    and isinstance(values, Mapping)):
                display_settings[group].update(copy.deepcopy(dict(values)))
            else:
                display_settings[group] = copy.deepcopy(values)

        self._cancel_pending_scroll()
        self._clear_field_traces()
        for tab_id in self.notebook.tabs():
            tab = self.notebook.nametowidget(tab_id)
            self.notebook.forget(tab_id)
            tab.destroy()
        self.field_vars.clear()
        self.rows.clear()
        self._row_by_path.clear()
        self._tab_canvases.clear()
        self._tab_bodies.clear()
        self._tab_rows.clear()
        self._tab_empty_labels.clear()
        self._group_records.clear()
        self._inactive_paths.clear()

        tab_names = tuple(_TAB_GROUPS)
        for tab_name in tab_names:
            tab, canvas, body, empty_label = self._make_tab(tab_name)
            self.notebook.add(tab, text=tab_name)
            self._tab_canvases[tab_name] = canvas
            self._tab_bodies[tab_name] = body
            self._tab_rows[tab_name] = []
            self._group_records[tab_name] = []
            self._tab_empty_labels[tab_name] = empty_label

        method_groups = [
            group for group in display_settings
            if group not in _COMMON_GROUPS
        ]
        preferred_method_order = (
            "HRL", "HRL_UPPER", "HRL_LOWER",
            "HRL_PPO", "HRL_PPO_UPPER", "HRL_PPO_LOWER",
        )
        ordered_method_groups = [
            group for group in preferred_method_order if group in method_groups
        ]
        ordered_method_groups.extend(sorted(set(method_groups) - set(ordered_method_groups)))
        group_order = {
            **{group: index for index, group in enumerate(("TRAIN", "ENV", "FUNCTIONAL", "FIXED", "EVAL", "RECORD"))},
            **{group: index for index, group in enumerate(ordered_method_groups)},
        }
        for group in sorted(display_settings, key=lambda name: (group_order.get(name, 100), name)):
            group_values = display_settings[group]
            if not isinstance(group_values, Mapping):
                continue
            tab_name = self._tab_for_group(group)
            self._add_group(tab_name, group, group_values)

        self._selected_path = next(
            (record["path"] for record in self._tab_rows.get("訓練", [])), None)
        self._update_help()
        self.apply_filter()

    # 建立可獨立捲動的頁籤，滾輪繫結只作用於本編輯器。
    def _make_tab(self, tab_name: str):
        tab = ttk.Frame(self.notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        canvas = tk.Canvas(tab, highlightthickness=0, takefocus=False)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=(5, 5, 5, 8))
        body.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=body, anchor="nw", tags="body")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        body.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.bind("<Configure>", lambda event, c=canvas, item=window: c.itemconfigure(item, width=event.width))
        empty = ttk.Label(
            body, text="目前沒有符合的設定欄位。可清除搜尋，或顯示不適用欄位。",
            foreground="#666666", wraplength=360, justify="left")
        canvas.bind(
            "<Configure>",
            lambda event, label=empty: label.configure(wraplength=max(120, event.width - 20)),
            add="+",
        )
        self._bind_scoped_wheel(canvas, canvas)
        return tab, canvas, body, empty

    # 以四個工作流程頁籤分配核心設定群組。
    @staticmethod
    def _tab_for_group(group: str) -> str:
        for tab_name, groups in _TAB_GROUPS.items():
            if group in groups:
                return tab_name
        return "方法"

    # 在指定頁籤建立群組與穩定欄位列。
    def _add_group(self, tab_name: str, group: str, values: Mapping) -> None:
        body = self._tab_bodies[tab_name]
        group_box = ttk.LabelFrame(body, text=group_label(group), padding=(7, 5))
        group_box.pack(fill="x", expand=True, pady=(0, 7))
        group_box.columnconfigure(0, weight=1)
        description = group_help(group)
        if description:
            group_help_label = ttk.Label(
                group_box, text=description, foreground="#555555",
                justify="left", anchor="w", wraplength=420)
            group_help_label.pack(fill="x", pady=(0, 3))
            group_box.bind(
                "<Configure>",
                lambda event, label=group_help_label: label.configure(
                    wraplength=max(140, event.width - 24)))
        else:
            group_help_label = None

        group_record = {
            "widget": group_box,
            "rows": [],
            "pack_options": group_box.pack_info(),
            "visible": True,
        }
        self._group_records[tab_name].append(group_record)
        field_keys = tuple(str(key) for key in values)
        for key in ordered_fields(group, field_keys):
            if key not in values:
                continue
            project_default = self._get_value(self._project_defaults, (group, key), values[key])
            record = self._add_field_row(
                group_box, tab_name, group, key, values[key], project_default,
                description)
            group_record["rows"].append(record)
            self._tab_rows[tab_name].append(record)
            self.rows.append(record)
            self._row_by_path[(group, key)] = record
        self._bind_scoped_wheel(group_box, self._tab_canvases[tab_name])

    # 建立欄位輸入與短標籤，並保留完整路徑供搜尋及錯誤定位。
    def _add_field_row(self, parent, tab_name, group, key, value, prototype, group_description):
        path = (group, key)
        row = ttk.Frame(parent, padding=(0, 3))
        row.pack(fill="x", expand=True)
        row.columnconfigure(1, weight=1)

        label_block = ttk.Frame(row)
        label_block.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        label_block.columnconfigure(0, weight=1)
        title_text = field_label(path)
        title_label = ttk.Label(label_block, text=title_text, anchor="w", justify="left")
        title_label.grid(row=0, column=0, sticky="ew")
        path_label = ttk.Label(
            label_block, text=".".join(path), foreground="#666666",
            font=SMALL_FONT, anchor="w", wraplength=142, justify="left")
        path_label.grid(row=1, column=0, sticky="ew")
        reason_label = ttk.Label(
            label_block, text="", foreground="#7a4c00", font=SMALL_FONT,
            anchor="w", justify="left", wraplength=180)
        reason_label.grid(row=2, column=0, sticky="ew")

        option_values = self.core.setting_options(*path)
        option_map = OPTION_LABELS.get(path, {})
        if isinstance(value, bool):
            variable = tk.BooleanVar(master=self, value=value)
            control = ttk.Checkbutton(row, variable=variable)
            kind = "bool"
            initial_raw = value
            normal_state = "normal"
        elif option_values:
            display_values = tuple(option_map.get(option, option) for option in option_values)
            variable = tk.StringVar(master=self, value=option_map.get(value, str(value)))
            control = ttk.Combobox(
                row, textvariable=variable, values=display_values,
                state="readonly", width=1)
            kind = "option"
            initial_raw = variable.get()
            normal_state = "readonly"
        else:
            initial_raw = value if isinstance(value, str) else format_edit_value(value)
            variable = tk.StringVar(master=self, value=initial_raw)
            control = ttk.Entry(row, textvariable=variable)
            if isinstance(value, (list, tuple, dict)) or value is None:
                kind = "json"
            else:
                kind = "scalar"
            normal_state = "normal"
        # 留出窄視窗的下拉箭頭與輸入末端；標籤自然換行決定列高。
        row.columnconfigure(0, weight=0, minsize=150)
        row.columnconfigure(1, weight=1, minsize=90)
        title_label.configure(wraplength=142)
        reason_label.configure(wraplength=142)
        control.grid(row=0, column=1, sticky="ew")
        self.field_vars[path] = (variable, kind, copy.deepcopy(prototype))

        help_text = field_help(path)
        search_parts = (
            title_text, ".".join(path), help_text, group_label(group),
            group_description, *[str(item) for item in option_values],
            *[str(item) for item in option_map.values()],
        )
        record = {
            "widget": row,
            "path": path,
            "search_text": " ".join(str(part) for part in search_parts).casefold(),
            "visible": True,
            "inactive_reason": None,
            "tab": tab_name,
            "group": group,
            "group_widget": parent,
            "label_widget": title_label,
            "path_widget": path_label,
            "reason_widget": reason_label,
            "label_block": label_block,
            "input_widget": control,
            "variable": variable,
            "kind": kind,
            "prototype": copy.deepcopy(prototype),
            "normal_state": normal_state,
            "help_text": help_text,
            "initial_raw": initial_raw,
            "reason_text": "",
            "reason_visible": True,
            "control_disabled": False,
            "pack_options": row.pack_info(),
        }
        self._bind_field_focus(record)
        if path in _CONTROLLER_PATHS:
            trace_id = variable.trace_add("write", self.apply_filter)
            self._field_traces.append((variable, trace_id))
        if path in _BUDGET_PATHS:
            trace_id = variable.trace_add("write", self._budget_changed)
            self._field_traces.append((variable, trace_id))
        return record

    def _budget_changed(self, *_args):
        if self._selected_path in _BUDGET_PATHS:
            self._update_help()

    def _hrl_budget_preview(self):
        if self.method == "HRL_PPO" and self._selected_path in {
                ("TRAIN", "episodes"), ("HRL_PPO", "navigation_fraction")}:
            try:
                total = self.read_field(("TRAIN", "episodes"))
                navigation_fraction = self.read_field(("HRL_PPO", "navigation_fraction"))
                if (total <= 0 or not math.isfinite(navigation_fraction)
                        or not 0 <= navigation_fraction < 1):
                    raise ValueError("invalid navigation budget")
                navigation = int(total * navigation_fraction)
                joint = total - navigation
            except (ValueError, OverflowError):
                return "階段回合預覽：請填入有效總回合數與導航比例；導航回合向下取整，並保留至少一個聯合回合。"
            return (f"階段回合預覽：導航 {navigation:,}、聯合 {joint:,} 回合。"
                    "導航回合數向下取整，示範初始化不占環境訓練回合。")
        if self.method != "HRL" or self._selected_path not in _BUDGET_PATHS:
            return ""
        try:
            total = self.read_field(("TRAIN", "episodes"))
            upper_fraction = self.read_field(("HRL", "upper_fraction"))
            lower_fraction = self.read_field(("HRL", "lower_fraction"))
            if (total <= 0 or not math.isfinite(upper_fraction)
                    or not math.isfinite(lower_fraction)
                    or not 0 <= upper_fraction <= 1
                    or not 0 <= lower_fraction <= 1
                    or upper_fraction + lower_fraction > 1):
                raise ValueError("invalid episode budget")
            # 與正式固定配額一致：先各自截去小數，再把剩餘回合留給聯合訓練。
            upper = int(total * upper_fraction)
            lower = int(total * lower_fraction)
            joint = total - upper - lower
        except (ValueError, OverflowError):
            return "階段回合預覽：請填入有效總回合數與階段比例；兩個比例合計不可超過 1。"
        return (f"階段回合預覽：上層 {upper:,}、下層 {lower:,}、聯合 {joint:,} 回合。"
                "示範初始化不占環境訓練回合。")

    # 欄位標籤或輸入框取得焦點時更新說明區。
    def _bind_field_focus(self, record):
        path = record["path"]
        for widget in (record["widget"], record["label_widget"],
                       record["path_widget"], record["input_widget"]):
            widget.bind("<FocusIn>", lambda _event, item=record: self._on_field_focus(item))
            widget.bind("<Button-1>", lambda _event, p=path: self._select_help_field(p), add="+")

    def _on_field_focus(self, record):
        self._select_help_field(record["path"])
        self._cancel_pending_scroll()
        self._pending_scroll = self.after_idle(self._finish_focus_scroll, record["path"])

    def _finish_focus_scroll(self, path):
        self._pending_scroll = None
        record = self._row_by_path.get(path)
        if record is not None:
            self._scroll_to_record(record)

    def _select_help_field(self, path):
        self._selected_path = path
        self._update_help()

    # 只更新目前欄位的說明、原始路徑、專案預設與不適用原因。
    def _update_help(self):
        path = self._selected_path
        if path is None or path not in self.field_vars:
            self.help_title_var.set("選取任一欄位查看用途與預設值。")
            self.help_text_var.set("")
            return
        self.help_title_var.set(field_label(path))
        default = self._get_value(self._project_defaults, path, None)
        try:
            default_text = json.dumps(default, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            default_text = str(default)
        pieces = [self._hrl_budget_preview(), field_help(path),
                  f"路徑：{'.'.join(path)}", f"專案預設：{default_text}"]
        reason = self._inactive_paths.get(path)
        if reason:
            pieces.append(f"不適用：{reason}")
        self.help_text_var.set("　".join(piece for piece in pieces if piece))

    # 說明區依可用寬度換行，不讓長文字撐寬父視窗。
    def _resize_help(self, event):
        wraplength = max(180, event.width - 18)
        if int(self.help_title.cget("wraplength") or 0) != wraplength:
            self.help_title.configure(wraplength=wraplength)

    # 搜尋、適用性與顯示狀態只隱藏既有列，不重建變數或輸入控制項。
    def apply_filter(self, *_args) -> None:
        query = self.search_var.get().casefold().strip()
        context = self.context_settings()
        inactive = self.core.inactive_setting_paths(self.method, context)
        self._inactive_paths = dict(inactive)
        show_inactive = bool(self.show_inactive_var.get())
        visible_count = 0
        applicable_count = 0
        hidden_count = 0
        tab_matches = []
        for tab_name in _TAB_GROUPS:
            for record in self._tab_rows.get(tab_name, []):
                path = record["path"]
                reason = inactive.get(path)
                record["inactive_reason"] = reason
                reason_text = f"不適用：{reason}" if reason else ""
                if reason_text != record["reason_text"]:
                    record["reason_widget"].configure(text=reason_text)
                    record["reason_text"] = reason_text
                if reason and not record["reason_visible"]:
                    record["reason_widget"].grid()
                    record["reason_visible"] = True
                elif not reason and record["reason_visible"]:
                    record["reason_widget"].grid_remove()
                    record["reason_visible"] = False
                self._set_control_state(record, bool(reason and show_inactive))
                searchable = record["search_text"]
                if reason:
                    searchable = f"{searchable} {reason.casefold()}"
                matches = not query or query in searchable
                is_visible = matches and (not reason or show_inactive)
                record["visible"] = is_visible
                if is_visible:
                    visible_count += 1
                    if reason is None:
                        applicable_count += 1
                else:
                    hidden_count += 1
            tab_matches.append(self._filter_tab_groups(tab_name))

        self.match_count_var.set(
            f"顯示 {visible_count}｜適用 {applicable_count}｜隱藏 {hidden_count}")
        if query:
            try:
                current = self.notebook.index("current")
            except tk.TclError:
                current = -1
            if current < 0 or not tab_matches[current]:
                for index, matched in enumerate(tab_matches):
                    if matched:
                        self.notebook.select(index)
                        break
        # 同頁搜尋不會觸發換頁事件，說明仍須跟著可見欄位更新。
        self._on_tab_changed()

    # 依可見欄位顯示群組與頁籤空狀態，保留原始 pack 順序。
    def _filter_tab_groups(self, tab_name):
        groups = self._group_records.get(tab_name, [])
        body = self._tab_bodies.get(tab_name)
        if body is None:
            return False
        for group in groups:
            rows = group["rows"]
            self._restore_pack_order(rows)
            group["visible"] = any(row["visible"] for row in rows)
        self._restore_pack_order(groups)
        empty = self._tab_empty_labels[tab_name]
        has_rows = any(group["visible"] for group in groups)
        if has_rows:
            if empty.winfo_manager():
                empty.pack_forget()
        elif not empty.winfo_manager():
            empty.pack(fill="x", anchor="w", padx=4, pady=8)
        canvas = self._tab_canvases[tab_name]
        canvas.configure(scrollregion=canvas.bbox("all"))
        return has_rows

    # 將已隱藏元件插回原本相鄰欄位，不重新建立元件。
    @staticmethod
    def _restore_pack_order(records):
        for index, record in enumerate(records):
            widget = record["widget"]
            if record["visible"]:
                if not widget.winfo_manager():
                    next_widget = None
                    for following in records[index + 1:]:
                        if following["visible"] and following["widget"].winfo_manager():
                            next_widget = following["widget"]
                            break
                    options = record["pack_options"].copy()
                    if next_widget is not None:
                        options["before"] = next_widget
                    widget.pack(**options)
            elif widget.winfo_manager():
                widget.pack_forget()

    # 只有顯示不適用列時停用該列的輸入控制項。
    @staticmethod
    def _set_control_state(record, disabled):
        if record["control_disabled"] == disabled:
            return
        widget = record["input_widget"]
        state = "disabled" if disabled else record["normal_state"]
        try:
            widget.configure(state=state)
        except tk.TclError:
            if disabled:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled"])
        record["control_disabled"] = disabled

    # 只允許本頁籤內的一般滾輪事件操作對應 Canvas。
    def _bind_scoped_wheel(self, widget, canvas):
        if isinstance(widget, (ttk.Combobox, ttk.Scrollbar)):
            return
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, lambda event, c=canvas: self._scroll_wheel(event, c))
        for child in widget.winfo_children():
            self._bind_scoped_wheel(child, canvas)

    # Shift 滾輪與下拉選單保留給輸入控制項本身。
    @staticmethod
    def _scroll_wheel(event, canvas):
        if isinstance(event.widget, (ttk.Combobox, ttk.Scrollbar)):
            return None
        if not canvas.winfo_ismapped() or not event.widget.winfo_ismapped():
            return None
        if event.state & 0x1:
            return None
        if event.num == 4:
            units = -1
        elif event.num == 5:
            units = 1
        else:
            if event.delta == 0:
                return None
            units = -int(event.delta / 120)
            if units == 0:
                units = -1 if event.delta > 0 else 1
        bounds = canvas.bbox("all")
        if bounds and bounds[3] - bounds[1] > canvas.winfo_height():
            canvas.yview_scroll(units, "units")
            return "break"
        return None

    # 將條件欄位讀入新預設副本，未完成的控制草稿暫以預設值判斷適用性。
    def context_settings(self) -> dict:
        settings = copy.deepcopy(self._project_defaults)
        for path in _CONTROLLER_PATHS:
            if path not in self.field_vars:
                continue
            variable, kind, prototype = self.field_vars[path]
            raw = variable.get()
            try:
                if kind == "bool":
                    value = bool(raw)
                elif kind == "option":
                    value = self._option_value(path, raw, prototype, invalid_fallback=True)
                elif path in {("HRL", "pretrain_batches"),
                              ("HRL_PPO_LOWER", "pretrain_batches")}:
                    value = int(raw)
                    if value < 0:
                        continue
                else:
                    value = raw
            except (TypeError, ValueError):
                continue
            self._set_value(settings, path, value)
        return settings

    # 讀取一個欄位並保留既有表單的數值轉型規則。
    def read_field(self, path) -> Any:
        field_path = self._normalise_path(path)
        variable, kind, prototype = self.field_vars[field_path]
        raw = variable.get()
        try:
            if kind == "bool":
                value = bool(raw)
            elif kind == "json":
                value = json.loads(raw)
            elif kind == "option":
                value = self._option_value(field_path, raw, prototype)
            elif field_path in {("ENV", "energy_weight"), ("ENV", "penalty")}:
                value = float(raw)
            elif isinstance(prototype, int) and not isinstance(prototype, bool):
                value = int(raw)
            elif isinstance(prototype, float):
                value = float(raw)
            else:
                value = raw
        except json.JSONDecodeError as error:
            dotted = ".".join(field_path)
            raise ValueError(f"設定 {dotted} 的 JSON 格式錯誤：{error.msg}") from error
        except ValueError as error:
            dotted = ".".join(field_path)
            raise ValueError(f"設定 {dotted} 的數值格式錯誤：{raw!r}") from error
        return value

    # 在全新預設副本中套用表單值；不適用草稿沿用舊表單的特殊保留條件。
    def collect_settings(self) -> dict:
        settings = copy.deepcopy(self.core.default_settings(self.method))

        def read_and_focus(path):
            try:
                return self.read_field(path)
            except ValueError:
                self.focus_field(path)
                raise

        for path in _CONTROLLER_PATHS:
            if path in self.field_vars:
                self._set_value(settings, path, read_and_focus(path))

        inactive = self.core.inactive_setting_paths(self.method, settings)
        for path in self.field_vars:
            if path in _CONTROLLER_PATHS:
                continue
            if path in inactive and path not in _PRESERVED_INACTIVE_PATHS:
                continue
            if path in inactive:
                try:
                    value = self.read_field(path)
                except ValueError:
                    # 無效且不適用的保留草稿不應阻擋目前配置快照。
                    continue
                if path in {("HRL", "fragment_steps"),
                            ("HRL_PPO", "fragment_steps")}:
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        continue
                if path in {("HRL_UPPER", "risk_penalty_weight"),
                            ("HRL_PPO_UPPER", "risk_penalty_weight")}:
                    if not math.isfinite(value):
                        continue
                if path in {("HRL_UPPER", "gamma"), ("HRL_PPO_UPPER", "gamma")}:
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        continue
                if path in {("HRL_UPPER", "average_reward_lr"),
                            ("HRL_PPO_UPPER", "average_reward_lr")}:
                    if not math.isfinite(value) or value <= 0:
                        continue
                self._set_value(settings, path, value)
                continue
            self._set_value(settings, path, read_and_focus(path))
        return settings

    # 依目前條件回傳可用的點分隔參數路徑。
    def active_paths(self, settings=None) -> list[str]:
        context = self.context_settings() if settings is None else copy.deepcopy(settings)
        inactive = self.core.inactive_setting_paths(self.method, context)
        return sorted(".".join(path) for path in self.field_vars if path not in inactive)

    # 讓指定欄位可見、切換到對應頁籤並捲動至欄位。
    def focus_field(self, path) -> None:
        field_path = self._normalise_path(path)
        record = self._row_by_path.get(field_path)
        if record is None:
            return
        self._selected_path = field_path
        if record.get("inactive_reason") and not self.show_inactive_var.get():
            self.show_inactive_var.set(True)
        self.search_var.set(".".join(field_path))
        self.apply_filter()
        tab_names = tuple(_TAB_GROUPS)
        self.notebook.select(tab_names.index(record["tab"]))
        self.update_idletasks()
        self._scroll_to_record(record)
        if not record.get("inactive_reason"):
            record["input_widget"].focus_set()
        self._update_help()

    # 讓 Tab 鍵聚焦到的欄位留在目前捲動視窗內。
    def _scroll_to_record(self, record):
        canvas = self._tab_canvases.get(record["tab"])
        body = self._tab_bodies.get(record["tab"])
        if canvas is None or body is None or not record["widget"].winfo_ismapped():
            return
        canvas.update_idletasks()
        bounds = canvas.bbox("all")
        if not bounds:
            return
        row_y = record["widget"].winfo_rooty() - body.winfo_rooty()
        row_height = record["widget"].winfo_height()
        visible_height = max(1, canvas.winfo_height())
        scroll_height = max(1, bounds[3] - bounds[1])
        current_top = canvas.canvasy(0)
        current_bottom = current_top + visible_height
        if row_y < current_top:
            target = max(0.0, (row_y - 12) / scroll_height)
            canvas.yview_moveto(target)
        elif row_y + row_height > current_bottom:
            target = max(0.0, (row_y + row_height - visible_height + 12) / scroll_height)
            canvas.yview_moveto(target)

    # 僅回傳核心適用性規則提供的結果，不複製驗證或範圍條件。
    def _inactive_settings(self, settings):
        return self.core.inactive_setting_paths(self.method, settings)

    # 將選項顯示標籤轉回核心的 canonical 值。
    def _option_value(self, path, raw, prototype, invalid_fallback=False):
        options = self.core.setting_options(*path)
        labels = OPTION_LABELS.get(path, {})
        for option in options:
            if labels.get(option, option) == raw:
                return option
        if raw in options:
            return raw
        if invalid_fallback:
            return prototype
        return raw

    # 支援 tuple 路徑與 GROUP.field 字串。
    @staticmethod
    def _normalise_path(path):
        if isinstance(path, str):
            return tuple(path.split("."))
        if isinstance(path, Sequence):
            return tuple(path)
        raise TypeError("path must be a dotted string or a sequence")

    # 從巢狀設定副本讀取值。
    @staticmethod
    def _get_value(settings, path, fallback):
        value = settings
        try:
            for segment in path:
                value = value[segment]
        except (KeyError, TypeError):
            return fallback
        return value

    # 將完整欄位路徑寫入呼叫端之外的新設定副本。
    @staticmethod
    def _set_value(settings, path, value):
        target = settings
        for segment in path[:-1]:
            target = target.setdefault(segment, {})
        target[path[-1]] = copy.deepcopy(value)

    # 清除搜尋後仍將鍵盤焦點留在搜尋欄。
    def _clear_search(self):
        self.search_var.set("")
        self.search_entry.focus_set()

    # 切換頁籤時先說明該頁可見欄位，避免顯示另一頁的舊說明。
    def _on_tab_changed(self, _event=None):
        selected_tab = self.notebook.select()
        if not selected_tab:
            return
        tab_name = self.notebook.tab(selected_tab, "text")
        current = self._row_by_path.get(self._selected_path)
        if current is None or current["tab"] != tab_name or not current["visible"]:
            visible_rows = [row for row in self._tab_rows.get(tab_name, []) if row["visible"]]
            if visible_rows:
                self._selected_path = visible_rows[0]["path"]
            else:
                self._selected_path = None
        if self._selected_path in self.field_vars:
            self._update_help()
        else:
            self.help_title_var.set("目前沒有符合的設定欄位。")
            self.help_text_var.set("可清除搜尋，或勾選「顯示不適用欄位」查看條件。")

