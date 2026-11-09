"""Tk dialog for saving and applying sparse experiment setting presets."""

from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path

from runtime import fileIO as saving
from tkinter import messagebox, ttk
from typing import Any

from gui.preset_config import PRESET_DIRECTORY, build_preset
from gui.setting_labels import field_label
from gui.gui_style import MONO_FONT, configure_gui_style


_CHECKED = "☑"
_UNCHECKED = "☐"
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"}
_WINDOWS_RESERVED.update(f"COM{number}" for number in range(1, 10))
_WINDOWS_RESERVED.update(f"LPT{number}" for number in range(1, 10))


class PresetDialog(tk.Toplevel):
    """Manage sparse JSON presets without changing the experiment queue."""

    def __init__(self, app, directory=PRESET_DIRECTORY):
        super().__init__(app.root)
        configure_gui_style(self._root())
        self.app = app
        self.directory = Path(directory)
        self.title("設定範本")
        self.geometry("1000x760")
        self.minsize(850, 610)
        self.transient(app.root)

        self.selected_preset: dict[str, Any] | None = None
        self._field_labels = self._collect_field_labels()
        self._field_search_text = {
            path: f"{'.'.join(path)} {label}".casefold()
            for path, label in self._field_labels.items()
        }
        self._selection = {path: False for path in self._field_labels}
        self._field_iids: dict[str, tuple[str, str]] = {}
        self._method_trace = None

        self.file_status_var = tk.StringVar(value="")
        self.method_semantics_var = tk.StringVar(value="")
        self.search_var = tk.StringVar(value="")
        self.selected_count_var = tk.StringVar(value="")
        self.name_var = tk.StringVar(value="")
        self.save_status_var = tk.StringVar(value="")

        self._build_widgets()
        self.search_var.trace_add("write", self._refresh_field_rows)
        self.name_var.trace_add("write", self._refresh_save_preview)
        try:
            self._method_trace = app.method_var.trace_add(
                "write", self._refresh_method_semantics)
        except (AttributeError, tk.TclError):
            self._method_trace = None

        self._refresh_field_rows()
        self._refresh_file_list()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", self.close)

    def _collect_field_labels(self):
        return {path: field_label(path) for path in self.app.field_vars}

    def _build_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        content = ttk.Frame(self, padding=10)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        load_box = ttk.LabelFrame(content, text="載入與套用範本", padding=8)
        load_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        load_box.columnconfigure(0, weight=1, minsize=230)
        load_box.columnconfigure(1, weight=3)
        load_box.rowconfigure(0, weight=1)

        file_side = ttk.Frame(load_box)
        file_side.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        file_side.columnconfigure(0, weight=1)
        file_side.rowconfigure(0, weight=1)
        self.file_tree = ttk.Treeview(
            file_side, columns=("filename",), show="headings", height=6,
            selectmode="browse")
        self.file_tree.heading("filename", text="JSON 範本")
        self.file_tree.column("filename", width=250, anchor="w")
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        file_scroll = ttk.Scrollbar(file_side, orient="vertical", command=self.file_tree.yview)
        file_scroll.grid(row=0, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=file_scroll.set)
        self.file_tree.bind("<<TreeviewSelect>>", self._on_file_selected)

        file_buttons = ttk.Frame(file_side)
        file_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.refresh_button = ttk.Button(
            file_buttons, text="重新整理", command=self._refresh_file_list)
        self.refresh_button.pack(side="left")
        self.apply_button = ttk.Button(
            file_buttons, text="套用所選範本", command=self.apply_selected_preset,
            state="disabled")
        self.apply_button.pack(side="left", padx=(6, 0))

        preview_side = ttk.Frame(load_box)
        preview_side.grid(row=0, column=1, sticky="nsew")
        preview_side.columnconfigure(0, weight=1)
        preview_side.rowconfigure(1, weight=1)
        self.method_semantics_label = ttk.Label(
            preview_side, textvariable=self.method_semantics_var, wraplength=450)
        self.method_semantics_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.method_semantics_label.bind("<Configure>", self._fit_label_wrap)
        self.loaded_preview = self._make_text_area(preview_side, height=7)
        self.loaded_preview.grid(row=1, column=0, sticky="nsew")
        self.file_status_label = ttk.Label(
            load_box, textvariable=self.file_status_var, wraplength=700)
        self.file_status_label.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.file_status_label.bind("<Configure>", self._fit_label_wrap)

        save_box = ttk.LabelFrame(content, text="儲存目前表單", padding=8)
        save_box.grid(row=1, column=0, sticky="nsew")
        save_box.columnconfigure(0, weight=3)
        save_box.columnconfigure(1, weight=2)
        save_box.rowconfigure(1, weight=1)

        tools = ttk.Frame(save_box)
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        tools.columnconfigure(1, weight=1)
        ttk.Label(tools, text="搜尋欄位：").grid(row=0, column=0, sticky="w")
        self.search_entry = ttk.Entry(tools, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.search_entry.bind("<Escape>", self._clear_field_search)
        self.select_all_button = ttk.Button(tools, text="全選", command=self.select_all)
        self.select_all_button.grid(row=0, column=2, padx=(0, 4))
        self.clear_selection_button = ttk.Button(
            tools, text="清除選取", command=self.clear_selection)
        self.clear_selection_button.grid(row=0, column=3, padx=(0, 8))
        ttk.Label(tools, textvariable=self.selected_count_var).grid(
            row=0, column=4, sticky="e")

        field_frame = ttk.Frame(save_box)
        field_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        field_frame.columnconfigure(0, weight=1)
        field_frame.rowconfigure(0, weight=1)
        self.field_tree = ttk.Treeview(
            field_frame, columns=("selected", "path", "label", "raw"),
            show="headings", selectmode="browse", height=13)
        self.field_tree.heading("selected", text="選取")
        self.field_tree.heading("path", text="完整路徑")
        self.field_tree.heading("label", text="欄位說明")
        self.field_tree.heading("raw", text="目前原始值")
        self.field_tree.column("selected", width=46, minwidth=46, stretch=False,
                               anchor="center")
        self.field_tree.column("path", width=175, minwidth=125, stretch=False)
        self.field_tree.column("label", width=230, minwidth=150, stretch=False)
        self.field_tree.column("raw", width=300, minwidth=120, stretch=True)
        self.field_tree.grid(row=0, column=0, sticky="nsew")
        field_y_scroll = ttk.Scrollbar(
            field_frame, orient="vertical", command=self.field_tree.yview)
        field_y_scroll.grid(row=0, column=1, sticky="ns")
        field_x_scroll = ttk.Scrollbar(
            field_frame, orient="horizontal", command=self.field_tree.xview)
        field_x_scroll.grid(row=1, column=0, sticky="ew")
        self.field_tree.configure(
            yscrollcommand=field_y_scroll.set, xscrollcommand=field_x_scroll.set)
        self.field_tree.bind("<Button-1>", self._on_field_click)
        self.field_tree.bind("<space>", self._on_field_space)

        save_preview_side = ttk.Frame(save_box)
        save_preview_side.grid(row=1, column=1, sticky="nsew")
        save_preview_side.columnconfigure(0, weight=1)
        save_preview_side.rowconfigure(1, weight=1)
        ttk.Label(save_preview_side, text="所選欄位 JSON 預覽").grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.save_preview = self._make_text_area(save_preview_side, height=13)
        self.save_preview.grid(row=1, column=0, sticky="nsew")

        save_controls = ttk.Frame(save_box)
        save_controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        save_controls.columnconfigure(1, weight=1)
        ttk.Label(save_controls, text="範本名稱／檔名：").grid(
            row=0, column=0, sticky="w")
        name_entry = ttk.Entry(save_controls, textvariable=self.name_var)
        name_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(save_controls, text=".json").grid(row=0, column=2, padx=(3, 8))
        self.save_button = ttk.Button(
            save_controls, text="儲存範本", command=self.save_preset)
        self.save_button.grid(row=0, column=3)
        self.save_status_label = ttk.Label(
            save_box, textvariable=self.save_status_var, wraplength=700)
        self.save_status_label.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.save_status_label.bind("<Configure>", self._fit_label_wrap)

    def _fit_label_wrap(self, event):
        # The configured wrap width must remain inside the current grid cell,
        # especially at the dialog's 850-pixel minimum width.
        width = max(1, event.width - 12)
        try:
            current = int(event.widget.cget("wraplength"))
        except (TypeError, ValueError):
            current = -1
        if current != width:
            event.widget.configure(wraplength=width)

    def _make_text_area(self, parent, *, height):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(frame, height=height, wrap="none", font=MONO_FONT)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set,
                      state="disabled")
        # Let callers grid the frame while retaining a short, convenient name.
        return frame

    def _set_text(self, frame, value):
        widget = frame.winfo_children()[0]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _raw_field_value(self, path):
        variable, _kind, _default = self.app.field_vars[path]
        value = variable.get()
        if isinstance(value, bool):
            if value:
                return "true"
            else:
                return "false"
        return str(value)

    def _clear_field_search(self, _event=None):
        self.search_var.set("")
        # 已由搜尋欄處理 Escape，停止傳遞至視窗的關閉快捷鍵。
        return "break"

    def _refresh_field_rows(self, *_args):
        if not hasattr(self, "field_tree"):
            return
        query = self.search_var.get().strip().casefold()
        for iid in self.field_tree.get_children(""):
            self.field_tree.delete(iid)
        self._field_iids.clear()
        for path, label in self._field_labels.items():
            if query and query not in self._field_search_text[path]:
                continue
            iid = ".".join(path)
            # Paths in settings are unique dotted identifiers. Keep a mapping
            # rather than parsing the iid so path handling remains explicit.
            self._field_iids[iid] = path
            try:
                raw = self._raw_field_value(path)
            except (KeyError, AttributeError, tk.TclError):
                raw = ""
            if self._selection[path]:
                selected_glyph = _CHECKED
            else:
                selected_glyph = _UNCHECKED
            self.field_tree.insert(
                "", "end", iid=iid,
                values=(selected_glyph,
                        ".".join(path), label, raw))
        self._update_selected_count()
        self._refresh_save_preview()

    def _update_selected_count(self):
        count = sum(self._selection.values())
        self.selected_count_var.set(f"已選 {count} / {len(self._field_labels)} 個欄位")

    def _set_selected(self, path, selected):
        if path not in self._selection:
            return
        self._selection[path] = bool(selected)
        iid = ".".join(path)
        if self.field_tree.exists(iid):
            values = list(self.field_tree.item(iid, "values"))
            if self._selection[path]:
                values[0] = _CHECKED
            else:
                values[0] = _UNCHECKED
            self.field_tree.item(iid, values=values)
        self._update_selected_count()
        self._refresh_save_preview()

    def _toggle_path(self, path):
        self._set_selected(path, not self._selection.get(path, False))

    def _on_field_click(self, event):
        iid = self.field_tree.identify_row(event.y)
        if not iid or iid not in self._field_iids:
            return
        column = self.field_tree.identify_column(event.x)
        if column == "#1":
            self.field_tree.focus_set()
            self.field_tree.selection_set(iid)
            self.field_tree.focus(iid)
            self._toggle_path(self._field_iids[iid])
            return "break"

    def _on_field_space(self, _event):
        selection = self.field_tree.selection()
        if selection:
            path = self._field_iids.get(selection[0])
            if path is not None:
                self._toggle_path(path)
        return "break"

    def select_all(self):
        for path in self._field_labels:
            self._selection[path] = True
        self._refresh_field_rows()

    def clear_selection(self):
        for path in self._field_labels:
            self._selection[path] = False
        self._refresh_field_rows()

    def _selected_values(self):
        values = {}
        for path in self._field_labels:
            if self._selection[path]:
                # The app parser is called only for selected paths. An
                # unfinished unselected draft must not block a sparse save.
                values[path] = self.app._read_setting_field(path)
        return values

    def _build_save_preset(self, *, require_name=False):
        name = self.name_var.get().strip()
        if require_name and not name:
            raise ValueError("請輸入範本名稱。")
        if not name:
            name = "未命名"
        method = self.app.method_var.get()
        values = self._selected_values()
        return build_preset(name, method, values)

    def _refresh_save_preview(self, *_args):
        if not hasattr(self, "save_preview"):
            return
        try:
            preset = self._build_save_preset()
            preview = json.dumps(preset, ensure_ascii=False, indent=2)
        except (ValueError, TypeError, KeyError, tk.TclError) as exc:
            preview = f"目前選取欄位無法解析：\n{exc}"
        self._set_text(self.save_preview, preview)

    def _refresh_file_list(self, *, select_name=None):
        self.file_status_var.set("")
        previous_selection = self.file_tree.selection()
        if select_name is None and previous_selection:
            select_name = previous_selection[0]
        for iid in self.file_tree.get_children(""):
            self.file_tree.delete(iid)
        self.selected_preset = None
        self.apply_button.configure(state="disabled")
        self._set_text(self.loaded_preview, "正在讀取範本清單…")
        self.method_semantics_var.set("")
        try:
            paths = sorted(
                (path for path in self.directory.glob("*.json")
                 if path.is_file() and not path.is_symlink()),
                key=lambda path: path.name.casefold())
        except OSError as exc:
            self.file_status_var.set(f"無法讀取範本資料夾：{exc}")
            self._set_text(self.loaded_preview, "無法讀取範本清單。")
            return

        for path in paths:
            self.file_tree.insert("", "end", iid=path.name, values=(path.name,))
        if select_name and self.file_tree.exists(select_name):
            self.file_tree.selection_set(select_name)
            self.file_tree.focus(select_name)
            self._load_preset_file(self.directory / select_name)
        elif paths:
            self._set_text(self.loaded_preview, "請選取一個範本檔案。")
            self.method_semantics_var.set("")
        else:
            self._set_text(self.loaded_preview, "目前沒有 JSON 範本。")
            self.method_semantics_var.set("")

    def _on_file_selected(self, _event=None):
        selection = self.file_tree.selection()
        if not selection:
            self.selected_preset = None
            self.apply_button.configure(state="disabled")
            return
        self._load_preset_file(self.directory / selection[0])

    def _load_preset_file(self, path):
        self.selected_preset = None
        self.apply_button.configure(state="disabled")
        try:
            # UTF-8 BOM is common in files saved by Windows editors.
            preset = saving.read_json(path, encoding="utf-8-sig")
            if not isinstance(preset, dict):
                raise ValueError("JSON 根節點必須是物件。")
            preview = json.dumps(preset, ensure_ascii=False, indent=2)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.file_status_var.set(f"讀取失敗：{exc}")
            self._set_text(self.loaded_preview, "無法載入此 JSON 範本。")
            self.method_semantics_var.set("")
            return False

        self.selected_preset = preset
        self.file_status_var.set("")
        self.apply_button.configure(state="normal")
        self._set_text(self.loaded_preview, preview)
        self._refresh_method_semantics()
        return True

    def _refresh_method_semantics(self, *_args):
        preset = self.selected_preset
        if preset is None:
            self.method_semantics_var.set("")
            return
        method = preset.get("method")
        if method:
            self.method_semantics_var.set(
                f"此範本指定方法「{method}」。套用後切換至該方法，並以該方法預設值"
                "重設未列出的欄位。")
        else:
            current_method = self.app.method_var.get()
            self.method_semantics_var.set(
                f"此範本只有通用設定：套用於目前所選方法「{current_method}」，"
                "未列出的欄位會重設為該方法預設值。")

    def apply_selected_preset(self):
        if self.selected_preset is None:
            return False
        try:
            applied = self.app._apply_preset(self.selected_preset)
        except Exception as exc:
            self.file_status_var.set(f"套用失敗：{exc}")
            return False
        if not applied:
            self.file_status_var.set("套用未完成；請確認佇列未執行且設定有效。")
            return False
        self.file_status_var.set("範本已套用。")
        self.close()
        return True

    @staticmethod
    def _filename_stem(value):
        name = value.strip()
        if name.casefold().endswith(".json"):
            name = name[:-5]
        if not name or name in {".", ".."}:
            raise ValueError("請輸入有效的範本檔名。")
        if _INVALID_FILENAME.search(name) or any(ord(char) < 32 for char in name):
            raise ValueError("檔名含有 Windows 不允許的字元。")
        if name[-1] in {".", " "}:
            raise ValueError("檔名不可用句點或空白結尾。")
        if Path(name).name != name:
            raise ValueError("檔名不可包含資料夾路徑。")
        if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ValueError("此名稱為 Windows 保留檔名。")
        return name

    def save_preset(self):
        try:
            stem = self._filename_stem(self.name_var.get())
            # Use the normalized stem in both the JSON name and filename.
            values = self._selected_values()
            preset = build_preset(stem, self.app.method_var.get(), values)
        except (ValueError, TypeError, KeyError, tk.TclError) as exc:
            self.save_status_var.set(f"儲存失敗：{exc}")
            return False

        try:
            saving.make_directory(self.directory, parents=True, exist_ok=True)
            destination = self.directory / f"{stem}.json"
            if destination.exists():
                overwrite = messagebox.askyesno(
                    "覆寫範本", f"「{destination.name}」已存在，要覆寫嗎？", parent=self)
                if not overwrite:
                    self.save_status_var.set("已取消覆寫。")
                    return False
        except OSError as exc:
            self.save_status_var.set(f"儲存失敗：{exc}")
            return False

        try:
            saving.save_preset(destination, preset)
        except (OSError, TypeError, ValueError) as exc:
            self.save_status_var.set(f"儲存失敗：{exc}")
            return False

        self._refresh_file_list(select_name=destination.name)
        self.save_status_var.set(f"已儲存：{destination.name}")
        return True

    def close(self, _event=None):
        if self._method_trace is not None:
            try:
                self.app.method_var.trace_remove("write", self._method_trace)
            except (AttributeError, tk.TclError):
                pass
            self._method_trace = None
        self.destroy()
