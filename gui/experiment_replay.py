"""Embedded Tk replay panel for episode/step parquet traces.

Importing this module uses only the Python standard library. The trace viewer,
NumPy, pandas, and Matplotlib are loaded when data is first supplied or read.
"""

from __future__ import annotations

from bisect import bisect_left
import importlib
import math
import numbers
import queue
import threading
from decimal import Decimal, InvalidOperation
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from runtime import fileIO as saving
from gui.gui_style import BODY_BOLD_FONT, configure_gui_style


_DISPLAY_MODES = ("all", "move-only", "action-only", "static")
_DISPLAY_LABELS = ("完整顯示", "僅移動軌跡", "僅服務動作", "僅狀態")
_DISPLAY_LABEL_TO_MODE = dict(zip(_DISPLAY_LABELS, _DISPLAY_MODES))
_AOI_VIEW_LABELS = ("UAV 收到的 AoI", "裝置資料 AoI")
_AOI_VIEW_TO_MODE = dict(zip(_AOI_VIEW_LABELS, ("uav", "source")))
_AOI_MODE_TO_VIEW = {value: key for key, value in _AOI_VIEW_TO_MODE.items()}
_EPISODE_COLUMNS = ("episode", "init_UAV_pos", "init_device_pos", "risk_region")
_STEP_COLUMNS = ("episode", "step", "UAV_pos", "device_pos", "device_AOI",
                 "UAV_action_select", "UAV_action_move", "reward")


# 尋找同資料夾內的標準 episode 與 step 軌跡檔。
def detect_parquet_pair(selected_path):
    """Return canonical episode/step siblings when ``selected_path`` is one."""
    selected = Path(selected_path).expanduser()
    name = selected.name.casefold()
    if name not in ("episode.parquet", "step.parquet"):
        return None
    if name == "episode.parquet":
        episode_path = selected
        step_path = selected.with_name("step.parquet")
    else:
        episode_path = selected.with_name("episode.parquet")
        step_path = selected
    if episode_path.is_file() and step_path.is_file():
        return episode_path, step_path
    return None


# 將單一數值轉成有限浮點數，無效值則報錯。
def _scalar(value, label):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


# 將步數轉成精確整數，避免大型整數經浮點轉換後失真。
def _integer(value, label):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite integer")

    # float 只檢查原有數值範圍，整數仍從原值轉換以保留精度；
    # 先拒絕 1e1000000 等輸入，避免展開巨大整數而卡住介面。
    try:
        finite_number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a finite integer") from None
    if not math.isfinite(finite_number):
        raise ValueError(f"{label} must be a finite integer")

    if isinstance(value, numbers.Rational):
        if value.denominator != 1:
            raise ValueError(f"{label} must be a finite integer")
        return int(value.numerator)
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            raise ValueError(f"{label} must be a finite integer") from None
    else:
        if not finite_number.is_integer():
            raise ValueError(f"{label} must be a finite integer")
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label} must be a finite integer") from None

    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{label} must be a finite integer")
    return int(number)


# 判斷可選的單一欄位是否為空值。
def _missing_scalar(value):
    if value is None or value.__class__.__name__ in ("NAType", "NaTType"):
        return True
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return False
    if value is None or value.__class__.__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, numbers.Real):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, Decimal):
        return value.is_nan()
    return False


# 檢查回合識別值是否存在、有限且可作為索引。
def _episode_id(value, label):
    if _missing_scalar(value):
        raise ValueError(f"{label} must be present")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must be present")
    if isinstance(value, numbers.Real):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise ValueError(f"{label} must be finite")
    try:
        hash(value)
    except TypeError:
        raise ValueError(f"{label} must be a scalar identifier") from None
    return value


# 將可迭代欄位轉成列表，拒絕字串與空值。
def _sequence(value, label):
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return list(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a sequence") from None


# 檢查每個座標含兩個有限數值。
def _coordinates(value, label):
    rows = _sequence(value, label)
    result = []
    for index, row in enumerate(rows):
        point = _sequence(row, f"{label}[{index}]")
        if len(point) != 2:
            raise ValueError(f"{label}[{index}] must contain exactly 2 coordinates")
        result.append([_scalar(item, f"{label}[{index}]") for item in point])
    return result


# 檢查每個風險矩形含四個有限數值。
def _risk_regions(value, label):
    rows = _sequence(value, label)
    result = []
    for index, row in enumerate(rows):
        region = _sequence(row, f"{label}[{index}]")
        if len(region) != 4:
            raise ValueError(f"{label}[{index}] must contain exactly 4 coordinates")
        result.append([_scalar(item, f"{label}[{index}]") for item in region])
    return result


# 檢查數值向量長度並轉換其元素。
def _vector(value, label, expected_length):
    values = _sequence(value, label)
    if len(values) != expected_length:
        raise ValueError(f"{label} has {len(values)} values; expected {expected_length}")
    return [_scalar(item, label) for item in values]


# 檢查來源 AoI 為每個裝置一個有限正數。
def _source_aoi_vector(value, label, expected_length):
    values = _vector(value, label, expected_length)
    if any(item <= 0 for item in values):
        raise ValueError(f"{label} values must be greater than zero")
    return values


# 檢查動作向量長度，並確認每個動作為整數。
def _actions(value, label, expected_length):
    values = _sequence(value, label)
    if len(values) != expected_length:
        raise ValueError(f"{label} has {len(values)} values; expected {expected_length}")
    actions = []
    for item in values:
        number = _scalar(item, label)
        if not number.is_integer():
            raise ValueError(f"{label} values must be integers")
        actions.append(int(number))
    return actions


# 驗證可選的每 UAV 上層動作標記。
def _flags(value, label, expected_length):
    values = _sequence(value, label)
    if len(values) != expected_length:
        raise ValueError(f"{label} has {len(values)} values; expected {expected_length}")
    flags = []
    for item in values:
        if hasattr(item, "item"):
            try:
                item = item.item()
            except (TypeError, ValueError):
                pass
        if not isinstance(item, bool):
            raise ValueError(f"{label} values must be booleans")
        flags.append(item)
    return flags


# 取出資料表欄名與列資料，供軌跡驗證。
def _records_and_columns(frame, label):
    try:
        columns = set(frame.columns)
        records = frame.to_dict(orient="records")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{label} must be a pandas DataFrame") from None
    return columns, records


# 驗證回合與步驟資料的索引、形狀及數值。
def validate_trace_data(episode_df, step_df):
    """Validate every row shape before a trace can replace the current view."""
    episode_columns, episode_rows = _records_and_columns(episode_df, "episode data")
    step_columns, step_rows = _records_and_columns(step_df, "step data")
    missing_episode = [name for name in _EPISODE_COLUMNS if name not in episode_columns]
    missing_step = [name for name in _STEP_COLUMNS if name not in step_columns]
    if missing_episode:
        raise ValueError("episode parquet is missing required columns: " + ", ".join(missing_episode))
    if missing_step:
        raise ValueError("step parquet is missing required columns: " + ", ".join(missing_step))
    if not episode_rows:
        raise ValueError("episode parquet contains no episodes")

    episode_shapes = {}
    episode_labels = set()
    episode_initial_steps = {}
    for row_index, row in enumerate(episode_rows):
        episode_id = _episode_id(row.get("episode"), f"episode row {row_index} id")
        if episode_id in episode_shapes:
            raise ValueError(f"episode parquet contains duplicate episode id {episode_id}")
        episode_label = str(episode_id)
        if episode_label in episode_labels:
            raise ValueError(
                "episode parquet contains distinct episode ids with the same "
                f"display label {episode_label!r}"
            )
        episode_labels.add(episode_label)
        prefix = f"episode {episode_id}"
        uavs = _coordinates(row.get("init_UAV_pos"), f"{prefix} initial UAV positions")
        devices = _coordinates(row.get("init_device_pos"), f"{prefix} initial device positions")
        if not uavs or not devices:
            raise ValueError(f"{prefix} must contain at least one UAV and one device")
        _risk_regions(row.get("risk_region"), f"{prefix} risk regions")
        initial_step = row.get("initial_step", 0)
        if _missing_scalar(initial_step):
            initial_step = 0
        initial_step = _integer(initial_step, f"{prefix} initial_step")
        if initial_step < 0:
            raise ValueError(f"{prefix} initial_step must be a non-negative integer")
        initial_aoi = row.get("init_device_AOI")
        if _missing_scalar(initial_aoi):
            if initial_step > 0:
                raise ValueError(
                    f"{prefix} init_device_AOI is required when initial_step is greater than zero"
                )
            initial_aoi = [1.0] * len(devices)
        collector_aoi = _vector(initial_aoi, f"{prefix} initial device AoI", len(devices))
        if "init_source_AOI" in episode_columns:
            source_aoi = row.get("init_source_AOI")
            if not _missing_scalar(source_aoi):
                source_aoi = _source_aoi_vector(
                    source_aoi, f"{prefix} initial source AoI", len(devices)
                )
                if any(source > collector for source, collector in zip(
                        source_aoi, collector_aoi)):
                    raise ValueError(
                        f"{prefix} initial device AoI must be greater than or equal to source AoI"
                    )
        if "service_radius" in episode_columns:
            radius = row.get("service_radius")
            if not _missing_scalar(radius):
                radius_value = _scalar(radius, f"{prefix} service_radius")
                if radius_value <= 0:
                    raise ValueError(f"{prefix} service_radius must be greater than zero")
        if "service_rule" in episode_columns:
            service_rule = row.get("service_rule")
            if (not _missing_scalar(service_rule)
                    and not isinstance(service_rule, str)):
                raise ValueError(f"{prefix} service_rule must be a string")
        if "communication_model" in episode_columns:
            communication_model = row.get("communication_model")
            if (not _missing_scalar(communication_model)
                    and not isinstance(communication_model, str)):
                raise ValueError(f"{prefix} communication_model must be a string")
        map_size = row.get("map_size", 10.0)
        if _missing_scalar(map_size):
            map_size = 10.0
        if _scalar(map_size, f"{prefix} map_size") <= 0:
            raise ValueError(f"{prefix} map_size must be greater than zero")
        episode_shapes.setdefault(episode_id, (len(uavs), len(devices)))
        episode_initial_steps[episode_id] = initial_step

    steps_by_episode = {episode_id: set() for episode_id in episode_shapes}
    for row_index, row in enumerate(step_rows):
        episode_id = _episode_id(row.get("episode"), f"step row {row_index} episode id")
        if episode_id not in episode_shapes:
            raise ValueError(f"step row {row_index} refers to unknown episode {episode_id}")
        uav_count, device_count = episode_shapes[episode_id]
        prefix = f"episode {episode_id}, step row {row_index}"
        step_id = _integer(row.get("step"), f"{prefix} step")
        if step_id <= episode_initial_steps[episode_id]:
            raise ValueError(
                f"{prefix} step {step_id} must be greater than initial_step "
                f"{episode_initial_steps[episode_id]}"
            )
        if step_id in steps_by_episode[episode_id]:
            raise ValueError(f"episode {episode_id} contains duplicate step id {step_id}")
        steps_by_episode[episode_id].add(step_id)
        if len(_coordinates(row.get("UAV_pos"), f"{prefix} UAV positions")) != uav_count:
            raise ValueError(f"{prefix} UAV position count does not match the episode")
        if len(_coordinates(row.get("device_pos"), f"{prefix} device positions")) != device_count:
            raise ValueError(f"{prefix} device position count does not match the episode")
        collector_aoi = _vector(row.get("device_AOI"), f"{prefix} device AoI", device_count)
        if "source_AOI" in step_columns:
            source_aoi = row.get("source_AOI")
            if not _missing_scalar(source_aoi):
                source_aoi = _source_aoi_vector(
                    source_aoi, f"{prefix} source AoI", device_count
                )
                if any(source > collector for source, collector in zip(
                        source_aoi, collector_aoi)):
                    raise ValueError(
                        f"{prefix} device AoI must be greater than or equal to source AoI"
                    )
        selections = _actions(row.get("UAV_action_select"),
                              f"{prefix} UAV selections", uav_count)
        if any(selection < -1 or selection > device_count for selection in selections):
            raise ValueError(f"{prefix} has a device selection outside the valid range")
        moves = _coordinates(row.get("UAV_action_move"), f"{prefix} UAV movement")
        if len(moves) != uav_count:
            raise ValueError(f"{prefix} UAV movement count does not match the episode")
        _vector(row.get("reward"), f"{prefix} reward", uav_count)
        for column in ("communication_scheme",):
            if column in step_columns:
                scheme = row.get(column)
                if not _missing_scalar(scheme) and not isinstance(scheme, str):
                    raise ValueError(f"{prefix} {column} must be a string")
        if "UAV_service_success" in step_columns:
            success = row.get("UAV_service_success")
            if not _missing_scalar(success):
                _flags(success, f"{prefix} UAV_service_success", uav_count)
        for column in (
            "UAV_transmission_power", "UAV_transmission_peak_power",
            "UAV_transmission_time",
        ):
            if column in step_columns and not _missing_scalar(row.get(column)):
                values = _vector(row.get(column), f"{prefix} {column}", uav_count)
                if any(value < 0 for value in values):
                    raise ValueError(f"{prefix} {column} values must be non-negative")
        for column in ("UAV_current_position", "UAV_idle"):
            if column in step_columns:
                value = row.get(column)
                other_column = (
                    "UAV_idle" if column == "UAV_current_position"
                    else "UAV_current_position"
                )
                if (column in step_columns and other_column in step_columns
                        and _missing_scalar(value)
                        and not _missing_scalar(row.get(other_column))):
                    continue
                _flags(value, f"{prefix} {column}", uav_count)
        for column in ("upper_reward", "intrinsic_reward"):
            if column in step_columns and not _missing_scalar(row.get(column)):
                _vector(row.get(column), f"{prefix} {column}", uav_count)

    return [row["episode"] for row in episode_rows]


# 依已排序的整數步數找出對應影格；初始狀態使用影格零。
def _frame_for_step(step_ids, actual_step, initial_step):
    if actual_step == initial_step:
        return 0
    position = bisect_left(step_ids, actual_step)
    if position < len(step_ids) and step_ids[position] == actual_step:
        return position + 1
    return None


class ReplayPanel(ttk.Frame):
    """Self-contained embedded replay controls and Matplotlib canvas."""

    # 初始化軌跡載入、播放狀態與嵌入式回放控制元件。
    def __init__(self, parent):
        super().__init__(parent)
        configure_gui_style(self._root())
        self._trace_tool = None
        self._episode_df = None
        self._step_df = None
        self._episode_ids = []
        self._closed = False
        self._load_generation = 0
        self._load_pending = False
        self._load_queue = queue.Queue()
        self._after_ids = set()
        self._play_after_id = None
        self._poll_after_id = None
        self._viewer = None
        self._figure = None
        self._canvas = None

        self.episode_path = tk.StringVar(self)
        self.step_path = tk.StringVar(self)
        self.step_jump_value = tk.StringVar(self)
        self.status = tk.StringVar(self, "請匯入回合檔與步驟檔。")
        self.episode_choice = tk.StringVar(self)
        self.interval_value = tk.StringVar(self, "500")
        self.trajectory_value = tk.StringVar(self, "10")
        self.display_value = tk.StringVar(self, _DISPLAY_LABELS[0])
        self.aoi_view_value = tk.StringVar(self, _AOI_VIEW_LABELS[0])
        self._aoi_view_preference = "uav"
        self.loop_value = tk.BooleanVar(self, False)
        self.legend_value = tk.BooleanVar(self, True)
        self.frame_value = tk.DoubleVar(self, 0)
        self.frame_text = tk.StringVar(self, "影格 0 / 0")

        self._build_ui()
        self._update_frame_controls()
        self.bind("<Destroy>", self._on_destroy, add="+")

    # 回傳目前嵌入的軌跡檢視器。
    @property
    def viewer(self):
        """The current EpisodeStepViewer, exposed for integration and smoke use."""
        return self._viewer

    # 建立軌跡檔選擇、播放控制與成本資訊面板。
    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(5, weight=1, minsize=220)

        import_box = ttk.LabelFrame(
            self, text="匯入軌跡資料", style="Section.TLabelframe")
        import_box.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        import_box.columnconfigure(1, weight=1)
        ttk.Label(import_box, text="回合檔").grid(row=0, column=0, padx=6, pady=3, sticky="w")
        ttk.Entry(import_box, textvariable=self.episode_path).grid(
            row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(import_box, text="瀏覽…", command=self._browse_episode).grid(
            row=0, column=2, padx=4, pady=3)
        ttk.Label(import_box, text="步驟檔").grid(row=1, column=0, padx=6, pady=3, sticky="w")
        ttk.Entry(import_box, textvariable=self.step_path).grid(
            row=1, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(import_box, text="瀏覽…", command=self._browse_step).grid(
            row=1, column=2, padx=4, pady=3)
        ttk.Button(import_box, text="載入", command=self._load_from_fields,
                   style="Accent.TButton").grid(
            row=0, column=3, rowspan=2, padx=8, pady=3, sticky="ns")

        playback = ttk.LabelFrame(
            self, text="回放與步數", style="Section.TLabelframe")
        self.playback_controls = playback
        playback.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        playback.columnconfigure(1, weight=1)
        ttk.Label(playback, text="回合").grid(row=0, column=0, padx=(0, 4))
        self.episode_box = ttk.Combobox(
            playback, textvariable=self.episode_choice, state="disabled", width=12)
        self.episode_box.grid(row=0, column=1, padx=(0, 8))
        self.episode_box.bind("<<ComboboxSelected>>", self._on_episode_changed)
        self.prev_button = ttk.Button(playback, text="◀", width=4, command=self.previous)
        self.prev_button.grid(row=0, column=2, padx=(8, 2), pady=2)
        self.play_button = ttk.Button(
            playback, text="播放", width=8, command=self.play_pause,
            style="Accent.TButton")
        self.play_button.grid(row=0, column=3, padx=2)
        self.next_button = ttk.Button(playback, text="▶", width=4, command=self.next)
        self.next_button.grid(row=0, column=4, padx=2)
        self.start_button = ttk.Button(playback, text="從頭播放", command=self.start)
        self.start_button.grid(row=0, column=5, padx=(8, 2), pady=2)
        self.reset_button = ttk.Button(playback, text="回到開頭", command=self.reset)
        self.reset_button.grid(row=0, column=6, padx=2, pady=2)
        ttk.Label(playback, text="實際步數").grid(
            row=1, column=0, padx=(0, 4), pady=(3, 2), sticky="e")
        self.step_entry = ttk.Entry(
            playback, textvariable=self.step_jump_value, width=9, state="disabled")
        self.step_entry.grid(row=1, column=1, padx=2, pady=(3, 2), sticky="w")
        self.step_entry.bind("<Return>", self._jump_to_step)
        self.jump_button = ttk.Button(
            playback, text="跳至步數", command=self._jump_to_step, state="disabled")
        self.jump_button.grid(row=1, column=2, columnspan=2, padx=2, pady=(3, 2), sticky="w")
        self.export_button = ttk.Button(
            playback, text="匯出 PNG", command=self.export_png, state="disabled",
            style="Accent.TButton")
        self.export_button.grid(row=1, column=5, padx=(8, 2), pady=(3, 2), sticky="w")
        self.loop_button = ttk.Checkbutton(
            playback, text="循環播放", variable=self.loop_value)
        self.loop_button.grid(row=1, column=6, padx=2, pady=(3, 2), sticky="w")

        seek = ttk.Frame(self)
        seek.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 2))
        seek.columnconfigure(0, weight=1)
        self.seek_scale = ttk.Scale(
            seek, from_=0, to=0, orient="horizontal", variable=self.frame_value,
            command=self._on_seek)
        self.seek_scale.grid(row=0, column=0, sticky="ew")
        self.frame_label = ttk.Label(seek, textvariable=self.frame_text, anchor="e")
        self.frame_label.grid(row=0, column=1, padx=(12, 0), sticky="e")

        options = ttk.LabelFrame(
            self, text="時間與顯示", style="Section.TLabelframe")
        self.display_controls = options
        options.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 2))
        ttk.Label(options, text="播放間隔 (ms)", font=BODY_BOLD_FONT).grid(
            row=0, column=0, padx=(6, 4), pady=3, sticky="w")
        ttk.Spinbox(options, from_=50, to=60000, increment=50, width=8,
                    textvariable=self.interval_value).grid(
            row=0, column=1, padx=(0, 12), pady=3, sticky="w")
        ttk.Label(options, text="軌跡步數", font=BODY_BOLD_FONT).grid(
            row=0, column=2, padx=(0, 4), pady=3, sticky="w")
        trajectory_box = ttk.Spinbox(
            options, from_=0, to=100000, increment=1, width=8,
            textvariable=self.trajectory_value, command=self._on_render_option_changed)
        trajectory_box.grid(row=0, column=3, padx=(0, 12), pady=3, sticky="w")
        trajectory_box.bind("<Return>", self._on_render_option_changed)
        trajectory_box.bind("<FocusOut>", self._on_render_option_changed)
        ttk.Label(options, text="顯示模式", font=BODY_BOLD_FONT).grid(
            row=0, column=4, padx=(0, 4), pady=3, sticky="w")
        display_box = ttk.Combobox(options, textvariable=self.display_value,
                                   values=_DISPLAY_LABELS, state="readonly", width=13)
        display_box.grid(row=0, column=5, padx=(0, 8), pady=3, sticky="w")
        display_box.bind("<<ComboboxSelected>>", self._on_render_option_changed)
        self.legend_button = ttk.Checkbutton(
            options, text="顯示圖例（右上角）", variable=self.legend_value,
            command=self._on_legend_visibility_changed)
        self.legend_button.grid(row=0, column=6, padx=(4, 8), pady=3, sticky="w")
        ttk.Label(options, text="AoI 檢視", font=BODY_BOLD_FONT).grid(
            row=1, column=0, padx=(6, 4), pady=(0, 3), sticky="w")
        self.aoi_view_box = ttk.Combobox(
            options, textvariable=self.aoi_view_value, values=_AOI_VIEW_LABELS,
            state="disabled", width=17)
        self.aoi_view_box.grid(row=1, column=1, padx=(0, 8), pady=(0, 3), sticky="w")
        self.aoi_view_box.bind("<<ComboboxSelected>>", self._on_aoi_view_changed)
        options.columnconfigure(7, weight=1)

        ttk.Label(
            self, text="地圖快捷鍵：←/→ 換幀 · Space 播放 · Home/End 首尾",
            style="Muted.TLabel", wraplength=760, anchor="w",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 3))

        self.canvas_host = ttk.Frame(self)
        self.canvas_host.grid(row=5, column=0, sticky="nsew", padx=8, pady=4)
        self.canvas_host.columnconfigure(0, weight=1)
        self.canvas_host.rowconfigure(0, weight=1)
        self.empty_label = ttk.Label(
            self.canvas_host, text="匯入軌跡後，這裡會顯示初始畫面。", anchor="center")
        self.empty_label.grid(row=0, column=0, sticky="nsew")
        cost_box = ttk.LabelFrame(
            self, text="本步資訊", style="Section.TLabelframe")
        cost_box.grid(row=5, column=1, sticky="nsew", padx=(0, 8), pady=4)
        cost_box.columnconfigure(0, weight=1)
        cost_box.rowconfigure(0, weight=1)
        self.cost_textbox = tk.Text(
            cost_box, wrap="word", width=36, height=20, state="disabled",
            padx=8, pady=6, borderwidth=0,
        )
        self.cost_scrollbar = ttk.Scrollbar(
            cost_box, orient="vertical", command=self.cost_textbox.yview)
        self.cost_textbox.configure(yscrollcommand=self.cost_scrollbar.set)
        self.cost_textbox.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        self.cost_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.copy_info_button = ttk.Button(
            cost_box, text="複製本步資訊", command=self.copy_current_info,
            state="disabled")
        self.copy_info_button.grid(
            row=1, column=0, columnspan=2, sticky="e", padx=6, pady=(0, 6))
        ttk.Label(self, textvariable=self.status, anchor="w", style="Muted.TLabel").grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))

    # 選擇回合 parquet 檔案。
    def _browse_episode(self):
        self._browse("episode")

    # 選擇步驟 parquet 檔案。
    def _browse_step(self):
        self._browse("step")

    # 選擇指定軌跡檔，並嘗試配對同資料夾的另一個檔案。
    def _browse(self, kind):
        if kind == "episode":
            file_label = "回合"
        else:
            file_label = "步驟"
        selected = filedialog.askopenfilename(
            parent=self,
            title=f"選擇{file_label} parquet 檔",
            filetypes=(("Parquet 檔案", "*.parquet"), ("所有檔案", "*.*")),
        )
        if not selected:
            return
        if kind == "episode":
            self.episode_path.set(selected)
        else:
            self.step_path.set(selected)
        pair = detect_parquet_pair(selected)
        if pair:
            self.episode_path.set(str(pair[0]))
            self.step_path.set(str(pair[1]))
            self.status.set("已找到同資料夾中的回合檔與步驟檔。")
        else:
            self.status.set("請填入另一個檔案路徑，再載入資料。")

    # 確認兩個檔案路徑後啟動軌跡載入。
    def _load_from_fields(self):
        episode_path = self.episode_path.get().strip()
        step_path = self.step_path.get().strip()
        if not episode_path or not step_path:
            messagebox.showerror(
                "需要兩個軌跡檔",
                "請選擇回合 parquet 檔與步驟 parquet 檔。若檔名不同，請分別填入兩個路徑。",
                parent=self,
            )
            return
        self.load_paths(episode_path, step_path)

    # 在背景讀取並驗證軌跡檔，交由 Tk 主執行緒安裝。
    def load_paths(self, episode_path, step_path):
        """Read and validate a pair in a worker thread before installation."""
        if self._closed:
            raise RuntimeError("ReplayPanel is closed")
        try:
            trace_tool = self._get_trace_tool()
        except Exception as error:
            self._show_load_error(error)
            return False
        episode_path = Path(episode_path).expanduser()
        step_path = Path(step_path).expanduser()
        self.episode_path.set(str(episode_path))
        self.step_path.set(str(step_path))
        self.pause()
        self._load_generation += 1
        generation = self._load_generation
        self._load_pending = True
        result_queue = self._load_queue
        self.status.set("正在讀取 parquet 檔…")

        # 讀取並驗證兩份 parquet，將資料、回合 id 或錯誤送回佇列。
        def read_pair():
            try:
                episodes = trace_tool.load_episode_df(episode_path)
                steps = trace_tool.load_step_df(step_path)
                episode_ids = validate_trace_data(episodes, steps)
                result_queue.put((generation, episodes, steps, episode_ids, None))
            except Exception as error:  # forwarded to the Tk thread
                result_queue.put((generation, None, None, None, error))

        try:
            reader = threading.Thread(
                target=read_pair, name="replay-parquet-reader", daemon=True)
            reader.start()
        except Exception as error:
            self._load_pending = False
            self._show_load_error(error)
            return False
        if self._poll_after_id is None:
            self._schedule_poll()
        return True

    # 驗證並同步安裝已提供的軌跡資料表。
    def set_data(self, episode_df, step_df, episode=None):
        """Synchronously install validated frames; intended for GUI integration."""
        if self._closed:
            raise RuntimeError("ReplayPanel is closed")
        trace_tool = self._get_trace_tool()
        episode_ids = validate_trace_data(episode_df, step_df)
        # 保留原選擇；該回合不存在時回到資料中的首個回合。
        if episode in episode_ids:
            selected = episode
        else:
            selected = episode_ids[0]
        self._install_viewer(trace_tool, episode_df, step_df, episode_ids, selected)
        if (self._aoi_view_preference == "source"
                and self._viewer.aoi_view != "source"):
            self.status.set(
                f"已載入 {len(episode_ids)} 個回合；目前回合沒有完整裝置資料 AoI 記錄，"
                "顯示 UAV 收到的 AoI。"
            )
        else:
            self.status.set(f"已載入 {len(episode_ids)} 個回合。")
        return selected

    # 需要回放時才載入軌跡工具，缺少依賴時回報原因。
    def _get_trace_tool(self):
        if self._trace_tool is None:
            try:
                self._trace_tool = importlib.import_module("gui.trace_tool")
            except ModuleNotFoundError as error:
                dependency = error.name or "a replay dependency"
                raise RuntimeError(
                    f"軌跡播放需要 {dependency}。請安裝實驗功能所需套件：NumPy、pandas、Matplotlib，以及 PyArrow parquet 引擎。"
                ) from error
        return self._trace_tool

    # 建立可追蹤的 Tk 延遲回呼，關閉面板後不再執行。
    def _schedule(self, milliseconds, callback):
        if self._closed:
            return None
        holder = {}

        # 清除已執行的計時器記錄，面板未關閉時呼叫回呼。
        def run():
            after_id = holder.get("id")
            if after_id is not None:
                self._after_ids.discard(after_id)
            if not self._closed:
                callback()

        after_id = self.after(milliseconds, run)
        holder["id"] = after_id
        self._after_ids.add(after_id)
        return after_id

    # 排定下一次背景載入結果檢查。
    def _schedule_poll(self):
        self._poll_after_id = self._schedule(100, self._poll_load_queue)

    # 處理最新的載入結果，忽略已過期的讀檔請求。
    def _poll_load_queue(self):
        self._poll_after_id = None
        latest = None
        while True:
            try:
                result = self._load_queue.get_nowait()
            except queue.Empty:
                break
            if result[0] == self._load_generation:
                latest = result
        if latest is not None and not self._closed:
            _, episode_df, step_df, episode_ids, error = latest
            self._load_pending = False
            if error is not None:
                self._show_load_error(error)
            else:
                try:
                    # 讀取目前選擇一次，資料重載後仍保留有效的回合。
                    current_episode = self._selected_episode()
                    if current_episode in episode_ids:
                        selected = current_episode
                    else:
                        selected = episode_ids[0]
                    self._install_viewer(self._get_trace_tool(), episode_df,
                                         step_df, episode_ids, selected)
                    if (self._aoi_view_preference == "source"
                            and self._viewer.aoi_view != "source"):
                        self.status.set(
                            f"已載入 {len(episode_ids)} 個回合；目前回合沒有完整裝置資料 AoI 記錄，"
                            "顯示 UAV 收到的 AoI。"
                        )
                    else:
                        self.status.set(f"已載入 {len(episode_ids)} 個回合。")
                except Exception as install_error:
                    self._show_load_error(install_error)
        if self._load_pending and not self._closed:
            self._schedule_poll()

    # 在狀態列與對話框顯示載入失敗原因。
    def _show_load_error(self, error):
        detail = str(error).strip() or error.__class__.__name__
        self.status.set("載入軌跡失敗：" + detail)
        messagebox.showerror("載入軌跡失敗", detail, parent=self)

    # 先建立新圖與檢視器，成功後替換目前回放內容。
    def _install_viewer(self, trace_tool, episode_df, step_df, episode_ids, selected):
        # Build the new viewer and canvas before removing the existing trace, so
        # a rendering error cannot leave the user with a blank or stale panel.
        try:
            Figure = importlib.import_module("matplotlib.figure").Figure
            FigureCanvasTkAgg = importlib.import_module(
                "matplotlib.backends.backend_tkagg").FigureCanvasTkAgg
        except ModuleNotFoundError as error:
            dependency = error.name or "Matplotlib"
            raise RuntimeError(
                f"嵌入式軌跡播放需要 {dependency}。請安裝 Matplotlib 與 Tk 後端。"
            ) from error
        new_figure = Figure(figsize=(7, 5.5), dpi=100)
        new_canvas = None
        new_viewer = None
        mode = self._display_mode()
        interval = self._validated_integer(self.interval_value.get(), "播放間隔", 50)
        trajectory = self._validated_integer(
            self.trajectory_value.get(), "軌跡步數", 0)
        try:
            new_canvas = FigureCanvasTkAgg(new_figure, master=self.canvas_host)
            new_viewer = trace_tool.EpisodeStepViewer(
                episode=selected, episode_df=episode_df, step_df=step_df,
                interval=interval, display=mode, trajectory=trajectory,
                figure=new_figure, controls=False, cost_panel=False,
                show_legend=self.legend_value.get(),
            )
            if (self._aoi_view_preference == "source"
                    and new_viewer.source_aoi_available):
                new_viewer.set_aoi_view("source")
            new_widget = new_canvas.get_tk_widget()
            # 快捷鍵只綁在地圖畫布，讓輸入欄位保留原來的文字操作。
            new_widget.configure(takefocus=True)
            new_widget.bind("<Button-1>", lambda _event: new_widget.focus_set(), add="+")
            new_widget.bind("<KeyPress>", self._on_canvas_key, add="+")
            self._cancel_canvas_callbacks(new_canvas)
            new_canvas.draw()
            new_widget.grid(row=0, column=0, sticky="nsew")
        except Exception:
            if new_viewer is not None:
                new_viewer.close()
            if new_canvas is not None:
                self._cancel_canvas_callbacks(new_canvas)
                try:
                    new_canvas.get_tk_widget().destroy()
                except tk.TclError:
                    pass
            new_figure.clear()
            raise

        self.pause()
        old_viewer, old_canvas, old_figure = self._viewer, self._canvas, self._figure
        if old_viewer is not None:
            old_viewer.close()
        if old_canvas is not None:
            self._cancel_canvas_callbacks(old_canvas)
            try:
                old_canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
        if old_figure is not None:
            old_figure.clear()
        if self.empty_label.winfo_exists():
            self.empty_label.grid_remove()

        self._trace_tool = trace_tool
        self._episode_df, self._step_df = episode_df, step_df
        self._episode_ids = list(episode_ids)
        self._viewer, self._canvas, self._figure = new_viewer, new_canvas, new_figure
        self.aoi_view_value.set(_AOI_MODE_TO_VIEW[new_viewer.aoi_view])
        labels = [str(episode_id) for episode_id in self._episode_ids]
        self.episode_box.configure(values=labels)
        self.episode_choice.set(str(selected))
        self._update_frame_controls()
        self._update_cost_sidebar(reset_view=True)

    # 取消 Matplotlib 的 Tk 延遲回呼，避免銷毀後繼續執行。
    @staticmethod
    def _cancel_canvas_callbacks(canvas):
        """Cancel Matplotlib's deferred Tk callbacks before destroying its widget."""
        try:
            widget = canvas.get_tk_widget()
        except tk.TclError:
            return
        for attribute in ("_idle_draw_id", "_event_loop_id"):
            callback_id = getattr(canvas, attribute, None)
            if callback_id is None:
                continue
            try:
                widget.after_cancel(callback_id)
            except tk.TclError:
                pass
            setattr(canvas, attribute, None)

    # 將輸入轉為整數並檢查指定下限。
    @staticmethod
    def _validated_integer(value, label, minimum):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label}必須是整數") from None
        if parsed < minimum:
            raise ValueError(f"{label}不可小於 {minimum}")
        return parsed

    # 將下拉選單文字轉回原始回合識別值。
    def _selected_episode(self):
        label = self.episode_choice.get()
        for episode_id in self._episode_ids:
            if str(episode_id) == label:
                return episode_id
        return None

    # 將介面選項轉成檢視器使用的顯示模式。
    def _display_mode(self):
        value = self.display_value.get()
        if value in _DISPLAY_MODES:
            return value
        return _DISPLAY_LABEL_TO_MODE.get(value, "all")

    # 切換收集端／來源 AoI，不重設影格或中斷播放。
    def _on_aoi_view_changed(self, _event=None):
        if self._viewer is None:
            self.aoi_view_value.set(_AOI_MODE_TO_VIEW["uav"])
            return
        requested = _AOI_VIEW_TO_MODE.get(self.aoi_view_value.get(), "uav")
        if requested == "source" and not self._viewer.source_aoi_available:
            self.aoi_view_value.set(_AOI_MODE_TO_VIEW[self._viewer.aoi_view])
            self.status.set(
                f"回合 {self._viewer.episode} 沒有完整的裝置資料 AoI 記錄，"
                "目前仍顯示 UAV 收到的 AoI。"
            )
            return
        self._aoi_view_preference = requested
        self._viewer.set_aoi_view(requested)
        self.aoi_view_value.set(_AOI_MODE_TO_VIEW[self._viewer.aoi_view])
        if self._canvas is not None:
            self._canvas.draw_idle()

    # 切換到選取回合，重新建立回放內容。
    def _on_episode_changed(self, _event=None):
        if self._episode_df is None or self._step_df is None:
            return
        selected = self._selected_episode()
        if selected is None:
            return
        try:
            self._install_viewer(self._get_trace_tool(), self._episode_df,
                                 self._step_df, self._episode_ids, selected)
            if (self._aoi_view_preference == "source"
                    and self._viewer.aoi_view != "source"):
                self.status.set(
                    f"回合 {selected} 沒有完整的裝置資料 AoI 記錄，"
                    "目前顯示 UAV 收到的 AoI。"
                )
            else:
                self.status.set(f"目前顯示回合 {selected}。")
        except Exception as error:
            if self._viewer is not None:
                self.episode_choice.set(str(self._viewer.episode))
            self._show_load_error(error)

    # 依目前回合重設影格滑桿與播放按鈕。
    def _update_frame_controls(self):
        if self._viewer is None:
            frame_count = 0
            episode_state = "disabled"
            playback_state = "disabled"
            info_state = "disabled"
            self.episode_choice.set("")
            self.step_jump_value.set("")
        else:
            frame_count = self._viewer.frame_count
            episode_state = "readonly"
            if frame_count == 0:
                playback_state = "disabled"
            else:
                playback_state = "normal"
            info_state = "normal"
        self.episode_box.configure(state=episode_state)
        self.aoi_view_box.configure(
            state="readonly" if self._viewer is not None else "disabled")
        for widget in (
                self.prev_button, self.play_button, self.next_button,
                self.start_button, self.reset_button, self.seek_scale,
                self.loop_button, self.step_entry, self.jump_button):
            widget.configure(state=playback_state)
        self.export_button.configure(state=info_state)
        self.copy_info_button.configure(state=info_state)
        self.frame_value.set(0)
        self.seek_scale.configure(from_=0, to=frame_count)
        self._update_frame_text()
        self.play_button.configure(text="播放")

    # 顯示目前影格位置及軌跡中的實際步數。
    def _update_frame_text(self):
        if self._viewer is None:
            self.frame_text.set("影格 0 / 0")
            return
        frame = self._viewer.step
        count = self._viewer.frame_count
        if frame:
            actual = self._viewer.step_data["step"][frame - 1]
            self.frame_text.set(f"影格 {frame} / {count} · 步驟 {actual}")
        else:
            actual = self._viewer.episode_data["initial_step"]
            self.frame_text.set(f"影格 0 / {count} · 步驟 {actual}")
        self.step_jump_value.set(str(actual))

    # 依實際記錄步數定位影格，保留稀疏步數間的索引關係。
    def _jump_to_step(self, _event=None):
        if self._viewer is None:
            self.status.set("請先載入回放資料。")
            return
        try:
            actual_step = int(self.step_jump_value.get().strip())
        except (TypeError, ValueError, OverflowError):
            self.status.set("實際步數必須是整數。")
            return

        initial_step = self._viewer.episode_data["initial_step"]
        frame = _frame_for_step(
            self._viewer.step_data["step"], actual_step, initial_step)
        if frame is None:
            self.status.set(f"找不到步數 {actual_step}。")
            return

        self.pause()
        self._set_frame(frame)
        self.status.set(f"已跳至步驟 {actual_step}。")

    # 將目前已顯示的回放畫面存成使用者指定的 PNG 檔案。
    def export_png(self):
        if self._viewer is None or self._figure is None or self._canvas is None:
            self.status.set("請先載入回放資料。")
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="匯出目前回放畫面",
            defaultextension=".png",
            filetypes=(("PNG 圖片", "*.png"),),
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".png":
            self.status.set("PNG 匯出檔名必須使用 .png 副檔名。")
            return
        try:
            self._canvas.draw()
            saving.save_figure(self._figure, path, image_format="png")
        except Exception as error:
            detail = str(error).strip() or error.__class__.__name__
            self.status.set("PNG 匯出失敗：" + detail)

    # 只有可見地圖接收快捷鍵，切換頁籤後不操作隱藏的回放。
    def _on_canvas_key(self, event):
        if self._closed or self._viewer is None or self._canvas is None:
            return
        widget = self._canvas.get_tk_widget()
        if event.widget is not widget or not widget.winfo_ismapped():
            return
        if event.keysym == "Left":
            self.previous()
        elif event.keysym == "Right":
            self.next()
        elif event.keysym == "Home":
            self.reset()
        elif event.keysym == "End":
            self.pause()
            self._set_frame(self._viewer.frame_count)
        elif event.keysym == "space":
            self.play_pause()
        else:
            return
        return "break"

    # 暫停播放，跳到滑桿指定的影格。
    def _on_seek(self, value):
        if self._viewer is None:
            return
        self.pause()
        self._set_frame(int(round(float(value))))

    # 切換影格並更新圖形、成本資訊與步數文字。
    def _set_frame(self, frame):
        if self._viewer is None:
            return
        self._viewer.set_frame(frame)
        self.frame_value.set(self._viewer.step)
        self._update_cost_sidebar()
        if self._canvas is not None:
            self._canvas.draw_idle()
        self._update_frame_text()

    # 暫停播放並移到前一個影格。
    def previous(self):
        self.pause()
        if self._viewer is not None:
            self._set_frame(self._viewer.step - 1)

    # 暫停播放並移到下一個影格。
    def next(self):
        self.pause()
        if self._viewer is not None:
            self._set_frame(self._viewer.step + 1)

    # 暫停播放並回到初始影格。
    def reset(self):
        self.pause()
        self._set_frame(0)

    # 從初始影格重新開始播放。
    def start(self):
        if self._viewer is None or not self._viewer.frame_count:
            return
        self.pause()
        self._set_frame(0)
        self._start_playback()

    # 切換播放或暫停，播完時從初始影格重播。
    def play_pause(self):
        if self._play_after_id is not None:
            self.pause()
            return
        if self._viewer is None or not self._viewer.frame_count:
            return
        if self._viewer.step >= self._viewer.frame_count:
            self._set_frame(0)
        self._start_playback()

    # 檢查播放間隔，標記播放狀態並排定下一影格。
    def _start_playback(self):
        if self._closed or self._viewer is None or not self._viewer.frame_count:
            return
        try:
            self._validated_integer(self.interval_value.get(), "播放間隔", 50)
        except ValueError as error:
            self.status.set(str(error))
            self.pause()
            return
        self._viewer.playing = True
        self.play_button.configure(text="暫停")
        self._schedule_play_tick()

    # 依播放間隔排定下一次影格更新。
    def _schedule_play_tick(self):
        try:
            interval = self._validated_integer(self.interval_value.get(), "播放間隔", 50)
        except ValueError as error:
            self.status.set(str(error))
            self.pause()
            return
        self._play_after_id = self._schedule(interval, self._advance_playback)

    # 循環模式在末幀停留一個播放間隔，再回到初始幀。
    def _advance_playback(self):
        self._play_after_id = None
        if self._closed or self._viewer is None or not self._viewer.playing:
            return
        if self._viewer.step >= self._viewer.frame_count:
            if self.loop_value.get() and self._viewer.frame_count:
                self._set_frame(0)
                self._schedule_play_tick()
            else:
                self.pause()
            return
        self._set_frame(self._viewer.step + 1)
        if self._viewer.step >= self._viewer.frame_count and not self.loop_value.get():
            self.pause()
            return
        self._schedule_play_tick()

    # 取消指定的 Tk 計時器並移除追蹤記錄。
    def _cancel_after(self, after_id):
        if after_id is None:
            return
        self._after_ids.discard(after_id)
        try:
            self.after_cancel(after_id)
        except tk.TclError:
            pass

    # 停止播放計時器，將控制按鈕恢復為播放。
    def pause(self):
        self._cancel_after(self._play_after_id)
        self._play_after_id = None
        if self._viewer is not None:
            self._viewer.pause()
        if not self._closed and hasattr(self, "play_button"):
            self.play_button.configure(text="播放")

    # 套用顯示模式與軌跡長度，重繪目前影格。
    def _on_render_option_changed(self, _event=None):
        if self._viewer is None:
            return
        self.pause()
        mode = self._display_mode()
        try:
            trajectory = self._validated_integer(
                self.trajectory_value.get(), "軌跡步數", 0)
        except ValueError as error:
            self.status.set(str(error))
            return
        self._viewer.display = mode
        self._viewer.trajectory = trajectory
        self._viewer.set_frame(self._viewer.step)
        self._update_cost_sidebar()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # 即時切換圖例，不暫停播放或改變目前影格。
    def _on_legend_visibility_changed(self):
        if self._viewer is None:
            return
        self._viewer.set_legend_visible(self.legend_value.get())
        if self._canvas is not None:
            self._canvas.draw_idle()

    # 更新唯讀成本資訊；重設時回到頂端，否則保留目前捲動位置。
    def _update_cost_sidebar(self, reset_view=False):
        if self._viewer is None:
            return
        try:
            if reset_view:
                top = 0.0
            else:
                top = self.cost_textbox.yview()[0]
            self.cost_textbox.configure(state="normal")
            self.cost_textbox.delete("1.0", "end")
            self.cost_textbox.insert("1.0", self._viewer.cost_text)
            self.cost_textbox.configure(state="disabled")
            self.cost_textbox.yview_moveto(top)
        except tk.TclError:
            pass

    # 複製成本資訊側欄目前顯示的完整文字。
    def copy_current_info(self):
        if self._viewer is None:
            self.status.set("請先載入回放資料。")
            return
        current_info = self.cost_textbox.get("1.0", "end-1c")
        try:
            self.clipboard_clear()
            self.clipboard_append(current_info)
        except tk.TclError as error:
            detail = str(error).strip() or error.__class__.__name__
            self.status.set("複製本步資訊失敗：" + detail)
            return
        self.status.set("已複製本步資訊。")

    # 取消計時器並釋放檢視器、圖形與載入狀態。
    def close(self):
        """Cancel timers and dispose the embedded canvas; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        try:
            self.copy_info_button.configure(state="disabled")
        except tk.TclError:
            pass
        self._load_generation += 1
        self._load_pending = False
        self._load_queue = queue.Queue()
        self.pause()
        for after_id in tuple(self._after_ids):
            self._cancel_after(after_id)
        self._poll_after_id = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._canvas is not None:
            self._cancel_canvas_callbacks(self._canvas)
            try:
                self._canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass
            self._canvas = None
        if self._figure is not None:
            self._figure.clear()
            self._figure = None
        self._episode_df = self._step_df = None
        self._episode_ids = []
        self._trace_tool = None
        try:
            self._update_frame_controls()
        except tk.TclError:
            # A destroy event may arrive after Tk has already removed child controls.
            pass

    # 面板本身被銷毀時清理回放資源。
    def _on_destroy(self, event):
        if event.widget is self:
            self.close()


__all__ = ["ReplayPanel", "detect_parquet_pair", "validate_trace_data"]
