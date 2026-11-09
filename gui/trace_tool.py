import math
import numbers
from pathlib import Path
from decimal import Decimal, InvalidOperation
import json
import numpy as np
import pandas as pd

from runtime import fileIO as saving
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.artist import Artist
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox


# 輪流選取裝置並朝該裝置移動，供軌跡示範使用。
def fixed_policy(env, step):
    """Demo policy: cycle through devices and move toward the selected device."""
    actions = []
    for u in range(env.U):
        # 每個 UAV 錯開一個裝置，隨步數輪流選取目標。
        target = (step + u) % env.M
        move = env.device_coord[target] - env.UAV_coord[u]
        actions.append((target, move))
    return actions


# 執行指定步數，保存初始狀態與逐步動作、獎勵快照。
def record_trace(env, policy=fixed_policy, steps=100, episode=1, reset=True):
    """Record a finite viewing segment, without terminating the environment.

    policy(env, index) returns actions accepted by env.step(actions).
    Snapshots are independent copies; movement shown by the viewer is actual
    movement after the environment applies speed and map limits.
    """
    if reset:
        env.reset()
    episode_record = {
        "episode": episode,
        "init_UAV_pos": env.UAV_coord.tolist(),
        "init_device_pos": env.device_coord.tolist(),
        "init_device_AOI": env.AOI.tolist(),
        "init_source_AOI": env.source_AOI.tolist(),
        "risk_region": env.risk_region.tolist(),
        "map_size": env.map_size,
        "initial_step": env.steps,
    }
    episode_record.update(saving._communication_episode_record(env))
    episode_df = pd.DataFrame([episode_record])
    rows = []
    for index in range(steps):
        actions = policy(env, index)
        selected = [int(action[0]) for action in actions]
        moves = [np.asarray(action[1], dtype=float).tolist() for action in actions]
        env.step(actions)
        rewards = env.rewards
        row = {
            "episode": episode, "step": env.steps,
            "UAV_pos": env.UAV_coord.tolist(),
            "device_pos": env.device_coord.tolist(),
            "device_AOI": env.AOI.tolist(),
            "source_AOI": env.source_AOI.tolist(),
            "UAV_action_select": selected, "UAV_action_move": moves,
            "reward": list(rewards), "total_reward": env.total_reward,
        }
        row.update(saving._communication_step_record(env))
        rows.append(row)
    columns = ["episode", "step", "UAV_pos", "device_pos", "device_AOI", "source_AOI",
               "UAV_action_select", "UAV_action_move", "reward", "total_reward"]
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return episode_df, pd.DataFrame(rows, columns=columns)


# 將 parquet 座標欄位整理成二維座標列表。
def _coords(value):
    # Parquet may return an object array of row arrays instead of a 2-D array.
    return np.asarray(list(value), dtype=float).reshape(-1, 2).tolist()


# 讀取Episode
def load_episode_df(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 episode parquet：{path}")

    df = saving.read_parquet(path, engine="pyarrow")

    required = ["episode", "init_UAV_pos", "init_device_pos", "risk_region"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"episode parquet 缺少必要欄位：{col}")

    df = _enrich_episode_metadata(df, path)
    # 整場耗時在 checkpoint 保存完成後記錄，只讀取同一份軌跡旁的結果。
    runtime = _read_json_object(path.parent / "runtime.json")
    if runtime is not None:
        for column, key in (("training_wall_time_seconds", "training_wall_time_seconds"),
                            ("total_wall_time_seconds", "wall_time_seconds")):
            seconds = _parse_wall_time(runtime.get(key))
            if seconds is not None:
                if column not in df:
                    df[column] = seconds
                else:
                    df[column] = df[column].fillna(seconds)
    return df


# 從標準 run 設定讀取保存的服務範圍資訊。
def _episode_metadata(path):
    metadata = {}
    config_paths = [path.parent / "config.json"]
    if (path.parent.name == "Stored_Datas"
            and len(path.parent.parents) >= 3
            and path.parent.parents[1].name == "results"):
        # Standard run layout: <run>/results/<method>/Stored_Datas/episode.parquet.
        config_paths.append(path.parent.parents[2] / "config.json")
    config_path = next((candidate for candidate in config_paths if candidate.is_file()), None)
    if config_path is not None:
        config = _read_json_object(config_path)
    else:
        config = None
    if config is not None:
        method = config.get("method")
        if method not in ("DQN", "DQN_greedy", "DQN_service", "HRL", "HRL_PPO", "HRL_original", "HRL_risk", "HRL_uav_risk", "HRL_RDSAC", "HRL-SA", "HRL-MA", "MA-CQR"):
            return metadata
        settings = config.get("settings")
        if not isinstance(settings, dict):
            settings = {}
        environment = settings.get("ENV")
        if isinstance(environment, dict):
            metadata["map_size"] = _positive_metadata_number(environment.get("map_size"))

        method_groups = {
            "DQN_greedy": ("DQN_GREEDY", "within_range"),
            "DQN_service": ("DQN_SERVICE", "force_within_range_allow_early"),
            "HRL": ("HRL", "within_range"),
            "HRL_PPO": ("HRL_PPO", "within_range_inclusive"),
            "HRL_original": ("HRL", "within_range"),
            "HRL_risk": ("HRL", "within_range"),
            "HRL_uav_risk": ("HRL", "within_range"),
            "HRL_RDSAC": ("HRL", "within_range"),
            "HRL-SA": ("HRL", "within_range"),
            "HRL-MA": ("HRL", "within_range"),
        }
        details = method_groups.get(method)
        if details is not None:
            group, rule = details
            method_settings = settings.get(group)
            if isinstance(method_settings, dict):
                ratio = _positive_metadata_number(method_settings.get("service_ratio"))
                if ratio is not None and ratio <= 1:
                    metadata["service_ratio"] = ratio
                    metadata.setdefault("service_rule", rule)
    return metadata


# 只讀取 JSON 物件；損壞的可選 metadata 不會猜測目前預設值。
def _read_json_object(path):
    if path is None:
        return None
    try:
        value = saving.read_json(path, encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(value, dict):
        return value
    else:
        return None


# 將已保存的設定數字轉成有限正數，拒絕 bool 與超出範圍的比例。
def _positive_metadata_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


# 將保存的地圖尺寸與服務範圍補入舊 episode 列，不覆蓋有效逐列資料。
def _enrich_episode_metadata(df, path):
    metadata = _episode_metadata(path)
    if not metadata:
        return df

    has_map_size = "map_size" in df.columns
    if has_map_size:
        map_sizes = df["map_size"].tolist()
    else:
        map_sizes = [None] * len(df)
    has_radius = "service_radius" in df.columns
    if has_radius:
        radii = df["service_radius"].tolist()
    else:
        radii = [None] * len(df)
    has_rule = "service_rule" in df.columns
    if has_rule:
        rules = df["service_rule"].tolist()
    else:
        rules = [None] * len(df)

    for index in range(len(df)):
        map_size = map_sizes[index]
        if _is_missing_scalar(map_size) and metadata.get("map_size") is not None:
            map_size = metadata["map_size"]
            map_sizes[index] = map_size

        radius = radii[index]
        if _is_missing_scalar(radius):
            fallback_radius = None
            ratio = metadata.get("service_ratio")
            actual_map_size = _positive_metadata_number(map_size)
            if ratio is not None and actual_map_size is not None:
                fallback_radius = ratio * actual_map_size
            if fallback_radius is not None:
                radii[index] = fallback_radius

        rule = rules[index]
        if _is_missing_scalar(rule) and metadata.get("service_rule") is not None:
            rules[index] = metadata["service_rule"]

    if not has_map_size and any(value is not None for value in map_sizes):
        df["map_size"] = map_sizes
    elif has_map_size and any(
            _is_missing_scalar(original) and not _is_missing_scalar(updated)
            for original, updated in zip(df["map_size"].tolist(), map_sizes)):
        df["map_size"] = map_sizes
    if has_radius or any(value is not None for value in radii):
        df["service_radius"] = radii
    if has_rule or any(value is not None for value in rules):
        df["service_rule"] = rules
    return df


# 讀取Step
def load_step_df(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 step parquet：{path}")

    df = saving.read_parquet(path, engine="pyarrow")

    required = ["episode", "step", "UAV_pos", "device_pos", "device_AOI",
                "UAV_action_select", "UAV_action_move", "reward"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"step parquet 缺少必要欄位：{col}")

    return df

# 取得指定回合的初始資料列。
def get_episode_init_row(episode_df, episode_id):
    matched = episode_df[episode_df["episode"] == episode_id]
    if len(matched) == 0:
        raise ValueError(f"在 episode parquet 中找不到 episode = {episode_id}")
    return matched.iloc[0]


# 將步數解析成有限整數，接受數字字串但拒絕布林值與小數。
def _parse_step_id(value, label):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite integer")

    # float 只檢查原有數值範圍；整數從原值轉換，避免大型步數失真。
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


# 驗證步數唯一且晚於保存的初始環境步數。
def _parse_step_ids(values, initial_step, label):
    initial_step = _parse_step_id(initial_step, f"{label} initial_step")
    if initial_step < 0:
        raise ValueError(f"{label} initial_step must be a non-negative integer")
    parsed_steps = []
    seen_steps = set()
    for row_index, value in enumerate(values):
        step = _parse_step_id(value, f"{label} step row {row_index} step")
        if step <= initial_step:
            raise ValueError(
                f"{label} step {step} must be greater than initial_step {initial_step}"
            )
        if step in seen_steps:
            raise ValueError(f"{label} contains duplicate step id {step}")
        seen_steps.add(step)
        parsed_steps.append(step)
    return parsed_steps


# 取得指定回合的步驟資料，依步數排序並建立影格索引。
def get_episode_step_df(step_df, episode_id, initial_step=0):
    matched = step_df[step_df["episode"] == episode_id].copy()
    matched["step"] = _parse_step_ids(
        matched["step"].tolist(), initial_step, f"episode {episode_id}"
    )
    matched = matched.sort_values("step").reset_index(drop=True)
    matched["frame_idx"] = np.arange(len(matched))
    return matched

# 整理回合初始資料，補齊可選欄位的預設值。
def parse_episode_data(episode_data):
    init_UAV_pos = episode_data["init_UAV_pos"]
    init_UAV_pos = _coords(init_UAV_pos)

    initial_step = episode_data.get("initial_step")
    if _is_missing_scalar(initial_step):
        initial_step = 0
    initial_step = _parse_step_id(initial_step, "initial_step")
    if initial_step < 0:
        raise ValueError("initial_step must be a non-negative integer")
    init_device_pos = _coords(episode_data["init_device_pos"])
    init_device_AOI = episode_data.get("init_device_AOI")
    if _is_missing_scalar(init_device_AOI):
        if initial_step > 0:
            raise ValueError(
                "init_device_AOI is required when initial_step is greater than zero"
            )
        init_device_AOI = np.ones(len(init_device_pos))
    init_device_AOI = _parse_aoi_vector(
        init_device_AOI, "init_device_AOI", len(init_device_pos)
    )
    init_source_AOI = episode_data.get("init_source_AOI")
    if not _is_missing_scalar(init_source_AOI):
        init_source_AOI = _parse_aoi_vector(
            init_source_AOI, "init_source_AOI", len(init_device_pos)
        )
        if any(source > collector for source, collector in zip(
                init_source_AOI, init_device_AOI)):
            raise ValueError("init_device_AOI must be greater than or equal to init_source_AOI")
    else:
        init_source_AOI = None
    map_size = episode_data.get("map_size")
    if _is_missing_scalar(map_size):
        map_size = 10.0
    service_radius = _parse_service_radius(episode_data.get("service_radius"))
    service_rule = _parse_service_rule(episode_data.get("service_rule"))
    communication_model = _parse_optional_string(
        episode_data.get("communication_model"), "communication_model"
    )

    return {
        "episode": episode_data["episode"],
        "init_UAV_pos": init_UAV_pos,
        "init_device_pos": init_device_pos,
        "risk_region": np.asarray(list(episode_data["risk_region"]), dtype=float).reshape(-1, 4).tolist(),
        "init_device_AOI": init_device_AOI,
        "init_source_AOI": init_source_AOI,
        "map_size": float(map_size),
        "initial_step": initial_step,
        "service_radius": service_radius,
        "service_rule": service_rule,
        "communication_model": communication_model,
        "episode_wall_time_seconds": _parse_wall_time(episode_data.get("episode_wall_time_seconds")),
        "wall_time_seconds": _parse_wall_time(episode_data.get("wall_time_seconds")),
        "training_wall_time_seconds": _parse_wall_time(episode_data.get("training_wall_time_seconds")),
        "total_wall_time_seconds": _parse_wall_time(episode_data.get("total_wall_time_seconds")),
    }


# 判斷單一欄位是否缺值，避免將陣列轉成布林值。
def _is_missing_scalar(value):
    """Recognize scalar null metadata without coercing array values to bool."""
    if value is None or value.__class__.__name__ == "NAType":
        return True
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return False
    if isinstance(value, numbers.Real):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, Decimal):
        return value.is_nan()
    return False


# 驗證 AoI 向量為每個裝置一個有限正數。
def _parse_aoi_vector(value, column, expected_length):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{column} must be a finite positive numeric vector")
    try:
        values = list(value)
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be a finite positive numeric vector") from None
    if len(values) != expected_length:
        raise ValueError(
            f"{column} must contain {expected_length} values (one per device); "
            f"got {len(values)}"
        )
    parsed = []
    for index, item in enumerate(values):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{column}[{index}] must be a finite positive number")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{column}[{index}] must be a finite positive number") from None
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{column}[{index}] must be a finite positive number")
        parsed.append(number)
    return parsed


# 時間是可選的歷史紀錄，缺值或無效值不以目前設定或檔案日期推算。
def _parse_wall_time(value):
    if _is_missing_scalar(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


# 長訓練使用時分秒顯示，並保留毫秒與原始秒數以便比較。
def _format_wall_time(seconds):
    if seconds is None:
        return "未記錄"
    milliseconds = round(seconds * 1000)
    hours, remaining = divmod(milliseconds, 3600000)
    minutes, remaining = divmod(remaining, 60000)
    whole_seconds, milliseconds = divmod(remaining, 1000)
    return (f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d} "
            f"({seconds:.3f} s)")


# 保留缺省服務半徑為 None，拒絕明確記錄的非正數或非有限值。
def _parse_service_radius(value):
    if _is_missing_scalar(value):
        return None
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool):
        raise ValueError("service_radius must be a finite positive number")
    try:
        radius = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("service_radius must be a finite positive number") from None
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("service_radius must be a finite positive number")
    return radius


# 未知未來服務規則仍可使用通用圖例；型別錯誤則在載入時指出。
def _parse_service_rule(value):
    if _is_missing_scalar(value):
        return None
    if not isinstance(value, str):
        raise ValueError("service_rule must be a string")
    return value


# 新的逐步通訊中繼資料可省略，非空值需保留為字串。
def _parse_optional_string(value, column):
    if _is_missing_scalar(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{column} must be a string")
    return value

# 將逐步資料整理成檢視器使用的欄位列表。
def _parse_optional_reward(value, column, expected_length):
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{column} must be a finite numeric vector")
    try:
        values = list(value)
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be a finite numeric vector") from None
    if len(values) != expected_length:
        raise ValueError(
            f"{column} must contain {expected_length} values (one per UAV); "
            f"got {len(values)}"
        )

    rewards = []
    for index, item in enumerate(values):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{column}[{index}] must be a finite number")
        try:
            reward = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{column}[{index}] must be a finite number") from None
        if not math.isfinite(reward):
            raise ValueError(f"{column}[{index}] must be a finite number")
        rewards.append(reward)
    return rewards


# 解析可缺少欄列的每 UAV 非負功率或傳輸時間。
def _parse_optional_nonnegative_vector(value, column, expected_length):
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{column} must be a finite non-negative numeric vector")
    try:
        values = list(value)
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be a finite non-negative numeric vector") from None
    if len(values) != expected_length:
        raise ValueError(
            f"{column} must contain {expected_length} values (one per UAV); got {len(values)}"
        )
    parsed = []
    for index, item in enumerate(values):
        if _is_missing_scalar(item):
            parsed.append(None)
            continue
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{column}[{index}] must be finite and non-negative")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{column}[{index}] must be finite and non-negative") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{column}[{index}] must be finite and non-negative")
        parsed.append(number)
    return parsed


# 驗證每 UAV 的二元上層動作標記。
def _parse_flag_vector(value, column, expected_length):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{column} must contain one boolean per UAV")
    if len(value) != expected_length or any(
            not isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError(f"{column} must contain one boolean per UAV")
    return [bool(item) for item in value]


def parse_step_data(step_data, initial_step=0):
    keys = ["step", "UAV_pos", "device_pos", "device_AOI", "source_AOI",
            "UAV_action_select", "UAV_action_move", "reward"]
    optional_reward_columns = ("upper_reward", "intrinsic_reward")
    optional_vector_columns = (
        "UAV_service_success", "UAV_transmission_power",
        "UAV_transmission_peak_power", "UAV_transmission_time",
    )
    step_all = {key: [] for key in keys}
    for column in optional_reward_columns:
        step_all[column] = []
    for column in optional_vector_columns:
        if column in step_data.columns:
            step_all[column] = []
    has_communication_scheme = "communication_scheme" in step_data.columns
    if has_communication_scheme:
        step_all["communication_scheme"] = []
    has_idle = "UAV_idle" in step_data.columns
    has_current_position = "UAV_current_position" in step_data.columns
    if has_idle:
        step_all["UAV_idle"] = []
    if has_current_position:
        step_all["UAV_current_position"] = []

    parsed_steps = _parse_step_ids(
        step_data["step"].tolist(), initial_step, "step data"
    )
    for (_, step_row), step in zip(step_data.iterrows(), parsed_steps):

        UAV_pos = step_row["UAV_pos"]
        UAV_pos = _coords(UAV_pos)

        device_pos = _coords(step_row["device_pos"])

        UAV_action_move = _coords(step_row["UAV_action_move"])
        

        step_all["step"].append(step)
        step_all["UAV_pos"].append(UAV_pos)
        step_all["device_pos"].append(device_pos)
        collector_aoi = _parse_aoi_vector(
            step_row["device_AOI"], "device_AOI", len(device_pos)
        )
        step_all["device_AOI"].append(collector_aoi)
        source_aoi = None
        if "source_AOI" in step_data.columns:
            source_value = step_row["source_AOI"]
            if not _is_missing_scalar(source_value):
                source_aoi = _parse_aoi_vector(
                    source_value, "source_AOI", len(device_pos)
                )
                if any(source > collector for source, collector in zip(
                        source_aoi, collector_aoi)):
                    raise ValueError(
                        "device_AOI must be greater than or equal to source_AOI"
                    )
        step_all["source_AOI"].append(source_aoi)
        step_all["UAV_action_select"].append([int(v) for v in step_row["UAV_action_select"]])
        step_all["UAV_action_move"].append(UAV_action_move)
        rewards = list(step_row["reward"])
        step_all["reward"].append(rewards)
        if has_idle:
            idle_value = step_row["UAV_idle"]
            if (_is_missing_scalar(idle_value) and has_current_position
                    and not _is_missing_scalar(step_row["UAV_current_position"])):
                idle_flags = [False] * len(UAV_pos)
            else:
                idle_flags = _parse_flag_vector(
                    idle_value, "UAV_idle", len(UAV_pos)
                )
            step_all["UAV_idle"].append(idle_flags)
        if has_current_position:
            current_position_value = step_row["UAV_current_position"]
            if (_is_missing_scalar(current_position_value) and has_idle
                    and not _is_missing_scalar(step_row["UAV_idle"])):
                current_position_flags = [False] * len(UAV_pos)
            else:
                current_position_flags = _parse_flag_vector(
                    current_position_value,
                    "UAV_current_position",
                    len(UAV_pos),
                )
            step_all["UAV_current_position"].append(current_position_flags)
        for column in optional_reward_columns:
            if column in step_data.columns:
                value = step_row[column]
            else:
                value = None
            step_all[column].append(
                _parse_optional_reward(value, column, len(UAV_pos))
            )
        if has_communication_scheme:
            step_all["communication_scheme"].append(
                _parse_optional_string(
                    step_row["communication_scheme"], "communication_scheme"
                )
            )
        for column in optional_vector_columns:
            if column not in step_all:
                continue
            value = step_row[column]
            if column == "UAV_service_success":
                parsed_value = (
                    None if _is_missing_scalar(value)
                    else _parse_flag_vector(value, column, len(UAV_pos))
                )
            else:
                parsed_value = _parse_optional_nonnegative_vector(
                    value, column, len(UAV_pos)
                )
            step_all[column].append(parsed_value)

    return step_all


# 只取目前影格與軌跡視窗重疊的位置，避免每格複製整個回合歷史。
def _trajectory_window(initial_positions, step_positions, step, trajectory):
    start = max(0, step - trajectory)
    if start == 0:
        return [initial_positions] + step_positions[:step]
    return step_positions[start - 1:step]


class _EntityLabelLayout(Artist):
    """Keep each object's labels near its point with screen-space line spacing."""

    # 初始化貼近物件座標的文字標籤管理器。
    def __init__(self, ax):
        super().__init__()
        self.ax = ax
        self.groups = []
        self.set_zorder(2.8)

    # 建立可依畫面偏移調整的文字標籤。
    def _new_annotation(self, position, text, fontsize):
        annotation = Annotation(
            text,
            xy=position,
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            annotation_clip=False,
        )
        annotation.set_clip_on(True)
        annotation.set_zorder(3)
        self.ax.add_artist(annotation)
        return annotation

    # 為同一物件建立共用位置與行距的標籤群組。
    def add_group(self, position, lines, line_offsets=None):
        annotations = [self._new_annotation(position, text, fontsize)
                       for text, fontsize in lines]
        if line_offsets is not None:
            offsets = tuple(line_offsets)
        elif len(lines) == 2:
            # 雙行標籤需要兩個垂直位置，其餘使用單行偏移。
            offsets = (-8, -22)
        else:
            offsets = (-8,)
        group = {"position": position,
                 "annotations": annotations,
                 "line_offsets": offsets}
        self.groups.append(group)
        return group

    # 在指定物件標籤中追加一行文字。
    def add_line(self, group, text, fontsize):
        group["annotations"].append(
            self._new_annotation(group["position"], text, fontsize))
        if len(group["annotations"]) == 2:
            group["line_offsets"] = (-8, -22)

    # 依指定間距放大或縮小文字邊界框。
    @staticmethod
    def _expanded(box, pad):
        return Bbox.from_extents(box.x0 - pad, box.y0 - pad,
                                 box.x1 + pad, box.y1 + pad)

    # 計算同一物件全部標籤的合併邊界。
    def _group_box(self, group, renderer):
        for annotation in group["annotations"]:
            annotation.update_positions(renderer)
        boxes = [Text.get_window_extent(annotation, renderer)
                 for annotation in group["annotations"]]
        return Bbox.union(boxes)

    # 維持標籤行距並調整邊界，避免文字超出座標軸。
    def draw(self, renderer):
        axes_box = self.ax.bbox
        point_px = renderer.points_to_pixels(1)
        safe_axes_box = self._expanded(axes_box, -2)
        for group in self.groups:
            annotations = group["annotations"]
            line_offsets = group["line_offsets"]
            for annotation, line_y in zip(annotations, line_offsets):
                annotation.set_position((0, line_y))
            box = self._expanded(self._group_box(group, renderer), 2)
            # Only keep this object's text inside the axes. Other objects are
            # allowed to overlap; their positions never move this label group.
            shift_x = max(safe_axes_box.x0 - box.x0,
                          min(0.0, safe_axes_box.x1 - box.x1)) / point_px
            shift_y = max(safe_axes_box.y0 - box.y0,
                          min(0.0, safe_axes_box.y1 - box.y1)) / point_px
            for annotation, line_y in zip(annotations, line_offsets):
                annotation.set_position((shift_x, shift_y + line_y))

        self.stale = False

# 繪製地圖、風險區、UAV、裝置與 AoI 標籤。
def draw_common_scene(
    ax,
    UAV_pos,
    device_pos,
    risk_region,
    title,
    device_AOI,
    map_size=10.0,
    service_radius=None,
    service_rule=None,
):
    ax.clear()

    service_radius = _parse_service_radius(service_radius)
    service_rule = _parse_service_rule(service_rule)

    map_boundary = Rectangle(
        (0, 0),
        map_size,
        map_size,
        fill = False,
        alpha = 0.5,
        linewidth = 2,
        color = "black",
        zorder=2,
    )
    ax.add_patch(map_boundary)

    if service_radius is not None:
        if service_rule == "force_within_range_allow_early":
            service_label = "Forced service range"
        else:
            service_label = "Service range"
        for index, (x, y) in enumerate(device_pos):
            if index == 0:
                label = service_label
            else:
                label = "_nolegend_"
            service_circle = Circle(
                (float(x), float(y)),
                radius=service_radius,
                facecolor="#93c5fd",
                edgecolor="#60a5fa",
                linewidth=0.7,
                alpha=0.14,
                label=label,
                zorder=0.4,
            )
            service_circle.set_clip_path(map_boundary)
            ax.add_patch(service_circle)

    for risk_box in risk_region:
        x1, y1, x2, y2 = risk_box
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill = True,
            alpha = 0.15,
            linewidth = 2,
            label = "Risk Region",
            color = "red",
            zorder=0.8,
        )
        ax.add_patch(rect)

    UAV_x = [p[0] for p in UAV_pos]
    UAV_y = [p[1] for p in UAV_pos]
    ax.scatter(UAV_x, UAV_y, s=50, marker="s", label="UAV")
    label_layout = _EntityLabelLayout(ax)
    uav_label_groups = []
    for idx, (x, y) in enumerate(UAV_pos, start=1):
        position = (float(x), float(y))
        uav_label_groups.append(label_layout.add_group(
            position, [(f"UAV {idx}", 10)]))

    device_x = [p[0] for p in device_pos]
    device_y = [p[1] for p in device_pos]
    sizes = [25 + 175 * min(AOI / 100, 1) for AOI in device_AOI]
    ax.scatter(device_x, device_y, s=sizes, marker="o", alpha=1, label="Device")
    for idx, (x, y) in enumerate(device_pos, start=1):
        position = (float(x), float(y))
        label_layout.add_group(position, [
            (f"D {idx}", 10), (f"{int(device_AOI[idx - 1])}", 6)],
            line_offsets=(-16, 0))

    ax._trace_label_layout = label_layout
    ax._trace_uav_label_groups = uav_label_groups
    ax.add_artist(label_layout)

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

# 由前後 AoI、服務選擇與獎勵推算本步成本及服務降低量。
def step_costs(previous_aoi, current_aoi, selections, rewards, consecutive=True,
               *, current_source_aoi=None, service_success=None):
    """Current environment reward = -weighted transmission/risk cost - mean AoI.

    Use the saved source-packet age and service-success flags when available;
    legacy traces retain the age-one, requested-selection fallback. Attribute
    service reductions in UAV order, so duplicate service gets zero additional
    credit. Missing/intermediate frames cannot support exact attribution.
    """
    current = np.asarray(current_aoi, dtype=float)
    previous = np.asarray(previous_aoi, dtype=float)
    cost = float(current.mean())
    result = {"aoi_cost": cost,
              "energy_risk_cost": -np.asarray(rewards, dtype=float) - cost,
              "before_service_cost": None, "reductions": None,
              "total_reduction": None, "net_cost_reduction": None}
    if not consecutive or previous.shape != current.shape:
        return result
    working = previous + 1
    source = None
    if current_source_aoi is not None:
        source = np.asarray(current_source_aoi, dtype=float)
        if source.shape != current.shape or not np.isfinite(source).all():
            return result
    before = float(working.mean())
    reductions = []
    for uav_index, device in enumerate(selections):
        reduction = 0.
        did_succeed = (
            service_success is None
            or (uav_index < len(service_success) and service_success[uav_index])
        )
        if did_succeed and 0 <= device < len(working):
            delivered_age = 1. if source is None else source[device]
            reduction = float(working[device] - delivered_age)
            working[device] = delivered_age
        reductions.append(reduction)
    if not np.allclose(working, current):
        return result
    result.update(before_service_cost=before, reductions=reductions,
                  total_reduction=float(sum(reductions)),
                  net_cost_reduction=float(previous.mean() - cost))
    return result


class EpisodeStepViewer:
    # 整理指定回合資料，建立繪圖、成本面板與播放控制。
    def __init__(
        self,
        episode,
        episode_df,
        step_df,
        interval,
        display="all",
        trajectory=10,
        figure=None,
        controls=True,
        cost_panel=True,
        show_legend=True,
        aoi_view="uav",
    ):
        self.episode = episode
        self.episode_df = episode_df
        self.step_df = step_df
        self.interval = interval
        self.display = display
        self.trajectory = max(0, trajectory)
        self._standalone_figure = figure is None

        episode_data = self.episode_df[self.episode_df["episode"] == self.episode]
        if len(episode_data) == 0:
            raise ValueError(f"在 episode parquet 中找不到 episode = {self.episode}")
        self.episode_data = parse_episode_data(episode_data.iloc[0])

        self.step = 0
        self.playing = False
        self.closed = False
        self.controls = bool(controls)
        self.cost_panel = bool(cost_panel)
        self.show_legend = bool(show_legend)
        self.cost_text = ""

        step_data = get_episode_step_df(
            self.step_df, self.episode, self.episode_data["initial_step"]
        )
        self.step_data = parse_step_data(
            step_data, self.episode_data["initial_step"]
        )
        self.frame_count = len(self.step_data["step"])
        self.source_aoi_available = (
            self.episode_data["init_source_AOI"] is not None
            and all(value is not None for value in self.step_data["source_AOI"])
        )
        if aoi_view not in ("uav", "source"):
            raise ValueError("aoi_view must be 'uav' or 'source'")
        if aoi_view == "source" and not self.source_aoi_available:
            raise ValueError(
                f"Source AoI is unavailable for episode {self.episode}: "
                "the trace must contain init_source_AOI and source_AOI for every step."
            )
        self.aoi_view = aoi_view
        self.max_optional_reward_fields = 0
        self.optional_uav_fields = (
            "upper_reward", "intrinsic_reward", "UAV_transmission_power",
            "UAV_transmission_peak_power", "UAV_transmission_time",
        )
        for index in range(self.frame_count):
            optional_count = sum(
                column in self.step_data
                and self.step_data[column][index] is not None
                for column in self.optional_uav_fields
            )
            self.max_optional_reward_fields = max(
                self.max_optional_reward_fields, optional_count
            )
        self.cumulative_rewards = np.cumsum([
            np.mean(rewards) for rewards in self.step_data["reward"]
        ])
        available_fonts = {font.name for font in fontManager.ttflist}
        chinese_font = "sans-serif"
        for name in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "SimHei"):
            if name in available_fonts:
                chinese_font = name
                break
        self.sidebar_font = FontProperties(family=chinese_font)
        self.map_size = self.episode_data["map_size"]

        uav_count = len(self.episode_data["init_UAV_pos"])
        self.fig = figure
        if self.fig is None:
            self.fig, self.ax = plt.subplots(
                figsize=(12, max(7, 3 + 1.3 * uav_count))
            )
            if self.cost_panel:
                self.fig.subplots_adjust(left=0.06, bottom=0.18, right=0.62)
                self.ax_cost = self.fig.add_axes([0.66, 0.18, 0.33, 0.73])
            else:
                self.fig.subplots_adjust(left=0.04, bottom=0.08, right=0.98)
                self.ax_cost = None
        else:
            # A supplied Figure is owned by the embedding application. Do not
            # use pyplot here: this path must stay inside the caller's canvas.
            self.fig.clear()
            self.ax = self.fig.subplots()
            if self.cost_panel:
                self.ax.set_position([0.04, 0.08, 0.58, 0.84])
                self.ax_cost = self.fig.add_axes([0.65, 0.08, 0.33, 0.84])
            else:
                self.ax.set_position([0.12, 0.14, 0.85, 0.76])
                self.ax_cost = None

        self.btn_prev = self.btn_play = self.btn_next = self.btn_aoi = None
        self.timer = None
        if self.controls:
            self.ax_prev = self.fig.add_axes([0.20, 0.04, 0.15, 0.07])
            self.ax_play = self.fig.add_axes([0.42, 0.04, 0.15, 0.07])
            self.ax_next = self.fig.add_axes([0.64, 0.04, 0.15, 0.07])
            self.ax_aoi = self.fig.add_axes([0.81, 0.04, 0.17, 0.07])
            self.btn_prev = Button(self.ax_prev, "<")
            self.btn_play = Button(self.ax_play, "Play")
            self.btn_next = Button(self.ax_next, ">")
            self.btn_aoi = Button(self.ax_aoi, self._aoi_button_label())
            self.btn_prev.on_clicked(self.on_prev)
            self.btn_play.on_clicked(self.on_play_pause)
            self.btn_next.on_clicked(self.on_next)
            self.btn_aoi.on_clicked(self.on_toggle_aoi_view)
            if not self.source_aoi_available:
                self.btn_aoi.label.set_text("Source N/A")
                self.btn_aoi.set_active(False)
            self.timer = self.fig.canvas.new_timer(interval=self.interval)
            self.timer.add_callback(self.on_timer)
            self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.draw_init_frame()

    # 限制影格範圍並繪製初始或指定步驟畫面。
    def set_frame(self, frame):
        """Draw a zero-based frame index (zero is the episode's initial state)."""
        if self.closed:
            return
        self.step = max(0, min(self.frame_count, int(frame)))
        if self.step == 0:
            self.draw_init_frame()
        else:
            self.draw_frame()

    # 切換圖例並重繪目前影格，不改變步數或播放狀態。
    def set_legend_visible(self, visible):
        visible = bool(visible)
        if self.closed or visible == self.show_legend:
            return
        self.show_legend = visible
        if self.step == 0:
            self.draw_init_frame()
        else:
            self.draw_frame()

    # 切換地圖 AoI 來源並保留目前影格與播放狀態。
    def set_aoi_view(self, view):
        if view not in ("uav", "source"):
            raise ValueError("aoi_view must be 'uav' or 'source'")
        if view == "source" and not self.source_aoi_available:
            raise ValueError(
                f"Source AoI is unavailable for episode {self.episode}: "
                "the trace must contain init_source_AOI and source_AOI for every step."
            )
        if self.closed or view == self.aoi_view:
            return
        self.aoi_view = view
        if self.btn_aoi is not None:
            self.btn_aoi.label.set_text(self._aoi_button_label())
        if self.step == 0:
            self.draw_init_frame()
        else:
            self.draw_frame()

    # 取得目前 AoI 檢視標題與獨立按鈕文字。
    def _aoi_view_label(self):
        if self.aoi_view == "source":
            return "Source AoI (local packet age)"
        return "Collector AoI"

    def _aoi_button_label(self):
        if self.aoi_view == "source":
            return "AoI: Source"
        return "AoI: UAV"

    # 獨立播放視窗按鈕切換收集端／來源端 AoI。
    def on_toggle_aoi_view(self, _event):
        target = "source" if self.aoi_view == "uav" else "uav"
        self.set_aoi_view(target)

    # 依目前偏好在右上角建立或移除圖例。
    def _draw_legend(self):
        legend = self.ax.get_legend()
        if self.show_legend:
            self.ax.legend(loc="upper right")
        elif legend is not None:
            legend.remove()

    # 繪製回合初始位置、AoI 與成本資訊。
    def draw_init_frame(self):
        episode = self.episode_data["episode"]
        init_UAV_pos = self.episode_data["init_UAV_pos"]
        init_device_pos = self.episode_data["init_device_pos"]
        risk_region = self.episode_data["risk_region"]
        if self.aoi_view == "source":
            device_AOI = self.episode_data["init_source_AOI"]
        else:
            device_AOI = self.episode_data["init_device_AOI"]

        title = (f"Segment {episode} | Step {self.episode_data['initial_step']}"
                 f" | {self._aoi_view_label()}")

        draw_common_scene(
            ax = self.ax,
            UAV_pos = init_UAV_pos,
            device_pos = init_device_pos,
            risk_region = risk_region,
            title = title,
            device_AOI = device_AOI,
            map_size = self.map_size,
            service_radius = self.episode_data["service_radius"],
            service_rule = self.episode_data["service_rule"],
        )

        self.ax.set_xlim(-self.map_size * 0.1, self.map_size * 1.1)
        self.ax.set_ylim(-self.map_size * 0.1, self.map_size * 1.1)

        self._draw_legend()
        self.draw_cost_panel()
        self.fig.canvas.draw_idle()

    # 繪製目前步驟的狀態、軌跡、服務連線與獎勵。
    def draw_frame(self):
        episode = self.episode_data["episode"]
        risk_region = self.episode_data["risk_region"]

        assert self.step >= 1
        UAV_pos = self.step_data["UAV_pos"][self.step - 1]
        device_pos = self.step_data["device_pos"][self.step - 1]
        if self.aoi_view == "source":
            device_AOI = self.step_data["source_AOI"][self.step - 1]
        else:
            device_AOI = self.step_data["device_AOI"][self.step - 1]
        UAV_action_select = self.step_data["UAV_action_select"][self.step - 1]
        reward = self.step_data["reward"][self.step - 1]

        title = (f"Segment {episode} | Step {self.step_data['step'][self.step - 1]}"
                 f" | {self._aoi_view_label()}")

        draw_common_scene(
            ax = self.ax,
            UAV_pos = UAV_pos,
            device_pos = device_pos,
            risk_region = risk_region,
            title = title,
            device_AOI = device_AOI,
            map_size = self.map_size,
            service_radius = self.episode_data["service_radius"],
            service_rule = self.episode_data["service_rule"],
        )

        if self.display in ("all", "move-only") and self.trajectory:
            history = _trajectory_window(
                self.episode_data["init_UAV_pos"],
                self.step_data["UAV_pos"],
                self.step,
                self.trajectory,
            )
            self.draw_trajectory(history)

        service_success = None
        if "UAV_service_success" in self.step_data:
            service_success = self.step_data["UAV_service_success"][self.step - 1]
        if self.display in ("all", "action-only"):
            self.draw_service(
                UAV_pos, device_pos, UAV_action_select,
                service_success=service_success,
            )

        self.draw_reward(
            UAV_pos = UAV_pos,
            reward = reward
        )
        if "UAV_current_position" in self.step_data:
            for u, current_position in enumerate(
                    self.step_data["UAV_current_position"][self.step - 1]):
                if current_position:
                    self.ax._trace_label_layout.add_line(
                        self.ax._trace_uav_label_groups[u], "UAV 目前位置", 8)
        if "UAV_idle" in self.step_data:
            for u, idle in enumerate(self.step_data["UAV_idle"][self.step - 1]):
                if idle:
                    self.ax._trace_label_layout.add_line(
                        self.ax._trace_uav_label_groups[u], "Idle", 8)
        if service_success is not None:
            for u, (device, succeeded) in enumerate(zip(UAV_action_select, service_success)):
                if 0 <= device < len(device_pos) and not succeeded:
                    self.ax._trace_label_layout.add_line(
                        self.ax._trace_uav_label_groups[u], "Not delivered", 8)

        self.ax.set_xlim(-self.map_size * 0.1, self.map_size * 1.1)
        self.ax.set_ylim(-self.map_size * 0.1, self.map_size * 1.1)
        self._draw_legend()
        self.draw_cost_panel()
        self.fig.canvas.draw_idle()

    # 整理目前步驟的 AoI、傳輸與風險成本、服務降低量及獎勵資訊。
    def draw_cost_panel(self):
        if self.step == 0:
            aoi = np.mean(self.episode_data["init_device_AOI"])
            lines = ["本步資訊", "", f"平均收集端 AoI：{aoi:.3f}",
                     "傳輸與風險總成本：0.000", "平均環境 reward：0.000",
                     "服務降低的平均 AoI：0.000",
                     "片段累積環境回報：0.000"]
            for u in range(len(self.episode_data["init_UAV_pos"])):
                lines += ["", f"UAV {u + 1}｜尚未執行動作",
                          "  服務降低量（依序歸屬）：0.000",
                          "  個別傳輸與風險成本：0.000",
                          "  個別環境 reward：0.000"]
        else:
            index = self.step - 1
            current = self.step_data["device_AOI"][index]
            # 第一筆動作以前的狀態取回合初始資料，後續則取前一步記錄。
            if index == 0:
                previous = self.episode_data["init_device_AOI"]
                previous_step = self.episode_data["initial_step"]
            else:
                previous = self.step_data["device_AOI"][index - 1]
                previous_step = self.step_data["step"][index - 1]
            selections = self.step_data["UAV_action_select"][index]
            rewards = self.step_data["reward"][index]
            source_aoi = self.step_data["source_AOI"][index]
            service_success = None
            if "UAV_service_success" in self.step_data:
                service_success = self.step_data["UAV_service_success"][index]
            costs = step_costs(
                previous,
                current,
                selections,
                rewards,
                self.step_data["step"][index] == previous_step + 1,
                current_source_aoi=source_aoi,
                service_success=service_success,
            )
            if costs["reductions"] is None:
                # 降低量不可用時顯示提示，不計算平均值。
                reduction_text = "無法計算"
            else:
                reduction_text = f"{costs['total_reduction'] / len(current):.3f}"
            upper_rewards = self.step_data["upper_reward"][index]
            intrinsic_rewards = self.step_data["intrinsic_reward"][index]
            lines = ["本步資訊", "", f"平均收集端 AoI：{costs['aoi_cost']:.3f}",
                     f"傳輸與風險總成本：{np.sum(costs['energy_risk_cost']):.3f}",
                     f"平均環境 reward：{np.mean(rewards):.3f}",
                     f"服務降低的平均 AoI：{reduction_text}",
                     f"片段累積環境回報：{self.cumulative_rewards[index]:.3f}"]
            for u, device in enumerate(selections):
                if ("UAV_current_position" in self.step_data
                        and self.step_data["UAV_current_position"][index][u]):
                    service = "UAV 目前位置（一個環境步後重選）"
                elif ("UAV_idle" in self.step_data
                        and self.step_data["UAV_idle"][index][u]):
                    service = "Idle（停留一個環境步）"
                elif 0 <= device < len(current):
                    if (service_success is not None and u < len(service_success)
                            and not service_success[u]):
                        service = f"未完成：裝置 {device + 1}"
                    else:
                        service = f"服務裝置 {device + 1}"
                else:
                    service = "不服務"
                if costs["reductions"] is None:
                    reduction = "無法計算"
                else:
                    reduction = f"{costs['reductions'][u]:.3f}"
                lines += ["", f"UAV {u + 1}｜{service}",
                          f"  服務降低量（依序歸屬）：{reduction}",
                          f"  個別傳輸與風險成本：{costs['energy_risk_cost'][u]:.3f}",
                          f"  個別環境 reward：{rewards[u]:.3f}"]
                if upper_rewards is not None:
                    lines.append(f"  上層獎勵：{upper_rewards[u]:.3f}")
                if intrinsic_rewards is not None:
                    lines.append(f"  下層導航獎勵：{intrinsic_rewards[u]:.3f}")
                telemetry = []
                for column, label, unit in (
                    ("UAV_transmission_power", "平均", "W"),
                    ("UAV_transmission_peak_power", "峰值", "W"),
                    ("UAV_transmission_time", "時間", "s"),
                ):
                    values = self.step_data.get(column)
                    if values is None:
                        continue
                    row_values = values[index]
                    if row_values is None or row_values[u] is None:
                        continue
                    telemetry.append(f"{label} {row_values[u]:.3g} {unit}")
                if telemetry:
                    lines.append("  傳輸：" + " / ".join(telemetry))
        time_lines = [
            f"整場 Wall time（含存檔）：{_format_wall_time(self.episode_data['total_wall_time_seconds'])}",
            f"訓練 Wall time（不含存檔）：{_format_wall_time(self.episode_data['training_wall_time_seconds'])}",
            f"本回合 Wall time：{_format_wall_time(self.episode_data['episode_wall_time_seconds'])}",
            f"截至本回合 Wall time：{_format_wall_time(self.episode_data['wall_time_seconds'])}",
            "",
        ]
        lines = [lines[0], "", *time_lines, *lines[2:]]
        communication_model = self.episode_data.get("communication_model")
        scheme = None
        if self.step > 0 and "communication_scheme" in self.step_data:
            scheme = self.step_data["communication_scheme"][self.step - 1]
        communication_lines = []
        if communication_model:
            communication_lines.append(f"模型：{communication_model}")
        if scheme:
            communication_lines.append(f"配置：{scheme}")
        if communication_lines:
            lines[2:2] = communication_lines + [""]
        lines += ["", "成本與服務降低量依收集端 AoI（device_AOI）計算。",
                  "地圖切換來源 AoI 時，此成本面板仍採收集端 AoI。",
                  "成本為加權傳輸成本與實際風險懲罰，不代表焦耳能量。",
                  "服務降低量以服務前 AoI 為基準，包含自然增加的 1；",
                  "重複服務只計一次；HRL AoI 模式均分降低量。",
                  "AoI 扣加權事件；環境均值／SMDP 在 Option 結束結算。",
                  "SMDP 保存區間原始總回報，不是 rho 或 TD 誤差。",
                  "歷史上層獎勵依保存值呈現，不重算。",
                  "地圖數字為本步個別環境 reward。",
                  "片段累積值只加總已記錄步驟各 UAV 平均環境 reward；",
                  "不代表完整回合回報。"]
        self.cost_text = "\n".join(lines)
        if self._standalone_figure and self.cost_panel:
            uav_count = len(self.episode_data["init_UAV_pos"])
            current_optional_fields = 0
            if self.step > 0:
                current_optional_fields = sum(
                    column in self.step_data
                    and self.step_data[column][self.step - 1] is not None
                    for column in self.optional_uav_fields
                )
            missing_optional_fields = max(
                0, self.max_optional_reward_fields - current_optional_fields
            )
            required_lines = len(lines) + missing_optional_fields * uav_count
            # Reserve enough physical height for the longest optional-reward row
            # in this trace, plus a small bottom margin for the rendered text.
            line_height_inches = 10 * 1.4 / 72
            text_height_inches = required_lines * line_height_inches
            axes_height = self.ax_cost.get_position().height
            required_height = (text_height_inches + 0.4) / axes_height
            if required_height > self.fig.get_figheight():
                self.fig.set_size_inches(
                    self.fig.get_figwidth(), required_height, forward=True
                )
        if not self.cost_panel:
            return
        self.ax_cost.clear()
        self.ax_cost.set_axis_off()
        self.ax_cost.text(0, 1, "\n".join(lines), transform=self.ax_cost.transAxes,
                          va="top", fontsize=10, linespacing=1.4,
                          fontproperties=self.sidebar_font)

    # 繪製各 UAV 的歷史移動路徑並標示最新移動。
    def draw_trajectory(self, traj_UAV_pos):
        for u in range(len(traj_UAV_pos[0])):
            UAV_x = [p[u][0] for p in traj_UAV_pos]
            UAV_y = [p[u][1] for p in traj_UAV_pos]

            self.ax.plot(UAV_x[:-1], UAV_y[:-1], linewidth=2, alpha=0.35, linestyle='-')
            # 圖例只標記第一條軌跡，避免每個 UAV 重複顯示。
            if u == 0:
                label = "Latest Move"
            else:
                label = None
            self.ax.plot(UAV_x[-2:], UAV_y[-2:], linewidth=4, alpha=0.95,
                         color="red", linestyle='-', label=label)

    # 繪製 UAV 與其服務裝置之間的連線。
    def draw_service(self, UAV_pos, device_pos, UAV_action_select,
                     service_success=None):
        labeled = False
        for u in range(len(UAV_pos)):
            action_select = UAV_action_select[u]
            if (service_success is not None and u < len(service_success)
                    and not service_success[u]):
                continue
            if not 0 <= action_select < len(device_pos):
                continue
            
            service_x = [UAV_pos[u][0], device_pos[action_select][0]]
            service_y = [UAV_pos[u][1], device_pos[action_select][1]]

            # 多條服務連線共用一個圖例項目。
            if labeled:
                label = None
            else:
                label = "Service Link"
            self.ax.plot(service_x, service_y, linewidth=2, alpha=0.35,
                         linestyle='--', color="black", label=label)
            labeled = True
                
    # 在各 UAV 標籤中加入該步環境 reward。
    def draw_reward(self, UAV_pos, reward):
        for idx, (x, y) in enumerate(UAV_pos, start = 1):
            self.ax._trace_label_layout.add_line(
                self.ax._trace_uav_label_groups[idx - 1],
                f"{reward[idx - 1]:.1f}", 8)

    # 處理上一影格操作，先暫停再切換畫面。
    def on_prev(self, event):
        self.pause()
        self.set_frame(self.step - 1)

    # 處理下一影格操作，先暫停再切換畫面。
    def on_next(self, event):
        self.pause()
        self.set_frame(self.step + 1)

    # 切換獨立檢視器的播放狀態與計時器。
    def on_play_pause(self, event):
        if not self.controls or not self.frame_count:
            return
        self.playing = not self.playing

        if self.playing:
            if self.step == self.frame_count:
                self.step = 0
                self.draw_init_frame()
            self.timer.start()
        else:
            self.timer.stop()

        # 按鈕文字反映切換後的播放狀態。
        if self.playing:
            button_label = "Pause"
        else:
            button_label = "Play"
        self.btn_play.label.set_text(button_label)

    # 播放時計時推進影格，到最後一格則暫停。
    def on_timer(self):
        if not self.playing or self.closed:
            return

        if self.step < self.frame_count:
            self.set_frame(self.step + 1)
        if self.step == self.frame_count:
            self.pause()
            if self.btn_play:
                self.btn_play.label.set_text("Replay")

    # 停止播放與計時器，更新播放按鈕。
    def pause(self):
        self.playing = False
        if self.timer:
            self.timer.stop()
        if self.btn_play:
            self.btn_play.label.set_text("Play")

    # 視窗關閉時停止播放並標記檢視器已關閉。
    def on_close(self, event):
        self.pause()
        self.closed = True

    # 移除圖形或嵌入面板前，停止檢視器活動。
    def close(self):
        """Stop viewer activity before its Figure or embedding canvas is removed."""
        if self.closed:
            return
        self.pause()
        self.closed = True
        
# 解析參數並開啟軌跡檢視器；工具匯入時不載入訓練設定。
def main(argv=None):
    import utils
    args = utils.parse_args("trace_tool", argv)
    if args.snapshot:
        plt.switch_backend("Agg")
    if args.episode_parquet:
        episode_df = load_episode_df(args.episode_parquet)
        step_df = load_step_df(args.step_parquet)
    else:
        import random
        from Environment import Environment
        np.random.seed(args.seed)
        random.seed(args.seed)
        episode_df, step_df = record_trace(Environment(), steps=args.steps, episode=args.episode)
    if args.save_dir:
        saving.make_directory(args.save_dir, parents=True, exist_ok=True)
        saving.save_demo_trace(args.save_dir, episode_df, step_df)

    viewer = EpisodeStepViewer(
        episode = args.episode,
        episode_df = episode_df,
        step_df = step_df,
        interval = args.interval,
        display = args.display,
        trajectory = args.trajectory,
    )
    if args.snapshot:
        viewer.step = viewer.frame_count
        if viewer.step:
            viewer.draw_frame()
        saving.make_directory(args.snapshot.parent, parents=True, exist_ok=True)
        saving.save_figure(viewer.fig, args.snapshot, dpi=150)
        plt.close(viewer.fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
