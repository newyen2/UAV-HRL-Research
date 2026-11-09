"""Shared runtime configuration parsing and validation."""
from __future__ import annotations

import ast
import copy
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from runtime import fileIO as saving


METHODS = {
    "DQN": {"module": "DQN.py", "settings": ("DQN_SETTING",)},
    "DQN_service": {"module": "DQN_service.py", "settings": ("DQN_SERVICE_SETTING",)},
    "HRL": {"module": "HRL.py", "settings": ("HRL_UPPER_SETTING", "HRL_LOWER_SETTING", "HRL_SETTING")},
    "HRL_PPO": {"module": "HRL_PPO.py", "settings": ("HRL_PPO_UPPER_SETTING", "HRL_PPO_LOWER_SETTING", "HRL_PPO_SETTING")},
    "MA-CQR": {"module": "MA-CQR.py", "settings": ("MA_CQR_SETTING",)},
}
RETIRED_METHODS = frozenset({
    "DQN_greedy", "HRL_original", "HRL_risk", "HRL_uav_risk", "HRL_RDSAC",
})

_COMMON_SETTINGS = {
    "ENV": "ENV_SETTING",
    "TRAIN": "TRAIN_SETTING",
    "EVAL": "EVAL_SETTING",
    "FUNCTIONAL": "FUNCTIONAL_SETTING",
    "FIXED": "FIXED_SETTING",
    "RECORD": "RECORD_SETTING",
}
_METHOD_SETTING_GROUPS = {
    "DQN_SETTING": "DQN",
    "DQN_SERVICE_SETTING": "DQN_SERVICE",
    "HRL_UPPER_SETTING": "HRL_UPPER",
    "HRL_LOWER_SETTING": "HRL_LOWER",
    "HRL_SETTING": "HRL",
    "HRL_PPO_UPPER_SETTING": "HRL_PPO_UPPER",
    "HRL_PPO_LOWER_SETTING": "HRL_PPO_LOWER",
    "HRL_PPO_SETTING": "HRL_PPO",
    "MA_CQR_SETTING": "MA_CQR",
}
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "utils.py"
_SETTING_OPTION_CONSTANTS = {
    ("ENV", "arrival_model"): "ENV_ARRIVAL_MODEL_OPTIONS",
    ("ENV", "communication_model"): "ENV_COMMUNICATION_MODEL_OPTIONS",
    ("TRAIN", "device"): "TRAIN_DEVICE_OPTIONS",
    ("HRL_LOWER", "algorithm"): "HRL_LOWER_ALGORITHM_OPTIONS",
    ("HRL_LOWER", "warmup_mode"): "HRL_LOWER_WARMUP_MODE_OPTIONS",
    ("HRL_UPPER", "reward_mode"): "HRL_UPPER_REWARD_MODE_OPTIONS",
    ("HRL_PPO_UPPER", "reward_mode"): "HRL_PPO_UPPER_REWARD_MODE_OPTIONS",
    ("HRL_PPO_LOWER", "algorithm"): "HRL_PPO_LOWER_ALGORITHM_OPTIONS",
    ("HRL_PPO_LOWER", "order"): "HRL_PPO_LOWER_ORDER_OPTIONS",
}


# 解析設定中允許的常數、陣列與 NumPy 座標表示。
def _literal(node: ast.AST):
    """Read the small safe expression subset used by utils settings."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        return value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Call):
        # NumPy arrays are represented as ordinary nested lists in the GUI.
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "array"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
                and node.args):
            return _literal(node.args[0])
    raise ValueError(f"Unsupported setting expression: {ast.dump(node, include_attributes=False)}")


# 只快取相同原碼的語法樹；設定值仍由每次呼叫重新建立，避免共用可變資料。
@lru_cache(maxsize=1)
def _parse_settings_source(source: str, filename: str):
    return ast.parse(source, filename=filename)


# 從 utils 原碼讀取選項，避免排程器載入 Torch 或訓練模組。
def setting_options(group: str, field: str) -> list[str]:
    """Read literal setting choices from utils without importing training code."""
    constant_name = _SETTING_OPTION_CONSTANTS.get((group, field))
    if constant_name is None:
        return []
    tree = _parse_settings_source(
        saving.read_text(_CONFIG_PATH, encoding="utf-8"), str(_CONFIG_PATH))
    for statement in tree.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == constant_name):
            return _literal(statement.value)
    raise RuntimeError(f"Could not read {constant_name} from utils.py")


# 保留既有呼叫介面，供 GUI 與其他程式讀取裝置選項。
def training_device_options() -> list[str]:
    return setting_options("TRAIN", "device")


# 從 utils 原碼讀出共用與方法設定的預設值。
def _read_default_groups() -> dict[str, dict]:
    tree = _parse_settings_source(
        saving.read_text(_CONFIG_PATH, encoding="utf-8"), str(_CONFIG_PATH))
    wanted = set(_COMMON_SETTINGS.values()) | set(_METHOD_SETTING_GROUPS)
    namespaces: dict[str, dict] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "SimpleNamespace"):
            continue
        fields = {}
        for keyword in value.keywords:
            if keyword.arg == "PATH":
                continue
            if keyword.arg is not None:
                fields[keyword.arg] = _literal(keyword.value)
        namespaces[target.id] = fields

    missing = wanted - namespaces.keys()
    if missing:
        raise RuntimeError(f"Could not read default settings from utils.py: {sorted(missing)}")
    return namespaces


# 回傳指定方法的完整預設設定副本。
def default_settings(method: str) -> dict:
    """Return fresh JSON-safe defaults for common and method-specific settings."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    namespaces = _read_default_groups()
    result = {group: namespaces[namespace]
              for group, namespace in _COMMON_SETTINGS.items()}
    for namespace in METHODS[method]["settings"]:
        group = _METHOD_SETTING_GROUPS[namespace]
        result[group] = namespaces[namespace]
    return result


# 列出目前配置不會使用的欄位；其他方法仍需要的設定保留在預設 schema。
def inactive_setting_paths(method: str, settings: Mapping) -> dict:
    inactive = {}
    upper_group = "HRL_UPPER"
    method_settings = METHODS[method]["settings"]
    lower_group = (
        _METHOD_SETTING_GROUPS[method_settings[1]]
        if len(method_settings) > 2 else "HRL_LOWER"
    )
    functional = settings["FUNCTIONAL"]
    if not functional["risk"]:
        for path in (("ENV", "penalty"), ("ENV", "prob_of_risk"),
                     ("FIXED", "fixed_risk_region")):
            inactive[path] = "未啟用風險區"
        if method == "HRL":
            inactive[("HRL", "risk_weight")] = "未啟用風險區"
            inactive[("HRL", "risk_penalty_reference")] = "未啟用風險區"
            inactive[(upper_group, "risk_penalty_weight")] = "未啟用風險區"
        elif method == "HRL_PPO":
            inactive[("HRL_PPO", "risk_weight")] = "未啟用風險區"
            inactive[("HRL_PPO", "risk_penalty_reference")] = "未啟用風險區"
            inactive[("HRL_PPO_UPPER", "risk_penalty_weight")] = "未啟用風險區"
    if functional["random_device_coord"]:
        inactive[("FIXED", "fixed_device_coord")] = "使用隨機裝置座標"
    else:
        inactive[("ENV", "min_device_distance")] = "使用固定裝置座標"
    if functional["random_UAV_coord"]:
        inactive[("FIXED", "fixed_UAV_coord")] = "使用隨機 UAV 座標"
    if not functional["random_device_coord"] and not functional["random_UAV_coord"]:
        inactive[("ENV", "max_coord_attempts")] = "兩種座標均固定，不進行座標採樣"

    arrival_model = settings["ENV"]["arrival_model"]
    if arrival_model == "always":
        inactive[("ENV", "arrival_probability")] = "固定週期模式不使用到達機率"
        inactive[("ENV", "arrival_rate")] = "固定週期模式不使用到達率"
    elif arrival_model == "bernoulli":
        inactive[("ENV", "arrival_interval")] = "Bernoulli 模式不使用固定到達週期"
        inactive[("ENV", "arrival_rate")] = "bernoulli 模式不使用到達率"
    elif arrival_model == "poisson":
        inactive[("ENV", "arrival_interval")] = "Poisson 模式不使用固定到達週期"
        inactive[("ENV", "arrival_probability")] = "poisson 模式不使用到達機率"

    communication_model = settings["ENV"]["communication_model"]
    if communication_model == "snr":
        inactive[("ENV", "max_transmit_power")] = "SNR 模式沿用原傳輸功率公式"
    if communication_model != "sinr_adaptive":
        inactive[("ENV", "fixed_transmit_power")] = "僅固定功率自適應排程使用"
    if communication_model != "sinr_subchannels":
        inactive[("ENV", "subchannel_count")] = "僅子頻道配置模式使用"

    if method == "HRL":
        reward_mode = settings[upper_group]["reward_mode"]
        if reward_mode in {"option_mean_environment", "average_reward_smdp"}:
            inactive[(upper_group, "risk_penalty_weight")] = "此回報模式不使用風險事件係數"
            inactive[("HRL", "fragment_steps")] = "此回報模式使用完整 Option"
        if reward_mode == "average_reward_smdp":
            inactive[(upper_group, "gamma")] = "平均回報 SMDP 不使用折扣率"
        else:
            inactive[(upper_group, "average_reward_lr")] = "此回報模式不更新平均回報率 rho"
        algorithm = settings[lower_group]["algorithm"]
        if algorithm == "a2c":
            for field in ("tau", "alpha", "std_min", "td_bound", "std_rate", "policy_delay"):
                inactive[(lower_group, field)] = "A2C 不使用 target、熵正則或分布 critic"
            inactive[("HRL", "lower_batch_size")] = "A2C 使用新資料，不使用下層 replay"
            inactive[(lower_group, "warmup_steps")] = "A2C 不使用下層 replay 預填"
            inactive[(lower_group, "warmup_mode")] = "A2C 不使用下層 replay 預填"
        elif algorithm == "sac":
            inactive[(lower_group, "a2c_steps")] = "此設定只用於 A2C rollout"
            for field in ("std_min", "td_bound", "std_rate", "policy_delay"):
                inactive[(lower_group, field)] = "SAC 不使用分布 critic 或延遲策略更新"
        elif algorithm == "dsac":
            inactive[(lower_group, "a2c_steps")] = "此設定只用於 A2C rollout"
            inactive[(lower_group, "std_rate")] = "標準差平均只用於 DSACv2"
        elif algorithm == "dsac_t":
            inactive[(lower_group, "a2c_steps")] = "此設定只用於 A2C rollout"
            inactive[(lower_group, "td_bound")] = "DSACv2 由標準差平均決定 TD 界限"

        if settings["HRL"]["pretrain_batches"] == 0:
            for path in ((lower_group, "pretrain_batch_size"), ("HRL", "navigation_cases")):
                inactive[path] = "示範預訓練批數為 0"
    elif method == "HRL_PPO":
        upper_group = "HRL_PPO_UPPER"
        lower_group = "HRL_PPO_LOWER"
        reward_mode = settings[upper_group]["reward_mode"]
        if reward_mode in {"option_mean_environment", "average_reward_smdp"}:
            inactive[(upper_group, "risk_penalty_weight")] = "此回報模式不使用風險事件係數"
            inactive[("HRL_PPO", "fragment_steps")] = "此回報模式使用完整 Option"
        if reward_mode == "average_reward_smdp":
            inactive[(upper_group, "gamma")] = "平均回報 SMDP 不使用折扣率"
        else:
            inactive[(upper_group, "average_reward_lr")] = "此回報模式不更新平均回報率 rho"
        if settings[lower_group]["algorithm"] == "ppo":
            inactive[(lower_group, "order")] = "PPO 不使用 A2PO agent 更新順序"
            inactive[(lower_group, "trace_clip")] = "PPO 不使用 A2PO 軌跡裁剪"
        if settings[lower_group]["pretrain_batches"] == 0:
            for field in ("pretrain_batch_size", "navigation_cases"):
                inactive[(lower_group, field)] = "示範預訓練批數為 0"
    return inactive


# 依預設值的型別檢查並轉換輸入設定。
def _normalise_value(value, prototype, path: str):
    if isinstance(prototype, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        return value
    if isinstance(prototype, int) and not isinstance(prototype, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
        return value
    if isinstance(prototype, float):
        return _normalise_finite_number(value, path)
    if isinstance(prototype, list):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be an array")
        if prototype and not isinstance(prototype[0], list) and len(value) != len(prototype):
            raise ValueError(f"{path} must contain {len(prototype)} items")
        if not prototype and value:
            raise ValueError(f"{path} must be empty")
        if not prototype:
            return []
        item_prototype = prototype[0]
        return [_normalise_value(item, item_prototype, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(prototype, str):
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return value
    if value != prototype:
        raise ValueError(f"{path} has an unsupported value")
    return value


# 檢查整數設定是否達到指定下限。
def _require_int_range(groups: dict, group: str, field: str, minimum: int = 1):
    value = groups[group][field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{group}.{field} must be an integer")
    if value < minimum:
        if minimum == 1:
            message = f"{group}.{field} must be a positive integer"
        elif minimum == 0:
            message = f"{group}.{field} must be a non-negative integer"
        else:
            message = f"{group}.{field} must be an integer of at least {minimum}"
        raise ValueError(message)


# 檢查數值設定是否位於指定範圍。
def _require_number_range(groups: dict, group: str, field: str, minimum=None,
                          maximum=None, minimum_open=False):
    value = groups[group][field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{group}.{field} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{group}.{field} must be a finite number")
    if minimum is not None:
        # minimum_open 為真時，下界不包含等號。
        if minimum_open:
            below_minimum = value <= minimum
            left = "greater than"
        else:
            below_minimum = value < minimum
            left = "at least"
        if below_minimum:
            raise ValueError(f"{group}.{field} must be {left} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{group}.{field} must be at most {maximum}")


# 驗證 HRL_PPO 專屬設定，避免與既有 HRL namespace 共用規則。
def _check_hrl_ppo_ranges(groups: dict):
    upper_group = "HRL_PPO_UPPER"
    lower_group = "HRL_PPO_LOWER"
    method_group = "HRL_PPO"

    for group, fields in (
            (upper_group, ("hidden_size", "eps_decisions")),
            (lower_group, ("hidden_size", "epochs", "minibatch_size", "rollout_steps",
                           "pretrain_batch_size", "navigation_cases")),
            (method_group, ("cpu_threads",))):
        for field in fields:
            _require_int_range(groups, group, field)
    if groups[upper_group]["reward_mode"] not in setting_options(upper_group, "reward_mode"):
        raise ValueError(
            f"{upper_group}.reward_mode must be one of "
            f"{setting_options(upper_group, 'reward_mode')}")
    if groups[upper_group]["reward_mode"] == "aoi_risk_count":
        _require_int_range(groups, method_group, "fragment_steps")
        if groups["FUNCTIONAL"]["risk"]:
            _require_number_range(groups, upper_group, "risk_penalty_weight", 0)
    _require_int_range(groups, upper_group, "warmup_steps", 0)
    _require_int_range(groups, lower_group, "pretrain_batches", 0)

    if groups[lower_group]["algorithm"] not in setting_options(lower_group, "algorithm"):
        raise ValueError(
            f"{lower_group}.algorithm must be one of {setting_options(lower_group, 'algorithm')}")
    if groups[lower_group]["order"] not in setting_options(lower_group, "order"):
        raise ValueError(
            f"{lower_group}.order must be one of {setting_options(lower_group, 'order')}")

    for group, field in ((upper_group, "tau"),
                         (upper_group, "start_eps"), (upper_group, "min_eps"),
                         (lower_group, "gamma"), (lower_group, "gae_lambda")):
        _require_number_range(groups, group, field, 0, 1)
    if groups[upper_group]["reward_mode"] != "average_reward_smdp":
        _require_number_range(groups, upper_group, "gamma", 0, 1)
    else:
        _require_number_range(
            groups, upper_group, "average_reward_lr", 0, minimum_open=True)
    if groups[upper_group]["min_eps"] > groups[upper_group]["start_eps"]:
        raise ValueError(f"{upper_group}.min_eps cannot exceed {upper_group}.start_eps")

    for group, field in ((upper_group, "lr"), (lower_group, "actor_lr"),
                         (lower_group, "critic_lr")):
        _require_number_range(groups, group, field, 0, minimum_open=True)
    _require_number_range(groups, lower_group, "clip_ratio", 0, 1,
                          minimum_open=True)
    _require_number_range(groups, lower_group, "entropy_coef", 0)
    _require_number_range(groups, lower_group, "target_kl", 0)
    _require_number_range(groups, lower_group, "trace_clip", 0, 1,
                          minimum_open=True)
    _require_number_range(groups, method_group, "service_ratio", 0, 1,
                          minimum_open=True)
    navigation_fraction = groups[method_group]["navigation_fraction"]
    _require_number_range(groups, method_group, "navigation_fraction", 0, 1)
    if navigation_fraction >= 1:
        raise ValueError(f"{method_group}.navigation_fraction must be less than 1")
    if (groups["TRAIN"]["episodes"]
            - int(groups["TRAIN"]["episodes"] * navigation_fraction) < 1):
        raise ValueError("HRL_PPO.navigation_fraction must leave at least one joint episode")

    for field in ("progress_weight", "success_reward", "step_penalty", "risk_weight",
                  "environment_reward_weight"):
        _require_number_range(groups, method_group, field, 0)
    _require_number_range(groups, method_group, "risk_penalty_reference", 0,
                          minimum_open=True)
    _require_number_range(groups, upper_group, "warmup_steps", 0)
    if groups["TRAIN"]["buffer_size"] < groups[upper_group]["warmup_steps"]:
        raise ValueError(f"{upper_group}.warmup_steps cannot exceed TRAIN.buffer_size")

    lower = groups[lower_group]
    _require_number_range(groups, lower_group, "log_std_min")
    _require_number_range(groups, lower_group, "log_std_max")
    if lower["log_std_min"] > lower["log_std_max"]:
        raise ValueError(f"{lower_group}.log_std_min cannot exceed {lower_group}.log_std_max")


# 檢查共用與方法參數的合法範圍及相互限制。
def _check_ranges(groups: dict, method: str):
    upper_group = "HRL_UPPER"
    method_settings = METHODS[method]["settings"]
    lower_group = (
        _METHOD_SETTING_GROUPS[method_settings[1]]
        if len(method_settings) > 2 else "HRL_LOWER"
    )
    for field in ("random_device_coord", "random_UAV_coord", "risk"):
        if not isinstance(groups["FUNCTIONAL"][field], bool):
            raise ValueError(f"FUNCTIONAL.{field} must be a boolean")
    if groups["TRAIN"]["device"] not in setting_options("TRAIN", "device"):
        raise ValueError(f"TRAIN.device must be one of {setting_options('TRAIN', 'device')}")
    if groups["ENV"]["arrival_model"] not in setting_options("ENV", "arrival_model"):
        raise ValueError(
            f"ENV.arrival_model must be one of {setting_options('ENV', 'arrival_model')}")
    communication_options = setting_options("ENV", "communication_model")
    if groups["ENV"]["communication_model"] not in communication_options:
        raise ValueError(f"ENV.communication_model must be one of {communication_options}")
    for group, field in (("ENV", "M"), ("ENV", "U"), ("ENV", "max_coord_attempts"),
                         ("ENV", "arrival_interval"), ("ENV", "subchannel_count"),
                         ("TRAIN", "max_steps"), ("TRAIN", "episodes"),
                         ("TRAIN", "buffer_size"), ("TRAIN", "batch_size"),
                         ("EVAL", "max_steps"), ("EVAL", "eval_runs"),
                         ("EVAL", "eval_periods"), ("RECORD", "reward_smoothing_window")):
        _require_int_range(groups, group, field)

    for field in ("map_size", "max_mov", "channel_gain", "altitude", "cell_distance",
                  "bandwidth", "packet_size", "noise_power", "max_transmit_power",
                  "fixed_transmit_power"):
        _require_number_range(groups, "ENV", field, 0, minimum_open=True)
    if (groups["ENV"]["communication_model"] == "sinr_adaptive"
            and groups["ENV"]["fixed_transmit_power"] > groups["ENV"]["max_transmit_power"]):
        raise ValueError("ENV.fixed_transmit_power must not exceed ENV.max_transmit_power")
    for field in ("energy_weight", "penalty", "min_device_distance"):
        _require_number_range(groups, "ENV", field, 0)
    _require_number_range(groups, "ENV", "prob_of_risk", 0, 1)
    _require_number_range(groups, "ENV", "arrival_probability", 0, 1)
    _require_number_range(groups, "ENV", "arrival_rate", 0)

    if groups["TRAIN"]["buffer_size"] < groups["TRAIN"]["batch_size"]:
        raise ValueError("TRAIN.buffer_size must be at least TRAIN.batch_size")
    for group in ("TRAIN", "EVAL"):
        seed = groups[group]["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{group}.seed must be an integer")
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError(f"{group}.seed must be between 0 and 4294967295")
    if groups["ENV"]["min_device_distance"] > groups["ENV"]["map_size"] * math.sqrt(2):
        raise ValueError("ENV.min_device_distance cannot exceed the map diagonal")

    # Greedy/service collectors wait until each replay can hold the full warmup.
    method_group = _METHOD_SETTING_GROUPS[METHODS[method]["settings"][0]]
    if "warmup_steps" in groups[method_group]:
        _require_int_range(groups, method_group, "warmup_steps", 0)
        if method == "DQN_service" and groups[method_group]["warmup_steps"] > groups["TRAIN"]["buffer_size"]:
            raise ValueError(f"{method_group}.warmup_steps cannot exceed TRAIN.buffer_size")

    positive_int_fields = {
        "DQN": ("hidden_size", "eps_frames"),
        "DQN_SERVICE": ("hidden_size", "eps_frames"),
        "MA_CQR": ("hidden_size", "N", "eps_frames", "cpu_threads"),
        "HRL_UPPER": ("hidden_size",),
        "HRL_LOWER": ("hidden_size", "a2c_steps", "pretrain_batch_size", "policy_delay"),
        "HRL": ("cpu_threads", "lower_batch_size", "fragment_steps", "eps_decisions", "reach_cases",
                "reach_steps", "navigation_cases", "convergence_window", "convergence_patience",
                "convergence_min_episodes"),
    }
    nonnegative_int_fields = {
        "HRL_UPPER": ("warmup_steps",),
        "HRL_LOWER": ("warmup_steps", "imitation_decay_episodes"),
        "HRL": ("pretrain_batches", "navigation_eval_seed"),
    }
    for group, fields in positive_int_fields.items():
        if group in groups:
            for field in fields:
                if (group, field) == ("HRL", "fragment_steps") and (
                        groups[upper_group]["reward_mode"] in {
                            "option_mean_environment", "average_reward_smdp"}):
                    continue
                _require_int_range(groups, group, field)
    for group, fields in nonnegative_int_fields.items():
        if group in groups:
            for field in fields:
                _require_int_range(groups, group, field, 0)

    lower_is_off_policy = groups.get(lower_group, {}).get("algorithm") != "a2c"
    if ("HRL" in groups and lower_is_off_policy
            and groups["TRAIN"]["buffer_size"] < groups["HRL"]["lower_batch_size"]):
        raise ValueError("TRAIN.buffer_size must be at least HRL.lower_batch_size")
    if "HRL" in groups:
        if groups["TRAIN"]["buffer_size"] < groups[upper_group]["warmup_steps"]:
            raise ValueError(f"{upper_group}.warmup_steps cannot exceed TRAIN.buffer_size")
        if (lower_is_off_policy
                and groups["TRAIN"]["buffer_size"] < groups[lower_group]["warmup_steps"]):
            raise ValueError(f"{lower_group}.warmup_steps cannot exceed TRAIN.buffer_size")

    for group in ("DQN", "DQN_SERVICE", "HRL_UPPER", "HRL_LOWER", "HRL", "MA_CQR"):
        if group not in groups:
            continue
        for field in ("gamma", "tau", "start_eps", "min_eps"):
            if field in groups[group]:
                if group == upper_group and field == "gamma" and (
                        groups[upper_group]["reward_mode"] == "average_reward_smdp"):
                    continue
                _require_number_range(groups, group, field, 0, 1)
        for field in ("lr", "actor_lr", "critic_lr", "huber_kappa"):
            if field in groups[group]:
                _require_number_range(groups, group, field, 0, minimum_open=True)
        for field in ("alpha", "imitation_weight", "min_imitation_weight"):
            if field in groups[group]:
                _require_number_range(groups, group, field, 0)
        if "start_eps" in groups[group] and groups[group]["min_eps"] > groups[group]["start_eps"]:
            raise ValueError(f"{group}.min_eps cannot exceed {group}.start_eps")
    if "MA_CQR" in groups:
        _require_number_range(groups, "MA_CQR", "eta", 0, 1, minimum_open=True)
    if "DQN_SERVICE" in groups:
        _require_number_range(groups, "DQN_SERVICE", "service_ratio", 0, 1, minimum_open=True)
    if "HRL" in groups:
        _require_int_range(groups, "HRL", "convergence_window", 2)
        if groups[upper_group]["reward_mode"] not in setting_options(upper_group, "reward_mode"):
            raise ValueError(
                f"{upper_group}.reward_mode must be one of "
                f"{setting_options(upper_group, 'reward_mode')}")
        if groups[upper_group]["reward_mode"] == "average_reward_smdp":
            _require_number_range(
                groups, upper_group, "average_reward_lr", 0, minimum_open=True)
        if lower_group == "HRL_LOWER":
            if groups[lower_group]["algorithm"] not in setting_options(lower_group, "algorithm"):
                raise ValueError(
                    f"{lower_group}.algorithm must be one of "
                    f"{setting_options(lower_group, 'algorithm')}")
        if groups[lower_group]["warmup_mode"] not in setting_options(lower_group, "warmup_mode"):
            raise ValueError(
                f"{lower_group}.warmup_mode must be one of "
                f"{setting_options(lower_group, 'warmup_mode')}")
        for field in ("service_ratio", "upper_fraction", "lower_fraction", "reach_threshold"):
            _require_number_range(groups, "HRL", field, 0, 1, minimum_open=(field == "service_ratio"))
        _require_number_range(groups, "HRL", "convergence_tolerance", 0, 1)
        if groups["HRL"]["upper_fraction"] + groups["HRL"]["lower_fraction"] > 1:
            raise ValueError("HRL.upper_fraction + HRL.lower_fraction must be at most 1")
        for field in ("progress_weight", "success_reward", "step_penalty", "risk_weight"):
            _require_number_range(groups, "HRL", field, 0)
        _require_number_range(
            groups, "HRL", "risk_penalty_reference", 0, minimum_open=True)
        if (method == "HRL"
                and groups[lower_group]["imitation_decay_episodes"] > 0
                and groups[lower_group]["min_imitation_weight"]
                > groups[lower_group]["imitation_weight"]):
            raise ValueError(
                f"{lower_group}.min_imitation_weight cannot exceed "
                f"{lower_group}.imitation_weight "
                "when imitation_decay_episodes is positive")
    if "HRL_LOWER" in groups:
        _require_number_range(groups, "HRL_LOWER", "std_min", 0, minimum_open=True)
        _require_number_range(groups, "HRL_LOWER", "td_bound", 0, minimum_open=True)
        _require_number_range(groups, "HRL_LOWER", "std_rate", 0, 1, minimum_open=True)
        _require_number_range(groups, "HRL_LOWER", "log_std_min")
        _require_number_range(groups, "HRL_LOWER", "log_std_max")
        if groups["HRL_LOWER"]["log_std_min"] > groups["HRL_LOWER"]["log_std_max"]:
            raise ValueError("HRL_LOWER.log_std_min cannot exceed HRL_LOWER.log_std_max")
    if method == "HRL_PPO":
        _check_hrl_ppo_ranges(groups)
    _validate_fixed(groups)
    figure_size = groups["RECORD"]["figure_size"]
    if (not isinstance(figure_size, list) or len(figure_size) != 2
            or any(not isinstance(value, (int, float)) or value <= 0 for value in figure_size)):
        raise ValueError("RECORD.figure_size must contain two positive numbers")


# 檢查固定座標結構、數量及啟用時的地圖範圍。
def _validate_fixed(groups: dict):
    map_size = groups["ENV"]["map_size"]
    fixed = groups["FIXED"]
    for field, expected_rows, randomized in (
        ("fixed_device_coord", groups["ENV"]["M"], groups["FUNCTIONAL"]["random_device_coord"]),
        ("fixed_UAV_coord", groups["ENV"]["U"], groups["FUNCTIONAL"]["random_UAV_coord"]),
    ):
        rows = fixed[field]
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"FIXED.{field}[{index}] must contain x and y")
            if not randomized and any(value < 0 or value > map_size for value in row):
                raise ValueError(f"FIXED.{field}[{index}] must be inside the map")
        if not randomized and len(rows) != expected_rows:
            raise ValueError(f"FIXED.{field} must have {expected_rows} rows when random placement is disabled")

    risk_enabled = groups["FUNCTIONAL"]["risk"]
    for index, region in enumerate(fixed["fixed_risk_region"]):
        if not isinstance(region, list) or len(region) != 4:
            raise ValueError(f"FIXED.fixed_risk_region[{index}] must contain xmin, ymin, xmax, ymax")
        xmin, ymin, xmax, ymax = region
        if risk_enabled and any(value < 0 or value > map_size for value in region):
            raise ValueError(f"FIXED.fixed_risk_region[{index}] must be inside the map")
        if risk_enabled and (xmin > xmax or ymin > ymax):
            raise ValueError(f"FIXED.fixed_risk_region[{index}] has reversed bounds")


# 將設定陣列轉成只含有限數值的列表。
def _normalise_numeric_array(value, path: str):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, (list, tuple)):
            result.append([_normalise_finite_number(number, f"{path}[{index}][{column}]")
                           for column, number in enumerate(item)])
        else:
            result.append(_normalise_finite_number(item, f"{path}[{index}]"))
    return result


# 將合法數值轉為 float，拒絕非有限值。
def _normalise_finite_number(value, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


# 從新預設複製設定並正規化明確提供的欄位，供實驗與局部範本共用。
def _normalise_settings_overrides(defaults: Mapping, supplied: Mapping) -> dict:
    if not isinstance(supplied, Mapping):
        raise ValueError("settings must be an object")
    normalized = copy.deepcopy(defaults)
    for group, fields in supplied.items():
        if group not in normalized:
            if group == "RECORD" and isinstance(fields, Mapping) and "PATH" in fields:
                raise ValueError("RECORD.PATH is restricted and assigned by the worker")
            raise ValueError(f"Unknown or restricted settings group: {group!r}")
        if not isinstance(fields, Mapping):
            raise ValueError(f"settings.{group} must be an object")
        for field, value in fields.items():
            if field == "PATH" and group == "RECORD":
                raise ValueError("RECORD.PATH is restricted and assigned by the worker")
            if field not in normalized[group]:
                raise ValueError(f"Unknown or restricted setting: {group}.{field}")
            if group == "FIXED" and field.startswith("fixed_"):
                normalized[group][field] = _normalise_numeric_array(value, f"{group}.{field}")
            elif group == "RECORD" and field == "figure_size":
                normalized[group][field] = _normalise_numeric_array(value, f"{group}.{field}")
            elif group == "ENV" and field in {"energy_weight", "penalty"}:
                normalized[group][field] = _normalise_finite_number(value, f"{group}.{field}")
            else:
                normalized[group][field] = _normalise_value(
                    value, normalized[group][field], f"{group}.{field}")
    return normalized


# 驗證指定方法的完整或局部設定，補齊預設值並回傳獨立正規化副本。
def validate_settings(method: str, settings: Mapping) -> dict:
    """Validate settings for one method without resetting inactive values."""
    normalized = _normalise_settings_overrides(default_settings(method), settings)
    _check_ranges(normalized, method)
    return normalized


# 驗證單場實驗，補齊預設值並回傳獨立設定副本。
def validate_experiment(spec: Mapping) -> dict:
    """Validate a GUI experiment spec and return a detached normalized copy."""
    if not isinstance(spec, Mapping):
        raise ValueError("experiment spec must be an object")
    extra_top = set(spec) - {"name", "method", "settings"}
    if extra_top:
        raise ValueError(f"Unknown experiment fields: {sorted(extra_top)}")
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    name = name.strip()
    if len(name) > 120 or any(ord(character) < 32 for character in name):
        raise ValueError("name must be at most 120 characters and contain no control characters")
    method = spec.get("method")
    supplied = spec.get("settings", {})
    # 舊 MA 的 JSON 可沿用；輸出及排程一律使用正式 HRL 名稱。
    if method == "HRL-MA":
        method = "HRL"
        if isinstance(supplied, Mapping):
            supplied = copy.deepcopy(dict(supplied))
            if "HRL_MA_UPPER" in supplied:
                if "HRL_UPPER" in supplied:
                    raise ValueError("Cannot combine HRL_MA_UPPER and HRL_UPPER settings")
                supplied["HRL_UPPER"] = supplied.pop("HRL_MA_UPPER")
    if method in RETIRED_METHODS:
        raise ValueError(
            f"Method {method!r} is retired and cannot be launched")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")

    defaults = default_settings(method)
    normalized = _normalise_settings_overrides(defaults, supplied)

    # 上層回報模式隱藏的值仍保留，讓切換模式後可恢復合法原值。
    preserved_inactive_paths = {
        ("HRL_UPPER", "risk_penalty_weight"),
        ("HRL", "fragment_steps"),
        ("HRL_UPPER", "gamma"),
        ("HRL_UPPER", "average_reward_lr"),
        ("HRL_PPO_UPPER", "risk_penalty_weight"),
        ("HRL_PPO", "fragment_steps"),
        ("HRL_PPO_UPPER", "gamma"),
        ("HRL_PPO_UPPER", "average_reward_lr"),
    }
    # 其他不作用的值回復預設，避免保存誤導的覆寫或阻擋目前方法的設定。
    for group, field in inactive_setting_paths(method, normalized):
        if (group, field) in preserved_inactive_paths:
            value = normalized[group][field]
            if field == "gamma" and group in {
                    "HRL_UPPER", "HRL_PPO_UPPER"} and not 0 <= value <= 1:
                normalized[group][field] = copy.deepcopy(defaults[group][field])
            elif (field == "average_reward_lr" and group in {
                    "HRL_UPPER", "HRL_PPO_UPPER"}
                    and value <= 0):
                normalized[group][field] = copy.deepcopy(defaults[group][field])
            continue
        normalized[group][field] = copy.deepcopy(defaults[group][field])
    _check_ranges(normalized, method)
    return {"name": name, "method": method, "settings": normalized}


