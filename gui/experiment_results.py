"""瀏覽訓練結果並以 Tk Canvas 比較歷史指標，另存顯示名稱。"""
from __future__ import annotations

import csv
from heapq import nlargest
import io
import json
import math
import os
import queue
from statistics import mean
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

from runtime import fileIO as saving
from tkinter import filedialog, simpledialog, ttk

from runtime.configuration import METHODS, RETIRED_METHODS
from gui.gui_style import MONO_FONT, SMALL_FONT, configure_gui_style
from gui.setting_labels import OPTION_LABELS, field_label

# 結果瀏覽包含已退役方法；不能使用可啟動訓練的方法清單過濾歷史資料。
RESULT_METHODS = frozenset(METHODS) | RETIRED_METHODS | {"HRL-SA", "HRL-MA"}
from gui.experiment_log_view import show_training_log


_METRICS = {
    "訓練 Reward": "reward",
    "HRL 上層學習 Reward": "learning_reward",
    "訓練平均 AoI": "mean_aoi",
    "評估 AoI": "eval_aoi",
    "評估 Reward": "eval_reward",
    "評估加權功耗": "eval_weighted_power",
    "評估違規次數": "eval_violation_count",
    "累積 Wall time（秒）": "wall_time_seconds",
    "回合 Wall time（秒）": "episode_wall_time_seconds",
}
_METRIC_DESCRIPTIONS = {
    "訓練 Reward": "每步先對 UAV 的環境回報取平均，再於單一訓練回合累加。",
    "HRL 上層學習 Reward": "HRL 所有 UAV 的上層回饋合計；AoI 模式累加降低量減加權事件，環境均值模式累加 Option 均值，平均回報 SMDP 模式累加 Option 原始團隊環境回報總和（非 rho 或 TD 誤差）；歷史資料依保存值。",
    "訓練平均 AoI": "單一訓練回合內，先對裝置 AoI 取平均，再對時間步取平均。",
    "評估 AoI": "每個評估回合先對裝置與時間取平均，再跨評估回合平均。",
    "評估 Reward": "評估回合的環境累積回報平均。",
    "評估加權功耗": "各 UAV 的加權傳輸成本先取平均，再於每回合累加，最後跨評估回合平均。",
    "評估違規次數": "新資料為所有 UAV 在風險區的累積步數，再跨評估回合平均；舊資料可能是實際受罰次數。",
    "累積 Wall time（秒）": "從訓練初始化到此回合記錄完成的累積時間，含暖身與評估，不含最後寫檔。",
    "回合 Wall time（秒）": "此回合訓練、評估與導航檢查的耗時；HRL 離線初始化計入下一回合。",
}
_REQUIRED_COLUMNS = {"method", "iteration", "iteration_unit"}
_OPTIONAL_METRIC_COLUMNS = set(_METRICS.values())
_NON_TRAINING_PHASES = {
    "collect", "warmup", "offline", "pretrain", "eval", "evaluation",
}
_COLORS = ("#2468a2", "#d65f36", "#3b8b58", "#8458a6", "#b48718", "#278c91")
# 每六條共用一組顏色；後續以四種線型循環區分。
_DASH_PATTERNS = ((), (8, 4), (2, 3), (8, 3, 2, 3))
_MPL_LINESTYLES = ("solid", "dashed", "dotted", "dashdot")
# 圖例短符號依序對應實線、虛線、點線與點虛線。
_DASH_GLYPHS = ("━━", "– –", "···", "–·–")
_STATUS_LABELS = {
    "pending": "等待中", "queued": "排程中", "running": "執行中",
    "completed": "完成", "failed": "失敗", "cancelled": "已取消／略過",
    "canceled": "已取消／略過",
}
_COMPARISON_MODE_LABELS = {
    "shared": "SA/shared",
    "independent": "MA/independent",
}
_UNSET = object()


def run_config_path(run_dir):
    """Return the preferred saved config path for a run, if one exists."""
    if run_dir is None:
        return None
    directory = Path(run_dir)
    for filename in ("config.json", "worker_config.json"):
        candidate = directory / filename
        try:
            if candidate.exists() or candidate.is_symlink():
                return candidate
        except OSError:
            return candidate
    return None


def result_config_path(result):
    """Reuse a discovered source path, or resolve one for older result records."""
    if "config_path" not in result:
        return run_config_path(result.get("run_dir"))
    source_path = result.get("config_path")
    if source_path is None:
        return None
    run_dir = result.get("run_dir")
    if run_dir is None:
        return None
    source_path = Path(source_path)
    if source_path.name not in ("config.json", "worker_config.json"):
        return None
    try:
        if source_path.parent.resolve() != Path(run_dir).resolve():
            return None
        if source_path.resolve().parent != Path(run_dir).resolve():
            return None
    except OSError:
        return None
    return Path(run_dir) / source_path.name


def _read_run_config(run_dir, source_path=_UNSET):
    """Read the chosen run config without trusting paths from saved reports."""
    if source_path is _UNSET:
        source_path = run_config_path(run_dir)
    if source_path is None:
        return {}, False, None, "missing"
    source_path = Path(source_path)
    if run_dir is None:
        return {}, True, source_path, "unreadable"
    try:
        run_root = Path(run_dir).resolve()
        resolved_source = source_path.resolve()
        if (source_path.name not in ("config.json", "worker_config.json")
                or resolved_source.parent != run_root):
            return {}, True, source_path, "unreadable"
        value = saving.read_json(source_path, encoding="utf-8")
    except FileNotFoundError:
        return {}, False, source_path, "missing"
    except (OSError, UnicodeError):
        return {}, True, source_path, "unreadable"
    except ValueError:
        return {}, True, source_path, "invalid"
    if not isinstance(value, dict):
        return {}, True, source_path, "invalid_object"
    return value, False, source_path, None


def _comparison_context(run_dir):
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    if not _is_seed_dir_name(run_dir.name):
        return None
    mode_name = run_dir.parent.name.casefold()
    if mode_name not in _COMPARISON_MODE_LABELS:
        return None
    standard_layout = run_dir.parent.parent.parent.name.casefold() == "comparisons"
    marked_custom_layout = _has_valid_comparison_manifest(run_dir.parent.parent)
    if not standard_layout and not marked_custom_layout:
        return None
    return _COMPARISON_MODE_LABELS[mode_name]


def _is_seed_dir_name(name):
    name = str(name)
    return name.casefold().startswith("seed_") and name[5:].isdigit()


def _has_valid_comparison_manifest(comparison_root):
    comparison_root = Path(comparison_root)
    manifest_path = comparison_root / "comparison.json"
    try:
        if comparison_root.is_symlink() or manifest_path.is_symlink():
            return False
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("name") == "HRL lower-agent comparison"
        and isinstance(manifest.get("fixed_stage_counts"), dict)
    )


def _result_display_name(result):
    name = result["name"]
    context = result.get("comparison_context")
    if context:
        return f"{name} · {context}"
    return name


# 只檢查正式、單場與比較輸出的已知層級，不做任意遞迴。
def discover_result_files(output_root):
    root = Path(output_root).expanduser()
    found = []
    issues = 0
    seen = set()

    def is_directory_without_link(path):
        try:
            return path.is_dir() and not path.is_symlink()
        except OSError:
            return False

    def collect_methods(method_parent, run_dir=None):
        nonlocal issues
        has_result_csv = False
        scan_complete = True
        try:
            method_dirs = list(method_parent.iterdir())
        except (OSError, PermissionError):
            issues += 1
            return None
        for method_dir in method_dirs:
            try:
                is_directory = method_dir.is_dir()
                is_link = method_dir.is_symlink()
            except OSError:
                issues += 1
                scan_complete = False
                continue
            if not is_directory or is_link:
                continue
            result_datas_dir = method_dir / "Result_Datas"
            if not is_directory_without_link(result_datas_dir):
                continue
            result_path = result_datas_dir / f"Result_{method_dir.name}.csv"
            try:
                is_file = result_path.is_file()
            except OSError:
                issues += 1
                scan_complete = False
                continue
            if not is_file:
                continue
            has_result_csv = True
            try:
                identity = str(result_path.resolve()).casefold()
            except OSError:
                identity = str(result_path.absolute()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            found.append({"path": result_path, "run_dir": run_dir})
        if not scan_complete:
            return None
        return has_result_csv

    def collect_run(run_dir):
        nonlocal issues
        try:
            results_dir = run_dir / "results"
            if results_dir.is_symlink():
                return
            if results_dir.is_dir():
                has_result_csv = collect_methods(results_dir, run_dir)
                if has_result_csv is None or has_result_csv:
                    return
            config, bad_config, config_path, _reason = _read_run_config(run_dir)
            method = config.get("method")
            settings = config.get("settings")
            if config_path is None or bad_config or not isinstance(method, str) or method not in RESULT_METHODS:
                return
            if not isinstance(settings, dict):
                return
            method_dir = results_dir / method
            if method_dir.is_symlink():
                return
            result_datas_dir = method_dir / "Result_Datas"
            if result_datas_dir.is_symlink():
                return
            result_path = result_datas_dir / f"Result_{method}.csv"
            try:
                identity = str(result_path.resolve()).casefold()
            except OSError:
                identity = str(result_path.absolute()).casefold()
            if identity in seen:
                return
            seen.add(identity)
            found.append({
                "path": result_path,
                "run_dir": run_dir,
                "metadata_only": True,
                "config_path": config_path,
            })
        except OSError:
            issues += 1

    def collect_comparison_root(comparison_root):
        if comparison_root.is_symlink():
            return
        for mode_name in _COMPARISON_MODE_LABELS:
            mode_dir = comparison_root / mode_name
            if not is_directory_without_link(mode_dir):
                continue
            collect_comparison_mode(mode_dir)

    def collect_comparison_mode(mode_dir):
        nonlocal issues
        try:
            seed_dirs = list(mode_dir.iterdir())
        except (OSError, PermissionError):
            issues += 1
            return
        for seed_dir in seed_dirs:
            if not _is_seed_dir_name(seed_dir.name):
                continue
            if is_directory_without_link(seed_dir):
                collect_run(seed_dir)

    def collect_comparison_container(comparisons_root):
        nonlocal issues
        if comparisons_root.is_symlink():
            return
        try:
            comparison_dirs = list(comparisons_root.iterdir())
        except (OSError, PermissionError):
            issues += 1
            return
        for comparison_dir in comparison_dirs:
            if is_directory_without_link(comparison_dir):
                collect_comparison_root(comparison_dir)

    def collect_experiment_comparisons(experiments_root):
        nonlocal issues
        if experiments_root.is_symlink():
            return
        try:
            comparison_dirs = list(experiments_root.iterdir())
        except (OSError, PermissionError):
            issues += 1
            return
        for comparison_dir in comparison_dirs:
            if (is_directory_without_link(comparison_dir)
                    and _has_valid_comparison_manifest(comparison_dir)):
                collect_comparison_root(comparison_dir)

    def collect_runs(container, skip_experiments_container=False):
        nonlocal issues
        try:
            run_dirs = list(container.iterdir())
        except (OSError, PermissionError):
            issues += 1
            return
        for run_dir in run_dirs:
            if skip_experiments_container and run_dir.name == "Experiments":
                continue
            try:
                is_directory = run_dir.is_dir()
                is_link = run_dir.is_symlink()
            except OSError:
                issues += 1
                continue
            if is_directory and not is_link:
                collect_run(run_dir)

    try:
        if root.is_dir():
            result_datas_dir = root / "Result_Datas"
            if is_directory_without_link(result_datas_dir):
                # 使用者直接選取 METHOD 資料夾時仍採用同一固定檔名規則。
                expected = result_datas_dir / f"Result_{root.name}.csv"
                if expected.is_file():
                    found.append({"path": expected, "run_dir": None})
            else:
                collect_methods(root)
                if root.name != "Experiments":
                    collect_run(root)
                try:
                    children = list(root.iterdir())
                except (OSError, PermissionError):
                    children = []
                    issues += 1
                for child in children:
                    if child.name != "Experiments":
                        continue
                    try:
                        is_directory = child.is_dir()
                        is_link = child.is_symlink()
                    except OSError:
                        issues += 1
                        continue
                    if is_directory and not is_link:
                        collect_runs(child)
                        collect_experiment_comparisons(child)

                root_name = root.name.casefold()
                if _has_valid_comparison_manifest(root):
                    collect_comparison_root(root)
                elif root_name == "experiments":
                    collect_experiment_comparisons(root)
                elif root_name == "outputs":
                    comparisons_root = root / "Comparisons"
                    if is_directory_without_link(comparisons_root):
                        collect_comparison_container(comparisons_root)
                elif root_name == "comparisons" and root.parent.name.casefold() == "outputs":
                    collect_comparison_container(root)
                elif root.parent.name.casefold() == "comparisons":
                    collect_comparison_root(root)
                elif (root_name in _COMPARISON_MODE_LABELS
                      and (root.parent.parent.name.casefold() == "comparisons"
                           or _has_valid_comparison_manifest(root.parent))):
                    collect_comparison_mode(root)
                elif (root.parent.name.casefold() in _COMPARISON_MODE_LABELS
                      and _is_seed_dir_name(root.name)
                      and (root.parent.parent.parent.name.casefold() == "comparisons"
                           or _has_valid_comparison_manifest(root.parent.parent))):
                    collect_run(root)
    except (OSError, PermissionError):
        issues += 1

    # 自訂 queue 根目錄同樣只檢查其直接 run 子目錄，不做任意遞迴。
    if root.is_dir():
        collect_runs(root, skip_experiments_container=True)

    found.sort(key=lambda item: str(item["path"]).casefold())
    return found, issues


def _finite_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_csv(path, required_columns):
    with saving.open_read(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError("missing CSV header")
        missing = required_columns - set(columns)
        if missing:
            raise ValueError("missing required columns")
        rows = []
        incomplete = False
        try:
            for row in reader:
                if not row:
                    continue
                # None 表示列數和標頭不一致；這類列不作為已完成資料使用。
                if None in row or any(value is None for value in row.values()):
                    incomplete = True
                    continue
                rows.append(row)
        except csv.Error:
            # strict reader 在未閉合引號等尾端損壞時停止，保留已完整讀取的前綴。
            incomplete = True
    return rows, set(columns), incomplete


def _read_metadata(run_dir, filename):
    if run_dir is None:
        return {}, False
    path = Path(run_dir) / filename
    try:
        if not path.exists():
            return {}, False
        value = saving.read_json(path, encoding="utf-8")
    except (OSError, ValueError, UnicodeError):
        return {}, True
    if isinstance(value, dict):
        return value, False
    return {}, True


def _normalized_result_name(name):
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    if not normalized or "\x00" in normalized:
        return None
    return normalized


def _read_result_display_name(method_dir, fallback):
    """Read the optional per-method display alias without following symlinks."""
    method_dir = Path(method_dir)
    alias_path = method_dir / "display_name.json"
    try:
        if method_dir.is_symlink() or alias_path.is_symlink():
            return fallback, 1
        if not alias_path.exists():
            return fallback, 0
        value = saving.read_json(alias_path, encoding="utf-8")
    except FileNotFoundError:
        return fallback, 0
    except (OSError, UnicodeError, ValueError):
        return fallback, 1
    if not isinstance(value, dict):
        return fallback, 1
    name = _normalized_result_name(value.get("name"))
    if name is None:
        return fallback, 1
    return name, 0


def rename_result(result, name) -> str:
    """Persist a display name beside a result method without changing its data."""
    normalized_name = _normalized_result_name(name)
    if normalized_name is None:
        raise ValueError("結果名稱必須是非空字串，且不能包含 NUL。")
    try:
        result_path = Path(result["path"])
        method_dir = Path(result["method_dir"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("結果資料夾與 CSV 路徑不正確。") from error

    if result_path.parent.name != "Result_Datas":
        raise ValueError("結果 CSV 不在預期的 Result_Datas 資料夾中。")
    expected_method_dir = result_path.parent.parent
    try:
        if (method_dir.resolve() != expected_method_dir.resolve()
                or method_dir.is_symlink() or expected_method_dir.is_symlink()):
            raise ValueError("結果 method_dir 與 CSV 路徑不相符。")
        result_datas_dir = expected_method_dir / "Result_Datas"
        if result_datas_dir.is_symlink():
            raise ValueError("結果 Result_Datas 資料夾不能是符號連結。")
        if not method_dir.is_dir():
            raise FileNotFoundError(f"結果資料夾不存在：{method_dir}")
    except (OSError, RuntimeError) as error:
        raise ValueError("無法確認結果資料夾與 CSV 路徑。") from error

    alias_path = method_dir / "display_name.json"
    # Atomic replacement writes a fresh file without reading or following an
    # existing alias symlink, so a user can repair a broken alias explicitly.
    saving.write_json(
        alias_path, {"name": normalized_name}, strict=True, newline=True, atomic=True)
    return normalized_name


def load_result(candidate):
    """讀取一個結果 CSV；NaN、空值及無效數值不會轉成 0。"""
    path = Path(candidate["path"])
    run_dir = candidate.get("run_dir")
    if candidate.get("metadata_only"):
        config_path = candidate.get("config_path", _UNSET)
        config, bad_config, config_path, _config_reason = _read_run_config(
            run_dir, config_path)
        method = config.get("method")
        settings = config.get("settings")
        if bad_config or not isinstance(method, str) or method not in RESULT_METHODS:
            raise ValueError("invalid metadata-only config")
        if not isinstance(settings, dict) or method != path.parent.parent.name:
            raise ValueError("invalid metadata-only result identity")
        status_data, bad_status = _read_metadata(run_dir, "status.json")
        status_path = None
        if run_dir is not None and status_data and not bad_status:
            status_path = Path(run_dir) / "status.json"
        name = config.get("name") or status_data.get("name")
        if not name and run_dir is not None:
            name = Path(run_dir).name
        if not name:
            name = f"{method} 結果"
        name, bad_display_name = _read_result_display_name(path.parent.parent, str(name))
        status = status_data.get("status") or "無狀態"
        status_text = str(status)
        if "尚無結果 CSV" not in status_text:
            status_text = f"{status_text}（尚無結果 CSV）"
        points = {key: [] for key in _OPTIONAL_METRIC_COLUMNS}
        return {
            "path": path,
            "method_dir": path.parent.parent,
            "run_dir": run_dir,
            "config_path": config_path,
            "name": str(name),
            "method": method,
            "status": status_text,
            "last_iteration": None,
            "last_unit": "",
            "units": [],
            "points": points,
            "eval_reward_source": None,
            "metadata_warnings": int(bad_status) + bad_display_name,
            "metadata_only": True,
            "started_at": status_data.get("started_at"),
            "finished_at": status_data.get("finished_at"),
            "status_path": status_path,
            "comparison_context": _comparison_context(run_dir),
        }
    rows, columns, incomplete_csv = _read_csv(path, _REQUIRED_COLUMNS)
    if not (_OPTIONAL_METRIC_COLUMNS & columns):
        raise ValueError("missing supported metric")

    method_dir = path.parent.parent
    for row in rows:
        value = (row.get("method") or "").strip()
        if value and value != method_dir.name:
            raise ValueError("result method does not match its directory")
    method = method_dir.name
    legacy_hrl = method == "HRL" and "environment_return" in columns
    points = {key: [] for key in _OPTIONAL_METRIC_COLUMNS}
    iterations = []
    warnings = int(incomplete_csv)

    for row in rows:
        row_method = (row.get("method") or "").strip()
        if not row_method:
            continue
        iteration = _finite_number(row.get("iteration"))
        unit = (row.get("iteration_unit") or "").strip()
        if iteration is None or not unit:
            continue
        iterations.append((iteration, unit))
        phase = (row.get("phase") or "train").strip().casefold()
        for metric in (
            "eval_aoi", "eval_reward", "eval_weighted_power",
            "eval_violation_count",
            "wall_time_seconds", "episode_wall_time_seconds",
        ):
            value = _finite_number(row.get(metric))
            if value is not None:
                points[metric].append((iteration, value, unit))
        if phase not in _NON_TRAINING_PHASES:
            mean_aoi = _finite_number(row.get("mean_aoi"))
            if mean_aoi is not None:
                points["mean_aoi"].append((iteration, mean_aoi, unit))
            if legacy_hrl:
                environment_reward = _finite_number(row.get("environment_return"))
                if environment_reward is not None:
                    points["reward"].append((iteration, environment_reward, unit))

                learning_reward = _finite_number(row.get("learning_reward"))
                if learning_reward is None:
                    learning_reward = _finite_number(row.get("reward"))
                if learning_reward is not None:
                    points["learning_reward"].append((iteration, learning_reward, unit))
            else:
                for metric in ("reward", "learning_reward"):
                    value = _finite_number(row.get(metric))
                    if value is not None:
                        points[metric].append((iteration, value, unit))

    eval_reward_path = path.parent / "eval_reward.csv"
    sidecar_used = False
    try:
        if eval_reward_path.is_file():
            eval_rows, _, sidecar_incomplete = _read_csv(
                eval_reward_path,
                _REQUIRED_COLUMNS | {"eval_reward"},
            )
            if sidecar_incomplete:
                warnings += 1
            recorded = []
            for row in eval_rows:
                row_method = (row.get("method") or "").strip()
                iteration = _finite_number(row.get("iteration"))
                unit = (row.get("iteration_unit") or "").strip()
                value = _finite_number(row.get("eval_reward"))
                if row_method == method and iteration is not None and unit and value is not None:
                    recorded.append((iteration, value, unit))
            if recorded:
                points["eval_reward"] = recorded
                sidecar_used = True
    except (OSError, csv.Error, UnicodeError, ValueError):
        sidecar_used = False
        warnings += 1

    config_path = candidate.get("config_path", _UNSET)
    config, bad_config, config_path, _config_reason = _read_run_config(
        run_dir, config_path)
    status_data, bad_status = _read_metadata(run_dir, "status.json")
    status_path = None
    if run_dir is not None and status_data and not bad_status:
        status_path = Path(run_dir) / "status.json"
    name = config.get("name") or status_data.get("name")
    if not name and run_dir is not None:
        name = Path(run_dir).name
    if not name:
        name = f"{method} 結果"
    name, bad_display_name = _read_result_display_name(method_dir, str(name))
    status = status_data.get("status") or "無狀態"
    if iterations:
        if len(set(unit for _value, unit in iterations)) == 1:
            last_iteration, last_unit = max(iterations, key=lambda item: item[0])
        else:
            last_iteration, last_unit = iterations[-1]
    else:
        last_iteration, last_unit = None, ""

    if bad_config:
        warnings += 1
    if bad_status:
        warnings += 1
    warnings += bad_display_name
    if not sidecar_used and not points["eval_reward"]:
        try:
            if eval_reward_path.exists():
                warnings += 1
        except OSError:
            warnings += 1

    if sidecar_used:
        eval_reward_source = eval_reward_path
    else:
        eval_reward_source = path

    return {
        "path": path,
        "method_dir": method_dir,
        "run_dir": run_dir,
        "config_path": config_path,
        "name": str(name),
        "method": method,
        "status": str(status),
        "last_iteration": last_iteration,
        "last_unit": last_unit,
        "units": list(dict.fromkeys(unit for _iteration, unit in iterations)),
        "points": points,
        "eval_reward_source": eval_reward_source,
        "metadata_warnings": warnings,
        "metadata_only": False,
        "started_at": status_data.get("started_at"),
        "finished_at": status_data.get("finished_at"),
        "status_path": status_path,
        "comparison_context": _comparison_context(run_dir),
    }


def load_results(output_root):
    candidates, issues = discover_result_files(output_root)
    results = []
    for candidate in candidates:
        try:
            result = load_result(candidate)
        except (OSError, csv.Error, UnicodeError, ValueError):
            issues += 1
            continue
        issues += result.pop("metadata_warnings")
        results.append(result)
    return results, issues


def _latest_value(result, metric):
    # 指標跨單位有紀錄時，優先採用主表最新迭代的單位。
    points = result["points"].get(metric, [])
    if not points:
        return None
    unit = result.get("last_unit")
    same_unit = [point for point in points if point[2] == unit]
    if same_unit:
        return max(same_unit, key=lambda point: point[0])[1]
    return points[-1][1]


def _summary_value(result, metric):
    """累積時間取最新值，其餘指標取末 20 個有效回合平均，不混合單位。"""
    if metric == "wall_time_seconds":
        return _latest_value(result, metric)
    points = result["points"].get(metric, [])
    if not points:
        return None
    unit = result.get("last_unit")
    same_unit = [point for point in points if point[2] == unit]
    if not same_unit:
        unit = points[-1][2]
        same_unit = [point for point in points if point[2] == unit]
    recent = nlargest(20, same_unit, key=lambda point: point[0])
    # 標準 mean 可處理有限大數的平均，避免加總或分項除法的舍入溢位。
    return mean(point[1] for point in recent)


def _format_number(value):
    if value is None:
        return "—"
    # 表格採六位有效數字；資料匯出仍使用有限的原始值。
    return f"{value:.6g}"


def _path_identity(path):
    # Windows 路徑忽略大小寫，同一檔案只保留一個選取身分。
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = candidate.absolute()
    return str(candidate).casefold()


def _complete_trace_paths(result):
    trace_dir = result["method_dir"] / "Stored_Datas"
    episode_path = trace_dir / "episode.parquet"
    step_path = trace_dir / "step.parquet"
    if episode_path.is_file() and step_path.is_file():
        return episode_path, step_path
    return None


def _source_csv_identities(results):
    identities = set()
    for result in results:
        if result.get("metadata_only", False):
            continue
        for field in ("path", "eval_reward_source"):
            path = result.get(field)
            if path is not None:
                identities.add(_path_identity(path))
    return identities


def _display_status(status):
    # 畫面翻譯已知狀態，保留原值供搜尋及匯出。
    prefix, separator, suffix = str(status).partition("（")
    label = _STATUS_LABELS.get(prefix.strip().casefold(), prefix)
    return label + separator + suffix


def _sort_results(items, column, descending):
    rows = list(items)
    if column in _OPTIONAL_METRIC_COLUMNS:
        present = []
        missing = []
        for result in rows:
            value = _summary_value(result, column)
            if value is None or not math.isfinite(value):
                missing.append(result)
            else:
                present.append(result)
        present.sort(key=lambda result: _summary_value(result, column), reverse=descending)
        return present + missing
    if column == "iteration":
        present = []
        missing = []
        for result in rows:
            iteration = result.get("last_iteration")
            unit = str(result.get("last_unit") or "")
            if iteration is None or not math.isfinite(iteration) or not unit:
                missing.append(result)
            else:
                present.append(result)
        present.sort(
            key=lambda result: (
                str(result.get("last_unit") or "").casefold(),
                result["last_iteration"],
            ),
            reverse=descending,
        )
        return present + missing
    if column == "status":
        rows.sort(key=lambda result: _display_status(result.get("status", "")).casefold(),
                  reverse=descending)
    elif column in {"name", "method"}:
        rows.sort(key=lambda result: str(result.get(column, "")).casefold(),
                  reverse=descending)
    return rows


def _result_display_metadata(result, identity=None):
    """Cache the small amount of row data used by search and table rendering."""
    status = result["status"]
    path_text = str(result["path"])
    if identity is None:
        identity = _path_identity(result["path"])
    if result["last_iteration"] is None:
        iteration_text = "—"
    else:
        iteration_text = f"{_format_number(result['last_iteration'])} ({result['last_unit']})"
    return {
        "identity": identity,
        "searchable": tuple(str(value).casefold() for value in (
            result["name"], result.get("comparison_context", ""),
            result["method"], status, _display_status(status), path_text,
        )),
        "display_name": _result_display_name(result),
        "method": result["method"],
        "display_status": _display_status(status),
        "iteration_text": iteration_text,
        "summary_values": {
            metric: _summary_value(result, metric) for metric in _OPTIONAL_METRIC_COLUMNS
        },
    }


def _simplify_display_coords(coords, pixel_width):
    """Keep endpoints and each pixel column's vertical extrema for Tk drawing."""
    point_count = len(coords) // 2
    width = max(1, int(pixel_width))
    if point_count <= 2 * width + 2:
        return coords

    extrema_by_column = {}
    for index in range(point_count):
        x = coords[index * 2]
        y = coords[index * 2 + 1]
        column = max(0, min(width - 1, math.floor(x)))
        extrema = extrema_by_column.get(column)
        if extrema is None:
            extrema_by_column[column] = [index, y, index, y, index, y, index, y]
        else:
            extrema[2] = index
            extrema[3] = y
            if y < extrema[5]:
                extrema[4] = index
                extrema[5] = y
            if y > extrema[7]:
                extrema[6] = index
                extrema[7] = y

    keep = {0, point_count - 1}
    for first_index, _first_y, last_index, _last_y, min_index, _min_y, max_index, _max_y in (
            extrema_by_column.values()):
        keep.add(first_index)
        keep.add(last_index)
        keep.add(min_index)
        keep.add(max_index)
    return [coordinate for index in sorted(keep)
            for coordinate in coords[index * 2:index * 2 + 2]]


def _first_complete_trace_candidate(candidates):
    """Resolve each distinct selected path once, stopping at the first usable trace."""
    checked = set()
    for result in candidates:
        identity = _path_identity(result["path"])
        if identity in checked:
            continue
        checked.add(identity)
        trace_paths = _complete_trace_paths(result)
        if trace_paths is not None:
            return result, trace_paths
    return None, None


def comparison_coordinates(results, metric, width, height, *, left_padding=58.0):
    """回傳 Canvas 座標；不同 iteration_unit 不共用同一橫軸。"""
    all_units = set()
    series_data = []
    for index, result in enumerate(results):
        points = result["points"].get(metric, [])
        finite_points = []
        for iteration, value, unit in points:
            if math.isfinite(iteration) and math.isfinite(value):
                finite_points.append((iteration, value, unit))
                all_units.add(unit)
        if finite_points:
            finite_points.sort(key=lambda item: item[0])
            display_name = " ".join(_result_display_name(result).split())
            # 完整保留 GUI 合法名稱；長外部名稱保留首尾供辨識。
            if len(display_name) > 120:
                display_name = display_name[:87] + "…" + display_name[-32:]
            label = f"{result['method']} / {display_name} ({finite_points[0][2]})"
            style_index = (index // len(_COLORS)) % len(_DASH_PATTERNS)
            series_data.append({
                "label": label,
                "color": _COLORS[index % len(_COLORS)],
                "dash": _DASH_PATTERNS[style_index],
                "linestyle": _MPL_LINESTYLES[style_index],
                "dash_glyph": _DASH_GLYPHS[style_index],
                "points": finite_points,
            })
    if not series_data:
        return None, "目前選取的結果沒有此指標的有限數值。"
    if len(all_units) > 1:
        return None, "iteration 單位不同，請選擇相同單位的實驗再比較。"

    x_values = [point[0] for series in series_data for point in series["points"]]
    y_values = [point[1] for series in series_data for point in series["points"]]
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    x_centered = False
    if math.isclose(xmin, xmax):
        x_padding = max(1.0, abs(xmin) * 0.05)
        padded_xmin = xmin - x_padding
        padded_xmax = xmax + x_padding
        if math.isfinite(padded_xmin) and math.isfinite(padded_xmax):
            if padded_xmin < padded_xmax:
                xmin = padded_xmin
                xmax = padded_xmax
        elif xmin == xmax:
            x_centered = True
    y_centered = False
    if math.isclose(ymin, ymax):
        y_padding = max(1.0, abs(ymin) * 0.05)
        padded_ymin = ymin - y_padding
        padded_ymax = ymax + y_padding
        if math.isfinite(padded_ymin) and math.isfinite(padded_ymax):
            if padded_ymin < padded_ymax:
                ymin = padded_ymin
                ymax = padded_ymax
        elif ymin == ymax:
            y_centered = True

    left, right, top, bottom = float(left_padding), 16.0, 20.0, 44.0
    plot_width = max(1.0, float(width) - left - right)
    plot_height = max(1.0, float(height) - top - bottom)
    x_scale = max(1.0, abs(xmin), abs(xmax))
    scaled_xmin = xmin / x_scale
    scaled_xmax = xmax / x_scale
    scaled_x_span = scaled_xmax - scaled_xmin
    if not math.isfinite(scaled_x_span) or scaled_x_span <= 0:
        x_centered = True
    y_scale = max(1.0, abs(ymin), abs(ymax))
    scaled_ymin = ymin / y_scale
    scaled_ymax = ymax / y_scale
    scaled_y_span = scaled_ymax - scaled_ymin
    if not math.isfinite(scaled_y_span) or scaled_y_span <= 0:
        y_centered = True
    for series in series_data:
        coords = []
        for iteration, value, _unit in series["points"]:
            # Scale before subtracting so finite extremes cannot overflow their ranges.
            if x_centered:
                x = left + plot_width / 2
            else:
                x_fraction = (iteration / x_scale - scaled_xmin) / scaled_x_span
                x_fraction = max(0.0, min(1.0, x_fraction))
                x = left + x_fraction * plot_width
            if y_centered:
                y = top + plot_height / 2
            else:
                y_fraction = (value / y_scale - scaled_ymin) / scaled_y_span
                y_fraction = max(0.0, min(1.0, y_fraction))
                # Tk's y axis grows downward, so larger values sit higher.
                y = top + (1.0 - y_fraction) * plot_height
            if math.isfinite(x) and math.isfinite(y):
                coords.extend((x, y))
        series["coords"] = coords
    return {
        "series": series_data,
        "unit": next(iter(all_units)),
        "bounds": (xmin, xmax, ymin, ymax),
        "plot": (left, top, left + plot_width, top + plot_height),
    }, None


def _flatten_saved_settings(values, prefix=""):
    flattened = {}
    if not isinstance(values, dict):
        return flattened
    for key, value in values.items():
        field_path = str(key)
        if prefix:
            field_path = f"{prefix}.{field_path}"
        if isinstance(value, dict):
            if value:
                flattened.update(_flatten_saved_settings(value, field_path))
            else:
                flattened[field_path] = value
        else:
            # List 保留為單一值，避免把陣列索引誤當設定欄位。
            flattened[field_path] = value
    return flattened


def read_saved_settings(results):
    """Read settings from the selected run config; never change saved files."""
    columns = []
    fields = set()
    for result in results:
        run_dir = result.get("run_dir")
        metadata_only = bool(result.get("metadata_only", False))
        result_path = None
        if not metadata_only:
            result_path = result.get("path")
        source_path = result_config_path(result)
        settings = {}
        valid = False
        unknown_reason = "未提供 run_dir"
        saved_name = None
        saved_method = None
        if run_dir is not None:
            config, bad_config, source_path, config_reason = _read_run_config(
                run_dir, source_path)
            filename = "config.json"
            if source_path is not None:
                filename = source_path.name
            if config_reason == "missing":
                unknown_reason = f"{filename} 不存在"
            elif config_reason == "unreadable":
                unknown_reason = f"{filename} 無法讀取"
            elif config_reason == "invalid":
                unknown_reason = f"{filename} 格式無效"
            elif config_reason == "invalid_object":
                unknown_reason = f"{filename} 缺少 settings 物件"
            elif bad_config:
                unknown_reason = f"{filename} 無法讀取"
            elif isinstance(config.get("settings"), dict):
                valid = True
                settings = _flatten_saved_settings(config["settings"])
                fields.update(settings)
                saved_name = config.get("name")
                saved_method = config.get("method")
            else:
                unknown_reason = f"{filename} 缺少 settings 物件"
        columns.append({
            "settings": settings,
            "valid": valid,
            "unknown_reason": unknown_reason,
            "source_path": source_path,
            "comparison_context": _comparison_context(run_dir),
            "saved_name": saved_name,
            "saved_method": saved_method,
            "result_name": result.get("name", ""),
            "result_display_name": _result_display_name(result),
            "result_method": result.get("method", ""),
            "result_path": result_path,
            "metadata_only": metadata_only,
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "status_path": result.get("status_path"),
        })
    return columns, sorted(fields, key=str.casefold)


def _saved_setting_signature(column, field_path):
    if not column["valid"]:
        return "unknown", column["unknown_reason"]
    if field_path not in column["settings"]:
        return "missing",
    value = column["settings"][field_path]
    return "value", json.dumps(value, ensure_ascii=False)


def _saved_setting_text(column, field_path):
    if not column["valid"]:
        return f"未知（{column['unknown_reason']}）"
    if field_path not in column["settings"]:
        return "（未保存）"
    return json.dumps(column["settings"][field_path], ensure_ascii=False)


def _saved_setting_display_text(column, field_path):
    """Format saved values for people while keeping raw values available."""
    raw_text = _saved_setting_text(column, field_path)
    if not column["valid"] or field_path not in column["settings"]:
        return raw_text

    value = column["settings"][field_path]
    if isinstance(value, bool):
        if value:
            return "開啟（true）"
        else:
            return "關閉（false）"

    path = tuple(field_path.split("."))
    if isinstance(value, str):
        option_label = OPTION_LABELS.get(path, {}).get(value)
        if option_label is not None:
            return f"{option_label}（{value}）"
    return raw_text


def _saved_settings_difference_fields(columns, fields):
    different = []
    for field_path in fields:
        signatures = {_saved_setting_signature(column, field_path) for column in columns}
        if len(signatures) > 1:
            different.append(field_path)
    return different


class ResultsPanel(ttk.Frame):
    """結果瀏覽、顯示名稱管理與多實驗曲線比較面板。"""

    def __init__(self, parent, output_root, *, on_replay=None, on_load_settings=None,
                 on_renamed=None):
        super().__init__(parent)
        configure_gui_style(self._root())
        self.output_root = Path(output_root).expanduser()
        self.on_replay = on_replay
        self.on_load_settings = on_load_settings
        self.on_renamed = on_renamed
        self._generation = 0
        self._closed = False
        self._events = queue.Queue()
        self._poll_after_id = None
        self._auto_refresh_var = tk.BooleanVar(master=self, value=False)
        self._auto_refresh_after_id = None
        self.results = []
        self._results_by_iid = {}
        self._iid_by_path = {}
        self._iid_by_result_id = {}
        self._display_metadata_by_result_id = {}
        self._tree_iids = set()
        self._next_result_id = 0
        self._loaded_root_identity = _path_identity(self.output_root)
        self._pending_root_identity = None
        self._pending_selection_paths = None
        self._replay_target_identity = None
        self._pending_replay_target_identity = None
        self._pending_focus_identity = None
        self._sort_column = None
        self._sort_descending = False
        self._configure_redraw_after_id = None
        self.path_var = tk.StringVar(master=self, value=str(self.output_root))
        self.status_var = tk.StringVar(master=self, value="")
        self.metric_var = tk.StringVar(master=self, value="訓練 Reward")
        self.metric_description_var = tk.StringVar(
            master=self, value=_METRIC_DESCRIPTIONS[self.metric_var.get()])
        self.comparison_note_var = tk.StringVar(master=self, value="")
        self.search_var = tk.StringVar(master=self, value="")
        self.count_var = tk.StringVar(master=self, value="顯示 0 / 已載入 0 項")
        self._build_ui()
        self.bind("<Destroy>", self._on_destroy)
        self._poll_after_id = self.after(80, self._poll_events)
        self.refresh()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        # 先讓清單分攤壓縮，長狀態訊息出現時仍保留可讀的圖表高度。
        self.rowconfigure(2, weight=4, minsize=100)
        self.rowconfigure(4, weight=3, minsize=210)

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
        self._auto_refresh_button = ttk.Checkbutton(
            toolbar, text="每15秒更新", variable=self._auto_refresh_var,
            command=self._auto_refresh_toggled)
        self._auto_refresh_button.grid(row=0, column=4, padx=(8, 0))

        actions = ttk.Frame(self, padding=(8, 2, 8, 4))
        actions.grid(row=1, column=0, sticky="ew")
        action_buttons = ttk.Frame(actions)
        action_buttons.pack(fill="x")
        action_buttons.columnconfigure((1, 2, 3, 4), weight=1)
        ttk.Label(action_buttons, text="查看", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(action_buttons, text="資料", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.summary_button = ttk.Button(
            action_buttons, text="匯出全部摘要 CSV", command=self.export_summary)
        self.summary_button.grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=(4, 0))
        self.replay_button = ttk.Button(
            action_buttons, text="觀看所選結果軌跡", command=self._replay_selected,
            state="disabled")
        self.replay_button.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.settings_button = ttk.Button(
            action_buttons, text="查看保存設定…", command=self.show_saved_settings,
            state="disabled")
        self.settings_button.grid(row=0, column=2, sticky="ew", padx=(0, 5))
        self.copy_paths_button = ttk.Button(
            action_buttons, text="複製結果路徑", command=self._copy_selected_paths,
            state="disabled")
        self.copy_paths_button.grid(row=1, column=2, sticky="ew", padx=(0, 5), pady=(4, 0))
        self.load_settings_button = ttk.Button(
            action_buttons, text="載入設定至表單", command=self._load_selected_settings,
            state="disabled")
        self.load_settings_button.grid(row=1, column=3, columnspan=2, sticky="ew", pady=(4, 0))
        self.log_button = ttk.Button(
            action_buttons, text="查看訓練日誌…", command=self._show_selected_log,
            state="disabled")
        self.log_button.grid(row=0, column=3, sticky="ew", padx=(0, 5))
        self.open_location_button = ttk.Button(
            action_buttons, text="開啟所在位置", command=self._open_selected_location,
            state="disabled")
        self.open_location_button.grid(row=0, column=4, sticky="ew")
        search_toolbar = ttk.Frame(actions)
        search_toolbar.pack(fill="x", pady=(5, 0))
        ttk.Label(search_toolbar, text="搜尋：").pack(side="left")
        self.search_entry = ttk.Entry(search_toolbar, textvariable=self.search_var, width=24)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(search_toolbar, text="清除", command=self._clear_search).pack(side="left")
        self.rename_button = ttk.Button(
            search_toolbar, text="重新命名…", command=self._rename_selected,
            state="disabled")
        self.rename_button.pack(side="left", padx=(5, 0))
        ttk.Label(search_toolbar, textvariable=self.count_var, anchor="e").pack(side="right", padx=(8, 0))
        self.search_var.trace_add("write", self._search_changed)

        table_frame = ttk.Frame(self, padding=(8, 2, 8, 2))
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("name", "method", "status", "iteration", "metric", "eval_reward")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended")
        headers = {
            "name": ("實驗名稱", 270),
            "method": ("方法", 100),
            "status": ("狀態", 95),
            "iteration": ("最後 iteration", 125),
            "metric": (f"末20回合平均{self.metric_var.get()}", 230),
            "eval_reward": ("末20回合平均評估 Reward", 210),
        }
        self._header_labels = {column: title for column, (title, _width) in headers.items()}
        for column, (title, width) in headers.items():
            self.tree.heading(
                column, text=title,
                command=lambda selected_column=column: self._sort_by(selected_column))
            self.tree.column(column, width=width, minwidth=70, anchor="w")
        for column in ("iteration", "metric", "eval_reward"):
            self.tree.column(column, anchor="e")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        # 快捷鍵只綁在結果表，輸入欄仍使用各自的文字操作。
        self.tree.bind("<Control-a>", self._select_all_results)
        self.tree.bind("<Control-A>", self._select_all_results)
        self.tree.bind("<Control-c>", self._copy_result_rows)
        self.tree.bind("<Control-C>", self._copy_result_rows)
        self.tree.bind("<Escape>", self._clear_result_selection)

        self.status_label = ttk.Label(
            self, textvariable=self.status_var, anchor="w", padding=(10, 4),
            style="Muted.TLabel", justify="left")
        self.status_label.grid(row=3, column=0, sticky="ew")
        self.status_label.bind("<Configure>", lambda event: self.status_label.configure(
            wraplength=max(160, event.width - 24)))

        plot_frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        plot_frame.grid(row=4, column=0, sticky="nsew")
        plot_frame.columnconfigure(1, weight=1)
        plot_frame.rowconfigure(3, weight=1, minsize=80)
        ttk.Label(plot_frame, text="比較指標：").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.metric_box = ttk.Combobox(
            plot_frame, textvariable=self.metric_var, values=list(_METRICS),
            state="readonly", width=18)
        self.metric_box.grid(row=0, column=1, sticky="w", pady=(0, 5))
        self.metric_box.bind("<<ComboboxSelected>>", self._metric_changed)
        self.metric_description = ttk.Label(
            plot_frame, textvariable=self.metric_description_var, anchor="w",
            justify="left", style="Muted.TLabel")
        self.metric_description.grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.metric_description.bind(
            "<Configure>",
            lambda event: self.metric_description.configure(
                wraplength=max(100, event.width - 8)),
        )
        self.comparison_note = ttk.Label(
            plot_frame, textvariable=self.comparison_note_var, anchor="w",
            justify="left", foreground="#66551f")
        self.comparison_note.grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.comparison_note.grid_remove()
        self.comparison_note.bind("<Configure>", lambda event: self.comparison_note.configure(
            wraplength=max(160, event.width - 8)))
        export_controls = ttk.Frame(plot_frame)
        export_controls.grid(row=0, column=2, sticky="e", pady=(0, 5))
        self.export_data_button = ttk.Button(
            export_controls, text="匯出比較數據 CSV…", command=self.export_data,
            state="disabled")
        self.export_data_button.pack(side="left", padx=(0, 5))
        self.export_plot_button = ttk.Button(
            export_controls, text="匯出所選比較圖…", command=self.export_plot,
            state="disabled")
        self.export_plot_button.pack(side="left")
        self.canvas = tk.Canvas(
            plot_frame, background="white", highlightthickness=1,
            highlightbackground="#cccccc", height=210)
        self.canvas.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.canvas.bind("<Configure>", self._schedule_comparison_redraw)
        self.legend_frame = ttk.Frame(plot_frame)
        self.legend_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.legend_text = tk.Text(
            self.legend_frame, height=3, wrap="word", state="disabled", borderwidth=0,
            font=SMALL_FONT)
        legend_scroll = ttk.Scrollbar(
            self.legend_frame, orient="vertical", command=self.legend_text.yview)
        self.legend_text.configure(yscrollcommand=legend_scroll.set)
        self.legend_text.pack(side="left", fill="x", expand=True)
        legend_scroll.pack(side="right", fill="y")
        self._draw_comparison()

    def set_root(self, path):
        """更新查詢位置；是否掃描由呼叫者透過 refresh() 決定。"""
        self.output_root = Path(path).expanduser()
        self.path_var.set(str(self.output_root))
        if _path_identity(self.output_root) != self._loaded_root_identity:
            self.tree.selection_remove(self.tree.selection())
            self._selection_changed()
            self.status_var.set("結果資料夾已變更，請按重新整理載入。")

    def refresh(self):
        if self._closed:
            return
        path_text = self.path_var.get().strip()
        if path_text:
            self.output_root = Path(path_text).expanduser()
        root_identity = _path_identity(self.output_root)
        self._generation += 1
        generation = self._generation
        root = self.output_root
        if (self._pending_root_identity == root_identity
                and self._pending_selection_paths is not None):
            selected_paths = set(self._pending_selection_paths)
            replay_target = self._pending_replay_target_identity
            focus_identity = self._pending_focus_identity
        elif root_identity == self._loaded_root_identity:
            selected_paths = self._selected_path_identities()
            replay_target = self._replay_target_identity
            focus_identity = self._focused_path_identity()
        else:
            selected_paths = set()
            replay_target = None
            focus_identity = None
        if replay_target not in selected_paths:
            replay_target = None
        if focus_identity not in selected_paths:
            focus_identity = None
        if root_identity != self._loaded_root_identity:
            self._iid_by_path.clear()
            self._next_result_id = 0
            self._iid_by_result_id.clear()
            self._display_metadata_by_result_id.clear()
            self._replay_target_identity = None
        self._pending_root_identity = root_identity
        self._pending_selection_paths = selected_paths
        self._pending_replay_target_identity = replay_target
        self._pending_focus_identity = focus_identity
        if root_identity != self._loaded_root_identity:
            self.results = []
            self._results_by_iid.clear()
            self._clear_tree_rows()
            self.count_var.set("顯示 0 / 已載入 0 項")
            self._draw_comparison()
        self.status_var.set("正在讀取歷史結果…")
        # 背景工作只持有純資料佇列，避免關閉後由背景執行緒釋放 Tk 元件。
        result_queue = self._events

        def work():
            try:
                results, issues = load_results(root)
            except Exception:
                results, issues = [], 1
            result_queue.put((generation, results, issues))

        try:
            threading.Thread(target=work, name="experiment-results-scan", daemon=True).start()
        except Exception as error:
            # 沒有工作執行緒會回報完成，必須解除 pending 才能重新整理。
            self._pending_root_identity = None
            self._pending_selection_paths = None
            self._pending_replay_target_identity = None
            self._pending_focus_identity = None
            self.status_var.set(f"無法啟動結果讀取：{error}")

    def _auto_refresh_toggled(self):
        if self._closed:
            return
        if self._auto_refresh_var.get():
            if self._auto_refresh_after_id is None:
                self._auto_refresh_after_id = self.after(15000, self._auto_refresh_tick)
        else:
            if self._auto_refresh_after_id is not None:
                try:
                    self.after_cancel(self._auto_refresh_after_id)
                except tk.TclError:
                    pass
                self._auto_refresh_after_id = None

    def _auto_refresh_tick(self):
        self._auto_refresh_after_id = None
        if self._closed or not self._auto_refresh_var.get():
            return
        try:
            is_mapped = self.winfo_ismapped()
        except tk.TclError:
            return
        if is_mapped:
            path_text = self.path_var.get().strip()
            if path_text:
                visible_root = Path(path_text).expanduser()
            else:
                visible_root = self.output_root
            visible_root_identity = _path_identity(visible_root)
            if visible_root_identity != self._pending_root_identity:
                self.refresh()
        if not self._closed and self._auto_refresh_var.get():
            try:
                self._auto_refresh_after_id = self.after(15000, self._auto_refresh_tick)
            except tk.TclError:
                self._auto_refresh_after_id = None

    def _poll_events(self):
        if self._closed:
            return
        while True:
            try:
                generation, results, issues = self._events.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation or self._closed:
                continue
            entry_path = self.path_var.get().strip()
            if entry_path:
                visible_root_identity = _path_identity(entry_path)
            else:
                visible_root_identity = _path_identity(self.output_root)
            if visible_root_identity != self._pending_root_identity:
                continue
            self._install_results(results, issues, generation)
        if not self._closed:
            self._poll_after_id = self.after(80, self._poll_events)

    def _install_results(self, results, issues, generation=None):
        if generation is not None and generation != self._generation:
            return
        same_root = self._pending_root_identity == self._loaded_root_identity
        if same_root:
            top_identity = self._top_visible_path_identity()
            # 同資料夾重整時，以背景讀取完成當下的清單選取為準。
            restore_paths = self._selected_path_identities()
            replay_target = self._replay_target_identity
            focus_identity = self._focused_path_identity()
        else:
            top_identity = None
            restore_paths = self._pending_selection_paths
            replay_target = self._pending_replay_target_identity
            focus_identity = self._pending_focus_identity
        unique_results = {}
        for result in results:
            identity = _path_identity(result["path"])
            if identity not in unique_results:
                unique_results[identity] = result
        self.results = list(unique_results.values())
        self._results_by_iid = {}
        self._iid_by_result_id = {}
        self._display_metadata_by_result_id = {}
        live_iids = set()
        for result in self.results:
            identity = _path_identity(result["path"])
            iid = self._iid_by_path.get(identity)
            if iid is None:
                iid = str(self._next_result_id)
                self._next_result_id += 1
                self._iid_by_path[identity] = iid
            self._results_by_iid[iid] = result
            self._iid_by_result_id[id(result)] = iid
            self._display_metadata_by_result_id[id(result)] = _result_display_metadata(
                result, identity)
            live_iids.add(iid)
        for iid in self._tree_iids - live_iids:
            if self.tree.exists(iid):
                self.tree.delete(iid)
        self._tree_iids.intersection_update(live_iids)
        if self._pending_root_identity is not None:
            self._loaded_root_identity = self._pending_root_identity
        self._pending_root_identity = None
        self._pending_selection_paths = None
        self._pending_replay_target_identity = None
        self._pending_focus_identity = None
        if restore_paths is not None and replay_target in restore_paths:
            self._replay_target_identity = replay_target
        else:
            self._replay_target_identity = None
        if not results:
            if issues:
                self.status_var.set("沒有可用結果；部分檔案或資料夾無法讀取。")
            else:
                self.status_var.set("此資料夾沒有可用的結果 CSV。")
        elif issues:
            self.status_var.set(
                f"已載入 {len(self.results)} 項結果；略過 {issues} 個無效或無法讀取的項目。")
        else:
            self.status_var.set(f"已載入 {len(self.results)} 項結果。")
        self._render_rows(
            restore_paths, focus_identity, capture_focus=False,
            preserve_top_identity=top_identity)

    def _clear_tree_rows(self):
        row_iids = set(self._tree_iids)
        row_iids.update(self.tree.get_children())
        for iid in row_iids:
            if self.tree.exists(iid):
                self.tree.delete(iid)
        self._tree_iids.clear()

    def _top_visible_path_identity(self):
        for iid in self.tree.get_children():
            if not self.tree.bbox(iid):
                continue
            result = self._results_by_iid.get(iid)
            if result is not None:
                return _path_identity(result["path"])
        return None

    def _selected_path_identities(self):
        selected_paths = set()
        for iid in self.tree.selection():
            result = self._results_by_iid.get(iid)
            if result is not None:
                selected_paths.add(_path_identity(result["path"]))
        return selected_paths

    def _focused_path_identity(self):
        focused = self._results_by_iid.get(self.tree.focus())
        if focused is None:
            return None
        return _path_identity(focused["path"])

    def _visible_results(self):
        query = self.search_var.get().casefold()
        visible = []
        for result in self.results:
            metadata = self._display_metadata_by_result_id.get(id(result))
            if metadata is None:
                metadata = _result_display_metadata(result)
                self._display_metadata_by_result_id[id(result)] = metadata
            if any(query in value for value in metadata["searchable"]):
                visible.append(result)
        if self._sort_column is not None:
            sort_column = self._sort_column
            if sort_column == "metric":
                sort_column = _METRICS.get(self.metric_var.get())
            if sort_column is not None:
                if sort_column in _OPTIONAL_METRIC_COLUMNS:
                    present = []
                    missing = []
                    for result in visible:
                        metadata = self._display_metadata_by_result_id.get(id(result))
                        if metadata is None:
                            metadata = _result_display_metadata(result)
                            self._display_metadata_by_result_id[id(result)] = metadata
                        value = metadata["summary_values"].get(sort_column)
                        if value is None or not math.isfinite(value):
                            missing.append(result)
                        else:
                            present.append(result)
                    present.sort(
                        key=lambda result: self._display_metadata_by_result_id[
                            id(result)]["summary_values"][sort_column],
                        reverse=self._sort_descending)
                    visible = present + missing
                else:
                    visible = _sort_results(
                        visible, sort_column, self._sort_descending)
        return visible

    def _render_rows(self, restore_paths=None, focus_identity=None, *,
                     capture_focus=True, preserve_top_identity=_UNSET):
        if restore_paths is None:
            restore_paths = self._selected_path_identities()
        if capture_focus:
            focus_identity = self._focused_path_identity()
        if preserve_top_identity is _UNSET:
            preserve_top_identity = self._top_visible_path_identity()
        visible = self._visible_results()
        visible_iids_by_path = {}
        desired_rows = []
        metric = _METRICS.get(self.metric_var.get())
        for result in visible:
            iid = self._iid_by_result_id.get(id(result))
            if iid is None:
                identity = _path_identity(result["path"])
                iid = self._iid_by_path.get(identity)
                if iid is None:
                    iid = str(self._next_result_id)
                    self._next_result_id += 1
                    self._iid_by_path[identity] = iid
                self._iid_by_result_id[id(result)] = iid
                self._results_by_iid[iid] = result
            metadata = self._display_metadata_by_result_id.get(id(result))
            if metadata is None:
                metadata = _result_display_metadata(result)
                self._display_metadata_by_result_id[id(result)] = metadata
            if metric:
                metric_text = _format_number(metadata["summary_values"].get(metric))
            else:
                metric_text = "—"
            values = (
                metadata["display_name"], metadata["method"],
                metadata["display_status"], metadata["iteration_text"],
                metric_text,
                _format_number(metadata["summary_values"].get("eval_reward")),
            )
            self._results_by_iid[iid] = result
            identity = metadata["identity"]
            visible_iids_by_path[identity] = iid
            desired_rows.append((iid, values))

        desired_iids = {iid for iid, _values in desired_rows}
        current_iids = list(self.tree.get_children())
        for iid in current_iids:
            if iid not in desired_iids:
                self.tree.detach(iid)
        current_iids = [iid for iid in current_iids if iid in desired_iids]
        for index, (iid, values) in enumerate(desired_rows):
            if iid not in self._tree_iids or not self.tree.exists(iid):
                self.tree.insert("", index, iid=iid, values=values)
                self._tree_iids.add(iid)
                current_iids.insert(index, iid)
            else:
                if iid not in current_iids:
                    self.tree.move(iid, "", index)
                    current_iids.insert(index, iid)
                else:
                    current_index = current_iids.index(iid)
                    if current_index != index:
                        self.tree.move(iid, "", index)
                        current_iids.pop(current_index)
                        current_iids.insert(index, iid)
                if tuple(self.tree.item(iid, "values")) != values:
                    self.tree.item(iid, values=values)

        selected_iids = self.tree.selection()
        if selected_iids:
            self.tree.selection_remove(*selected_iids)
        visible_iids = []
        for iid, _values in desired_rows:
            metadata = self._display_metadata_by_result_id[
                id(self._results_by_iid[iid])]
            if metadata["identity"] in restore_paths:
                visible_iids.append(iid)
        if visible_iids:
            self.tree.selection_set(*visible_iids)
        if focus_identity in visible_iids_by_path:
            self.tree.focus(visible_iids_by_path[focus_identity])
        else:
            self.tree.focus("")
        top_iid = visible_iids_by_path.get(preserve_top_identity)
        if top_iid is not None:
            item_count = len(desired_rows)
            if item_count:
                self.tree.yview_moveto(self.tree.index(top_iid) / item_count)
        self.count_var.set(f"顯示 {len(visible)} / 已載入 {len(self.results)} 項")
        self._selection_changed()

    def _search_changed(self, *_args):
        self._render_rows()

    def _clear_search(self):
        self.search_var.set("")

    def _sort_by(self, column):
        selected_paths = self._selected_path_identities()
        if self._sort_column == column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = False
        self._update_sort_indicators()
        self._render_rows(selected_paths)

    def _update_sort_indicators(self):
        for column, label in self._header_labels.items():
            heading = label
            if column == self._sort_column:
                if self._sort_descending:
                    heading = f"{label} ▼"
                else:
                    heading = f"{label} ▲"
            self.tree.heading(column, text=heading)

    def _selected_results(self):
        return [self._results_by_iid[iid] for iid in self.tree.selection()
                if iid in self._results_by_iid]

    def _select_all_results(self, _event=None):
        # 只選目前搜尋後可見的列。
        self.tree.selection_set(self.tree.get_children())
        self._selection_changed()
        return "break"

    def _clear_result_selection(self, _event=None):
        self.tree.selection_remove(self.tree.selection())
        self._selection_changed()
        return "break"

    def _copy_result_rows(self, _event=None):
        selected = set(self.tree.selection())
        if not selected:
            self.status_var.set("請先選取要複製的結果列。")
            return "break"
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow([self._header_labels[column] for column in self.tree["columns"]])
        # 按畫面順序複製顯示字串，TSV 引號保留名稱中的 Tab 與換行。
        for iid in self.tree.get_children():
            if iid in selected:
                writer.writerow(self.tree.item(iid, "values"))
        try:
            self.clipboard_clear()
            self.clipboard_append(buffer.getvalue())
        except tk.TclError as error:
            self.status_var.set(f"無法複製結果列：{error}")
            return "break"
        self.status_var.set(f"已複製 {len(selected)} 列結果。")
        return "break"

    def _selection_changed(self, event=None):
        selected = self._selected_results()
        if len(selected) == 1:
            self.open_location_button.configure(state="normal")
        else:
            self.open_location_button.configure(state="disabled")
        if hasattr(self, "rename_button"):
            if len(selected) == 1:
                self.rename_button.configure(state="normal")
            else:
                self.rename_button.configure(state="disabled")
        selected_paths = {_path_identity(result["path"]) for result in selected}
        candidates = []
        focused_result = None
        if event is not None:
            focus_iid = self.tree.focus()
            focused = self._results_by_iid.get(focus_iid)
            if focused is not None:
                focused_identity = _path_identity(focused["path"])
                if focused_identity in selected_paths:
                    focused_result = focused
        if focused_result is not None:
            candidates.append(focused_result)

        if self._replay_target_identity in selected_paths:
            for result in selected:
                identity = _path_identity(result["path"])
                if identity == self._replay_target_identity:
                    candidates.append(result)
                    break

        candidates.extend(selected)
        target, _trace_paths = _first_complete_trace_candidate(candidates)

        if target is None:
            self._replay_target_identity = None
        else:
            self._replay_target_identity = _path_identity(target["path"])
        enabled = self.on_replay is not None and target is not None
        if enabled:
            self.replay_button.configure(state="normal")
        else:
            self.replay_button.configure(state="disabled")
        copyable_results = []
        for result in selected:
            if not result.get("metadata_only", False):
                copyable_results.append(result)
        if copyable_results:
            self.copy_paths_button.configure(state="normal")
        else:
            self.copy_paths_button.configure(state="disabled")
        if selected:
            self.settings_button.configure(state="normal")
        else:
            self.settings_button.configure(state="disabled")
        load_settings_enabled = False
        if self.on_load_settings is not None and len(selected) == 1:
            config_path = result_config_path(selected[0])
            if config_path is not None:
                try:
                    load_settings_enabled = config_path.is_file()
                except OSError:
                    load_settings_enabled = False
        if load_settings_enabled:
            self.load_settings_button.configure(state="normal")
        else:
            self.load_settings_button.configure(state="disabled")
        log_enabled = False
        if len(selected) == 1:
            run_dir = selected[0].get("run_dir")
            if run_dir is not None:
                try:
                    log_enabled = (Path(run_dir) / "training.log").is_file()
                except OSError:
                    log_enabled = False
        if log_enabled:
            self.log_button.configure(state="normal")
        else:
            self.log_button.configure(state="disabled")
        contexts = sorted({
            result.get("comparison_context")
            for result in selected
            if result.get("comparison_context")
        })
        if contexts:
            context_text = "、".join(contexts)
            self.comparison_note_var.set(
                f"比較組別：{context_text}。曲線顯示逐回合歷史；最終固定評估摘要請看 comparison.md。")
            self.comparison_note.grid()
        else:
            self.comparison_note_var.set("")
            self.comparison_note.grid_remove()
        self._draw_comparison()

    def _rename_selected(self):
        selected = self._selected_results()
        if len(selected) != 1:
            self.status_var.set("請選取單一結果後重新命名。")
            return
        result = selected[0]
        entered_name = simpledialog.askstring(
            "重新命名結果", "實驗名稱：", parent=self.winfo_toplevel(),
            initialvalue=result["name"])
        if entered_name is None:
            return
        try:
            normalized_name = rename_result(result, entered_name)
        except Exception as error:
            self.status_var.set(f"重新命名失敗：{error}")
            return

        result["name"] = normalized_name
        self._display_metadata_by_result_id.pop(id(result), None)
        selected_paths = self._selected_path_identities()
        focus_identity = self._focused_path_identity()
        self._render_rows(
            selected_paths, focus_identity, capture_focus=False)
        self.status_var.set(f"結果已重新命名為「{normalized_name}」。")
        if self.on_renamed is not None:
            try:
                self.on_renamed(result)
            except Exception as error:
                self.status_var.set(
                    f"結果已重新命名為「{normalized_name}」，但其他頁面更新失敗：{error}")
        else:
            # Invalidate any scan that started before the saved alias changed.
            self.refresh()

    # 優先開啟整場實驗的輸出目錄；舊格式結果使用方法目錄。
    def _open_selected_location(self):
        selected = self._selected_results()
        if len(selected) != 1:
            self.status_var.set("請選取單一實驗後開啟所在位置。")
            return
        result = selected[0]
        target = result.get("run_dir")
        if target is None:
            target = result["method_dir"]
        try:
            target = Path(target).expanduser().resolve()
            if not target.is_dir():
                raise FileNotFoundError(f"資料夾不存在或不是資料夾：{target}")
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except (OSError, ValueError) as error:
            self.status_var.set(f"無法開啟所在位置：{error}")
            return
        self.status_var.set(f"已開啟實驗資料夾：{target}")

    # 日誌來自所選實驗；尚無結果 CSV 時也可查失敗原因。
    def _show_selected_log(self):
        selected = self._selected_results()
        if len(selected) != 1 or selected[0].get("run_dir") is None:
            self.status_var.set("請選取一場有訓練日誌的實驗。")
            return None
        log_path = Path(selected[0]["run_dir"]) / "training.log"
        try:
            return show_training_log(self, log_path)
        except (OSError, tk.TclError) as error:
            self.status_var.set(f"無法開啟訓練日誌：{error}")
            return None

    # 每個所選結果一行 CSV 路徑，方便貼至檔案工具或研究紀錄。
    def _copy_selected_paths(self):
        selected = self._selected_results()
        if not selected:
            self.status_var.set("請先選取要複製路徑的結果。")
            return
        copyable_results = []
        for result in selected:
            if not result.get("metadata_only", False):
                copyable_results.append(result)
        skipped = len(selected) - len(copyable_results)
        if not copyable_results:
            self.status_var.set("所選實驗尚無結果 CSV，沒有可複製的路徑。")
            return
        paths = "\n".join(
            str(result["path"].resolve()) for result in copyable_results)
        try:
            self.clipboard_clear()
            self.clipboard_append(paths)
        except tk.TclError as error:
            self.status_var.set(f"無法複製結果路徑：{error}")
            return
        if skipped:
            self.status_var.set(
                f"已複製 {len(copyable_results)} 個結果 CSV 路徑；"
                f"略過 {skipped} 個尚無結果 CSV 的實驗。")
        else:
            self.status_var.set(f"已複製 {len(copyable_results)} 個結果 CSV 路徑。")

    def _draw_comparison(self):
        if not hasattr(self, "canvas"):
            return
        self.export_data_button.configure(state="disabled")
        self.export_plot_button.configure(state="disabled")
        self.canvas.delete("all")
        selected = self._selected_results()
        if not selected:
            self._update_legend([])
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2,
                text="選取一個或多個實驗以比較曲線", fill="#777777")
            return
        metric = _METRICS.get(self.metric_var.get())
        if metric is None:
            self._update_legend([])
            return
        has_finite_data = False
        for result in selected:
            if result.get("metadata_only", False):
                continue
            for iteration, value, _unit in result["points"].get(metric, []):
                if math.isfinite(iteration) and math.isfinite(value):
                    has_finite_data = True
                    break
            if has_finite_data:
                break
        if has_finite_data:
            self.export_data_button.configure(state="normal")
        width = max(80, self.canvas.winfo_width())
        height = max(80, self.canvas.winfo_height())
        # 依目前字型留出完整的科學記數法刻度寬度，不縮小文字或更改 DPI。
        tick_width = int(self.canvas.tk.call("font", "measure", SMALL_FONT, "-1.23457e+308"))
        coordinates, hint = comparison_coordinates(
            selected, metric, width, height, left_padding=tick_width + 18)
        if coordinates is None:
            self._update_legend([])
            self.canvas.create_text(
                width / 2, height / 2, text=hint, fill="#777777", width=width - 24)
            return

        self.export_plot_button.configure(state="normal")
        self._update_legend(coordinates["series"])
        # 部分實驗缺少指標時說明實際繪製數量，避免誤以為選取失效。
        series_count = len(coordinates["series"])
        if series_count < len(selected):
            self.canvas.create_text(
                width / 2, 9, text=f"顯示 {series_count}/{len(selected)} 個實驗，其餘缺少此指標",
                fill="#666666", font=SMALL_FONT)

        left, top, right, bottom = coordinates["plot"]
        self.canvas.create_line(left, bottom, right, bottom, fill="#aaaaaa")
        self.canvas.create_line(left, top, left, bottom, fill="#aaaaaa")
        xmin, xmax, ymin, ymax = coordinates["bounds"]
        self.canvas.create_text(left, bottom + 14, text=_format_number(xmin), anchor="w",
                                fill="#526174", font=SMALL_FONT, tags=("comparison-tick",))
        self.canvas.create_text(right, bottom + 14, text=_format_number(xmax), anchor="e",
                                fill="#526174", font=SMALL_FONT, tags=("comparison-tick",))
        self.canvas.create_text(left - 5, top, text=_format_number(ymax), anchor="e",
                                fill="#526174", font=SMALL_FONT, tags=("comparison-tick",))
        self.canvas.create_text(left - 5, bottom, text=_format_number(ymin), anchor="e",
                                fill="#526174", font=SMALL_FONT, tags=("comparison-tick",))
        for series in coordinates["series"]:
            coords = _simplify_display_coords(series["coords"], width)
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords, fill=series["color"], width=2, dash=series["dash"])
            elif len(coords) == 2:
                x, y = coords
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        fill=series["color"], outline="")
        axis_title = {"episode": "訓練回合", "step": "步數", "epoch": "Epoch"}.get(
            coordinates["unit"], f"Iteration ({coordinates['unit']})")
        self.canvas.create_text(
            (left + right) / 2, height - 5,
            text=axis_title, fill="#526174", anchor="s",
            font=SMALL_FONT, tags=("comparison-axis-title",))

    def _schedule_comparison_redraw(self, _event=None):
        if self._closed or self._configure_redraw_after_id is not None:
            return
        try:
            self._configure_redraw_after_id = self.after_idle(
                self._comparison_redraw_idle)
        except tk.TclError:
            self._configure_redraw_after_id = None

    def _comparison_redraw_idle(self):
        self._configure_redraw_after_id = None
        if not self._closed:
            self._draw_comparison()

    # 圖例放在可捲動的獨立區域，多選時不遮曲線或互相覆蓋。
    def _update_legend(self, series):
        top = self.legend_text.yview()[0]
        self.legend_text.configure(state="normal")
        self.legend_text.delete("1.0", "end")
        for item in series:
            # 缺指標的實驗會被略過，以實際顏色命名避免圖例重新著色。
            tag = f"legend_{item['color']}"
            self.legend_text.tag_configure(tag, foreground=item["color"])
            self.legend_text.insert(
                "end", f"{item['dash_glyph']} {item['label']}\n", tag)
        self.legend_text.configure(state="disabled")
        self.legend_text.yview_moveto(top)
        if series:
            self.legend_frame.grid()
        else:
            self.legend_frame.grid_remove()

    def export_data(self):
        selected = self._selected_results()
        metric = _METRICS.get(self.metric_var.get())
        if not selected or metric is None:
            self.status_var.set("請先選取結果及比較指標。")
            return

        rows = []
        for result in selected:
            if result.get("metadata_only", False):
                continue
            for iteration, value, unit in result["points"].get(metric, []):
                if math.isfinite(iteration) and math.isfinite(value):
                    source_path = result["path"]
                    if metric == "eval_reward":
                        source_path = result.get("eval_reward_source") or source_path
                    rows.append({
                        "name": _result_display_name(result),
                        "method": result["method"],
                        "iteration": iteration,
                        "iteration_unit": unit,
                        "metric": metric,
                        "value": value,
                        "result_csv": str(Path(result["path"]).resolve()),
                        "source_csv": str(Path(source_path).resolve()),
                    })
        if not rows:
            self.status_var.set("目前選取的結果沒有此指標的有限數值。")
            return

        # 存檔視窗仍會處理背景更新，保留開啟前的來源身分。
        protected_sources = _source_csv_identities(self.results)
        path = filedialog.asksaveasfilename(
            parent=self, title="匯出所選比較數據 CSV", defaultextension=".csv",
            initialfile=f"comparison-{metric}.csv",
            filetypes=(("CSV 檔案", "*.csv"),))
        if not path:
            self.status_var.set("已取消比較數據 CSV 匯出。")
            return
        protected_sources.update(_source_csv_identities(self.results))
        if _path_identity(path) in protected_sources:
            self.status_var.set("目標是已載入的來源 CSV，請選擇新的匯出檔。")
            return
        fieldnames = (
            "name", "method", "iteration", "iteration_unit", "metric", "value",
            "result_csv", "source_csv",
        )
        try:
            saving.write_csv(path, fieldnames, rows, dict_rows=True)
        except Exception as error:
            self.status_var.set(f"比較數據 CSV 匯出失敗：{error}")
            return
        self.status_var.set(f"比較數據 CSV 已匯出至 {path}")

    # 用保存的原始指標繪製可分享的圖；Matplotlib 只在匯出時載入。
    def export_plot(self):
        selected = self._selected_results()
        metric = _METRICS.get(self.metric_var.get())
        if not selected or metric is None:
            self.status_var.set("請先選取結果及比較指標。")
            return
        coordinates, hint = comparison_coordinates(selected, metric, 900, 500)
        if coordinates is None:
            self.status_var.set(hint)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="匯出所選結果比較圖", defaultextension=".png",
            initialfile=f"comparison-{metric}.png",
            filetypes=(("PNG 圖片", "*.png"), ("SVG 向量圖", "*.svg"),
                       ("PDF 圖表", "*.pdf")))
        if not path:
            return
        figure = None
        try:
            from matplotlib.figure import Figure
            from matplotlib.font_manager import FontProperties, fontManager

            font = FontProperties()
            installed_fonts = {item.name for item in fontManager.ttflist}
            for name in ("Microsoft JhengHei", "Noto Sans CJK TC", "Microsoft YaHei", "SimHei"):
                if name in installed_fonts:
                    font.set_family(name)
                    break
            figure = Figure(figsize=(9, 5), dpi=150)
            axes = figure.subplots()
            for series in coordinates["series"]:
                points = series["points"]
                if len(points) == 1:
                    marker = "o"
                else:
                    marker = ""
                axes.plot([point[0] for point in points], [point[1] for point in points],
                          color=series["color"], label=series["label"], marker=marker,
                          linestyle=series["linestyle"])
            axes.set_xlabel(f"Iteration ({coordinates['unit']})", fontproperties=font)
            axes.set_ylabel(self.metric_var.get(), fontproperties=font)
            axes.grid(True, alpha=0.25)
            # 圖例放在座標軸外，選取多場實驗時不遮住曲線。
            axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1), prop=font)
            saving.save_figure(figure, path, bbox_inches="tight")
        except Exception as error:
            self.status_var.set(f"比較圖匯出失敗：{error}")
            return
        finally:
            if figure is not None:
                figure.clear()
        self.status_var.set(f"比較圖已匯出至 {path}")

    def _metric_changed(self, _event=None):
        label = self.metric_var.get()
        self.metric_description_var.set(_METRIC_DESCRIPTIONS[label])
        if _METRICS[label] == "wall_time_seconds":
            self._header_labels["metric"] = label
        else:
            self._header_labels["metric"] = f"末20回合平均{label}"
        self._update_sort_indicators()
        self._render_rows()

    def _choose_root(self):
        selected = filedialog.askdirectory(
            parent=self, title="選擇結果資料夾", initialdir=str(self.output_root))
        if selected:
            self.set_root(selected)
            self.refresh()

    def _replay_selected(self):
        if self.on_replay is None:
            return
        selected = self._selected_results()
        target = None
        trace_paths = None
        for result in selected:
            if _path_identity(result["path"]) == self._replay_target_identity:
                trace_paths = _complete_trace_paths(result)
                if trace_paths is not None:
                    target = result
                break
        if target is None:
            focus_identity = self._focused_path_identity()
            for result in selected:
                if _path_identity(result["path"]) != focus_identity:
                    continue
                trace_paths = _complete_trace_paths(result)
                if trace_paths is not None:
                    target = result
                break
        if target is None:
            for result in selected:
                trace_paths = _complete_trace_paths(result)
                if trace_paths is not None:
                    target = result
                    break
        if target is None:
            self._replay_target_identity = None
            self.replay_button.configure(state="disabled")
            self.status_var.set("所選結果沒有完整的 episode.parquet 與 step.parquet。")
            return

        self._replay_target_identity = _path_identity(target["path"])
        episode_path, step_path = trace_paths
        self.on_replay(episode_path, step_path)
        self.status_var.set(f"已開啟 {_result_display_name(target)} 的軌跡。")

    def _load_selected_settings(self):
        if self.on_load_settings is None:
            return
        selected = self._selected_results()
        if len(selected) != 1:
            self.load_settings_button.configure(state="disabled")
            self.status_var.set("請選取單一歷史結果以載入保存設定。")
            return
        result = selected[0]
        config_path = result_config_path(result)
        if config_path is None:
            self.load_settings_button.configure(state="disabled")
            self.status_var.set("所選結果沒有可載入的 run 設定檔。")
            return
        try:
            config_exists = config_path.is_file()
        except OSError:
            config_exists = False
        if not config_exists:
            self.load_settings_button.configure(state="disabled")
            self.status_var.set("保存設定檔已不存在或無法讀取。")
            return
        loaded = self.on_load_settings(result)
        if loaded:
            if result.get("comparison_context"):
                self.status_var.set(
                    "已載入比較來源設定。新實驗會使用目前正式 HRL（獨立下層與固定階段配額）；"
                    "收斂與導航指標只作診斷。可修改後加入佇列。")
            else:
                self.status_var.set("已載入實驗表單，可修改後加入佇列。")

    def show_saved_settings(self):
        selected = self._selected_results()
        if not selected:
            self.status_var.set("請先選取一項或多項歷史結果。")
            return None
        columns, fields = read_saved_settings(selected)
        valid_count = 0
        for column in columns:
            if column["valid"]:
                valid_count += 1
        if valid_count == 0:
            self.status_var.set("所選結果沒有可讀取的保存設定；未開啟比較視窗。")
            return None

        window = tk.Toplevel(self)
        window.title("歷史結果的保存設定")
        window.transient(self.winfo_toplevel())
        window.geometry("1100x640")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)

        source_frame = ttk.LabelFrame(window, text="來源與保存資訊", padding=6)
        source_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        source_frame.columnconfigure(0, weight=1)
        source_frame.rowconfigure(0, weight=1)
        source_text = tk.Text(
            source_frame, height=6, wrap="none", font=MONO_FONT)
        source_y = ttk.Scrollbar(source_frame, orient="vertical", command=source_text.yview)
        source_x = ttk.Scrollbar(source_frame, orient="horizontal", command=source_text.xview)
        source_text.configure(yscrollcommand=source_y.set, xscrollcommand=source_x.set)
        source_text.grid(row=0, column=0, sticky="nsew")
        source_y.grid(row=0, column=1, sticky="ns")
        source_x.grid(row=1, column=0, sticky="ew")
        for index, column in enumerate(columns, start=1):
            saved_name = column["saved_name"]
            if saved_name is None or saved_name == "":
                saved_name_text = "未知（未保存）"
            elif isinstance(saved_name, str):
                saved_name_text = saved_name
            else:
                saved_name_text = json.dumps(saved_name, ensure_ascii=False)
            saved_method = column["saved_method"]
            if saved_method is None or saved_method == "":
                saved_method_text = "未知（未保存）"
            elif isinstance(saved_method, str):
                saved_method_text = saved_method
            else:
                saved_method_text = json.dumps(saved_method, ensure_ascii=False)
            source_path = column["source_path"]
            if source_path is None:
                source_path_text = "未知（沒有 run_dir/config.json）"
            else:
                source_path_text = str(source_path)
            if column["metadata_only"]:
                result_path_text = "尚未生成"
            else:
                result_path_text = column["result_path"]
            comparison_context = column.get("comparison_context")
            if comparison_context:
                comparison_context_text = comparison_context
            else:
                comparison_context_text = "未知（非 comparison run）"
            started_at = column.get("started_at")
            if not isinstance(started_at, str) or not started_at.strip():
                started_at_text = "未知（未保存）"
            else:
                started_at_text = started_at
            finished_at = column.get("finished_at")
            if not isinstance(finished_at, str) or not finished_at.strip():
                finished_at_text = "未知（未保存）"
            else:
                finished_at_text = finished_at
            status_path = column.get("status_path")
            if status_path is None:
                status_path_text = "未知（status.json 未保存或無法讀取）"
            else:
                status_path_text = str(status_path)
            source_text.insert(
                "end",
                f"#{index} 保存名稱：{saved_name_text}｜保存方法：{saved_method_text}\n"
                f"清單項目：{column['result_name']}｜{column['result_method']}\n"
                f"比較組別：{comparison_context_text}\n"
                f"開始時間：{started_at_text}｜結束時間：{finished_at_text}\n"
                f"狀態來源：{status_path_text}\n"
                f"結果 CSV：{result_path_text}\n"
                f"設定來源：{source_path_text}\n",
            )
            if not column["valid"]:
                source_text.insert("end", f"狀態：未知（{column['unknown_reason']}）\n")
            source_text.insert("end", "\n")
        source_text.configure(state="disabled")

        controls = ttk.Frame(window, padding=(8, 4))
        controls.grid(row=1, column=0, sticky="ew")
        search_var = tk.StringVar(master=window, value="")
        only_differences_var = tk.BooleanVar(master=window, value=False)
        difference_count_var = tk.StringVar(master=window, value="")
        note_var = tk.StringVar(master=window, value="")
        ttk.Label(controls, text="搜尋欄位／保存值：").pack(side="left")
        search_entry = ttk.Entry(controls, textvariable=search_var, width=24)
        search_entry.pack(side="left", fill="x", expand=True, padx=(4, 3))
        def clear_search(_event=None):
            search_var.set("")
            if _event is not None:
                return "break"
            return None

        ttk.Button(controls, text="清除", command=clear_search).pack(
            side="left", padx=(0, 8))

        table_frame = ttk.LabelFrame(
            window, text="保存設定（只讀保存值；不以現行預設補值）",
            padding=(8, 2, 8, 4))
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        table_columns = ["label", "path"]
        for index in range(len(columns)):
            table_columns.append(f"result_{index}")
        settings_tree = ttk.Treeview(
            table_frame, columns=table_columns, show="headings", height=16)
        settings_tree.heading("label", text="欄位名稱")
        settings_tree.column("label", width=190, minwidth=130, anchor="w")
        settings_tree.heading("path", text="完整欄位路徑")
        settings_tree.column("path", width=250, minwidth=175, anchor="w")
        for index, column in enumerate(columns):
            column_name = f"result_{index}"
            saved_name = column["saved_name"]
            if isinstance(saved_name, str) and saved_name:
                heading_name = saved_name
            else:
                heading_name = f"結果 {index + 1}"
            saved_method = column["saved_method"]
            if isinstance(saved_method, str) and saved_method:
                heading_text = f"{heading_name} / {saved_method}"
            else:
                heading_text = heading_name
            context = column.get("comparison_context")
            if context:
                heading_text = f"{heading_text} · {context}"
            settings_tree.heading(column_name, text=heading_text)
            settings_tree.column(column_name, width=260, minwidth=150, anchor="w")
        settings_y = ttk.Scrollbar(
            table_frame, orient="vertical", command=settings_tree.yview)
        settings_x = ttk.Scrollbar(
            table_frame, orient="horizontal", command=settings_tree.xview)
        settings_tree.configure(yscrollcommand=settings_y.set, xscrollcommand=settings_x.set)
        settings_tree.grid(row=0, column=0, sticky="nsew")
        settings_y.grid(row=0, column=1, sticky="ns")
        settings_x.grid(row=1, column=0, sticky="ew")

        visible_field_paths = {}

        def render_settings_rows(*_args):
            difference_fields = set(_saved_settings_difference_fields(columns, fields))
            show_only_differences = only_differences_var.get()
            query = search_var.get().strip().casefold()
            visible_field_paths.clear()
            for item in settings_tree.get_children():
                settings_tree.delete(item)
            for index, field_path in enumerate(fields):
                if show_only_differences and field_path not in difference_fields:
                    continue
                label = field_label(tuple(field_path.split(".")))
                raw_values = [
                    _saved_setting_text(column, field_path) for column in columns
                ]
                display_values = [
                    _saved_setting_display_text(column, field_path)
                    for column in columns
                ]
                searchable_text = " ".join(
                    [label, field_path, *raw_values, *display_values]
                ).casefold()
                if query and query not in searchable_text:
                    continue

                iid = f"setting_{index}"
                row_values = [label, field_path, *display_values]
                settings_tree.insert("", "end", iid=iid, values=row_values)
                visible_field_paths[iid] = field_path
            difference_count_var.set(
                f"顯示 {len(visible_field_paths)} / 差異 {len(difference_fields)} / 共 {len(fields)} 欄")
            if not fields:
                note_var.set("config.json 中沒有可顯示的設定欄位。")
            elif show_only_differences and not difference_fields:
                note_var.set("所選保存設定沒有差異。")
            elif query and not visible_field_paths:
                note_var.set("沒有符合目前搜尋與差異條件的保存欄位。")
            else:
                note_var.set("")

        differences_button = ttk.Checkbutton(
            controls, text="只顯示差異", variable=only_differences_var,
            command=render_settings_rows)
        differences_button.pack(side="left")
        if len(columns) < 2:
            differences_button.configure(state="disabled")
        ttk.Label(controls, textvariable=difference_count_var).pack(side="left", padx=12)
        search_trace_id = search_var.trace_add("write", render_settings_rows)

        def remove_search_trace():
            nonlocal search_trace_id
            if search_trace_id is None:
                return
            try:
                search_var.trace_remove("write", search_trace_id)
            except tk.TclError:
                pass
            search_trace_id = None

        def close_window():
            remove_search_trace()
            window.destroy()

        def on_window_destroy(event):
            if event.widget is window:
                remove_search_trace()

        window.protocol("WM_DELETE_WINDOW", close_window)
        window.bind("<Destroy>", on_window_destroy, add="+")
        search_entry.bind("<Escape>", clear_search)

        # 複製目前顯示的列，差異模式不會混入已隱藏的欄位。
        def copy_settings_rows():
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            headers = ["完整欄位路徑"]
            for index, column in enumerate(columns, start=1):
                heading = (f"#{index} {column['result_display_name']} / "
                           f"{column['result_method']}")
                headers.append(heading)
            writer.writerow(headers)
            for item in settings_tree.get_children():
                field_path = visible_field_paths[item]
                raw_values = [
                    _saved_setting_text(column, field_path) for column in columns
                ]
                writer.writerow([field_path, *raw_values])
            try:
                window.clipboard_clear()
                window.clipboard_append(output.getvalue())
            except tk.TclError as error:
                note_var.set(f"無法複製表格：{error}")
                return
            note_var.set(f"已複製 {len(visible_field_paths)} 個顯示欄位。")

        copy_button = ttk.Button(controls, text="複製顯示表格", command=copy_settings_rows)
        copy_button.pack(side="right")
        ttk.Label(window, textvariable=note_var, anchor="w", padding=(10, 2)).grid(
            row=3, column=0, sticky="ew")
        ttk.Button(window, text="關閉", command=close_window).grid(
            row=4, column=0, sticky="e", padx=8, pady=(2, 8))
        render_settings_rows()
        return window

    def export_summary(self):
        # 背景重整可能在存檔對話框期間替換清單，匯出按下按鈕當時的資料。
        summary_results = list(self.results)
        # 對話框前後的來源都需保護，避免重新整理移除原有紀錄後誤覆寫。
        protected_sources = _source_csv_identities(self.results)
        selected_path = filedialog.asksaveasfilename(
            parent=self, title="匯出結果摘要", defaultextension=".csv",
            filetypes=(("CSV 檔案", "*.csv"), ("所有檔案", "*.*")),
            initialfile="experiment-results-summary.csv")
        if not selected_path:
            return
        protected_sources.update(_source_csv_identities(self.results))
        if _path_identity(selected_path) in protected_sources:
            self.status_var.set("目標是已載入的來源 CSV，請選擇新的匯出檔。")
            return
        try:
            headers = (
                "experiment", "method", "status", "last_iteration", "iteration_unit",
                "latest_training_reward", "latest_eval_aoi", "result_csv",
            )
            # CSV 保存讀入的數值；六位有效數字只用於畫面顯示。
            # csv.writer 將 None 寫成空欄，缺資料不會被混成文字或零。
            rows = []
            for result in summary_results:
                if result.get("metadata_only", False):
                    result_csv = ""
                else:
                    result_csv = str(result["path"])
                rows.append((
                    _result_display_name(result), result["method"], result["status"],
                    result["last_iteration"], result["last_unit"],
                    _latest_value(result, "reward"),
                    _latest_value(result, "eval_aoi"),
                    result_csv,
                ))
            saving.write_csv(selected_path, headers, rows, encoding="utf-8")
        except OSError:
            self.status_var.set("無法寫入摘要 CSV。")
            return
        self.status_var.set(f"摘要已匯出至 {selected_path}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        if self._auto_refresh_after_id is not None:
            try:
                self.after_cancel(self._auto_refresh_after_id)
            except tk.TclError:
                pass
            self._auto_refresh_after_id = None
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        if self._configure_redraw_after_id is not None:
            try:
                self.after_cancel(self._configure_redraw_after_id)
            except tk.TclError:
                pass
            self._configure_redraw_after_id = None

    # 面板被單獨移除時，也取消自己的輪詢，避免留下已銷毀元件的回呼。
    def _on_destroy(self, event):
        if event.widget is self:
            self.close()
