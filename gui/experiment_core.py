"""Parameter batches and subprocess queues, with shared configuration exports."""
from __future__ import annotations

import copy
import itertools
import json
import math
import queue
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from runtime import fileIO as saving
from runtime.configuration import (
    METHODS, RETIRED_METHODS, _COMMON_SETTINGS, _METHOD_SETTING_GROUPS,
    default_settings, inactive_setting_paths, setting_options,
    training_device_options, validate_experiment, validate_settings,
)


# 由同一份設定產生單欄參數實驗，全部有效後才交給介面加入佇列。
def expand_parameter_sweep(spec: Mapping, field_path: str, values: list) -> list[dict]:
    if not isinstance(field_path, str) or field_path.count(".") != 1:
        raise ValueError("參數欄位須使用群組與欄位名稱，例如 TRAIN.seed")
    if not isinstance(values, list) or not values:
        raise ValueError("參數值須為非空 JSON 清單，例如 [0, 1, 2]")
    if len(values) > 100:
        raise ValueError("一次最多產生 100 個實驗")

    baseline = validate_experiment(spec)
    group, field = field_path.split(".")
    if group not in baseline["settings"] or field not in baseline["settings"][group]:
        raise ValueError(f"設定欄位不存在：{field_path}")

    experiments = []
    for index, value in enumerate(values, start=1):
        # 每次從相同基準複製，避免上一個實驗的覆寫影響下一個。
        candidate = copy.deepcopy(baseline)
        candidate["settings"][group][field] = copy.deepcopy(value)
        try:
            display_value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if len(display_value) > 44:
                display_value = display_value[:41] + "…"
            suffix = f" [{index}: {field_path}={display_value}]"
            # 保留參數標記，較長的基準名稱截短至既有名稱上限。
            candidate["name"] = baseline["name"][:120 - len(suffix)].rstrip() + suffix
            inactive = inactive_setting_paths(candidate["method"], candidate["settings"])
            if (group, field) in inactive:
                raise ValueError(f"設定 {field_path} 在此配置沒有作用：{inactive[(group, field)]}")
            experiments.append(validate_experiment(candidate))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"第 {index} 個值（{field_path}）：{error}") from error
    return experiments


# 先套用每組參數，再驗證完整組合並回傳獨立實驗設定。
def expand_parameter_grid(spec: Mapping, axes: Mapping) -> list[dict]:
    baseline = validate_experiment(spec)
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("參數軸須為非空設定對照表")

    axis_items = []
    combination_count = 1
    for field_path, values in axes.items():
        if not isinstance(field_path, str) or field_path.count(".") != 1:
            raise ValueError("參數欄位須使用群組與欄位名稱，例如 TRAIN.seed")
        if not isinstance(values, list) or not values:
            raise ValueError(f"參數值須為非空 JSON 清單：{field_path}")
        if len(values) > 100:
            raise ValueError("一次最多產生 100 個實驗")

        group, field = field_path.split(".")
        if group not in baseline["settings"] or field not in baseline["settings"][group]:
            raise ValueError(f"設定欄位不存在：{field_path}")

        axis_items.append((field_path, group, field, values))
        combination_count *= len(values)
        if combination_count > 100:
            raise ValueError("一次最多產生 100 個實驗")

    # 用最大序號預留固定名稱空間，確保摘要截短後名稱仍唯一。
    summary_limit = 60
    longest_marker = f" [{combination_count}: "
    base_name = baseline["name"][:120 - len(longest_marker) - summary_limit - 1].rstrip()
    experiments = []
    value_axes = [item[3] for item in axis_items]
    for index, combination in enumerate(itertools.product(*value_axes), start=1):
        candidate = copy.deepcopy(baseline)
        try:
            summary_parts = []
            for (field_path, group, field, _), value in zip(axis_items, combination):
                candidate["settings"][group][field] = copy.deepcopy(value)
                display_value = json.dumps(value, ensure_ascii=False, allow_nan=False)
                summary_parts.append(f"{field_path}={display_value}")

            # 序號固定放在摘要前；長摘要只截短顯示文字。
            summary = ", ".join(summary_parts)
            if len(summary) > summary_limit:
                summary = summary[:summary_limit - 1] + "…"
            candidate["name"] = f"{base_name} [{index}: {summary}]"
            inactive = inactive_setting_paths(candidate["method"], candidate["settings"])
            for field_path, group, field, _ in axis_items:
                if (group, field) in inactive:
                    raise ValueError(f"設定 {field_path} 在此配置沒有作用：{inactive[(group, field)]}")
            experiments.append(validate_experiment(candidate))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"第 {index} 個組合：{error}") from error
    return experiments


# 將實驗名稱轉成可用於資料夾名稱的短字串。
def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:48] or "experiment"


# 將溢位的 JSON 浮點數轉成空值，避免進度事件含有無限值。
def _progress_float(value: str) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


class QueueRunner:
    """Run validated experiment specs sequentially in isolated Python children."""

    # 初始化輸出路徑、事件佇列與子程序管理狀態。
    def __init__(self, output_root, worker_path=None, *, popen_factory=None,
                 terminate_timeout: float = 2.0):
        self.output_root = Path(output_root).expanduser().resolve()
        if worker_path is not None:
            self.worker_path = Path(worker_path).resolve()
        else:
            self.worker_path = None
        self._popen_factory = popen_factory or subprocess.Popen
        self._terminate_timeout = max(0.0, float(terminate_timeout))
        self.events: queue.Queue[dict] = queue.Queue()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._watchdog_processes: set[int] = set()
        self._stop_requested = threading.Event()
        self._running = False
        self._specs: list[dict] | None = None
        self._accepting = False
        self._initial_spec_count: int | None = None

    # 回傳實驗佇列目前是否仍在執行。
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # 驗證並保存佇列快照，在背景依序啟動實驗。
    def start(self, specs: Sequence[Mapping]):
        snapshot = self._validated_specs(specs)
        with self._lock:
            if self._running:
                raise RuntimeError("experiment queue is already running")
        saving.make_directory(self.output_root, parents=True, exist_ok=True)
        with self._lock:
            if self._running:
                raise RuntimeError("experiment queue is already running")
            self._stop_requested = threading.Event()
            thread = threading.Thread(target=self._run_queue, args=(snapshot,),
                                      name="experiment-queue", daemon=True)
            self._specs = snapshot
            self._initial_spec_count = len(snapshot)
            self._accepting = True
            self._thread = thread
            self._running = True
            try:
                thread.start()
            except Exception:
                # 背景執行緒尚未啟動，釋放執行中狀態，讓使用者可重試。
                self._thread = None
                self._specs = None
                self._initial_spec_count = None
                self._accepting = False
                self._running = False
                raise
        return thread

    # 驗證整批實驗並建立與呼叫端分離的快照。
    @staticmethod
    def _validated_specs(specs: Sequence[Mapping]) -> list[dict]:
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
            raise ValueError("specs must be a sequence of experiment objects")
        source = list(specs)
        return [validate_experiment(spec) for spec in source]

    # 將整批新實驗加入目前仍接收工作的批次。
    def append(self, specs: Sequence[Mapping]) -> bool:
        snapshot = self._validated_specs(specs)
        with self._lock:
            if (not self._running or not self._accepting
                    or self._stop_requested.is_set() or self._specs is None):
                return False
            self._specs.extend(snapshot)
            return True

    # 請求停止目前子程序並取消後續實驗。
    def stop(self):
        """Request cancellation and terminate the current child, if any."""
        with self._lock:
            if not self._running:
                return
            self._stop_requested.set()
            self._accepting = False
            process = self._process
        if process is not None and process.poll() is None:
            self._ensure_watchdog(process)

    # 為指定子程序啟動一次背景終止監督。
    def _ensure_watchdog(self, process):
        identity = id(process)
        with self._lock:
            if identity in self._watchdog_processes:
                return
            self._watchdog_processes.add(identity)
        try:
            watchdog = threading.Thread(
                target=self._watchdog, args=(process, identity),
                name="experiment-cancel-watchdog", daemon=True)
            watchdog.start()
        except Exception as error:
            with self._lock:
                self._watchdog_processes.discard(identity)
            self.events.put({
                "type": "log",
                "text": "Cancellation watchdog failed to start: "
                        f"{type(error).__name__}: {error}",
            })
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    # The worker may exit between poll() and kill().
                    pass

    # 終止並回收子程序，完成後清除監督標記。
    def _watchdog(self, process, identity):
        try:
            self._terminate_and_reap(process)
        finally:
            with self._lock:
                self._watchdog_processes.discard(identity)

    # 先正常終止子程序，逾時則強制停止並回收。
    def _terminate_and_reap(self, process):
        if process.poll() is not None:
            process.wait()
            return
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=self._terminate_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except (OSError, ProcessLookupError):
            pass

    # 為單場實驗建立不重複的輸出資料夾。
    def _new_run_dir(self, index: int, spec: dict) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        while True:
            target = self.output_root / f"{index + 1:03d}-{_slug(spec['name'])}-{stamp}-{uuid.uuid4().hex[:8]}"
            try:
                saving.make_directory(target, parents=True, exist_ok=False)
                return target
            except FileExistsError:
                continue

    # 先寫入暫存檔，再替換 JSON 檔案。
    @staticmethod
    def _write_json(path: Path, value: dict):
        saving.write_json(path, value, safe=True, newline=True, atomic=True)

    # 依序執行實驗，保存狀態並回報佇列完成事件。
    def _run_queue(self, specs: list[dict]):
        cancelled = False
        queue_error = None
        index = -1
        next_index = 0
        try:
            while True:
                next_item = self._next_spec(specs, next_index)
                if next_item is None:
                    break
                index, spec = next_item
                next_index = index + 1
                # 先清空路徑，避免建立失敗時沿用前一場的目錄。
                run_dir = None
                try:
                    run_dir = self._new_run_dir(index, spec)
                    saving.make_directory(run_dir / "results")
                    saving.touch_file(run_dir / "training.log")
                    saving.touch_file(run_dir / "progress.jsonl")
                    self._write_json(run_dir / "config.json", spec)
                    self._write_json(run_dir / "status.json", {
                        "status": "queued", "name": spec["name"], "method": spec["method"]})
                    if self._stop_requested.is_set():
                        cancelled = True
                        self._finish_cancelled(index, spec, run_dir)
                        continue

                    self._write_json(run_dir / "status.json", {
                        "status": "running", "name": spec["name"], "method": spec["method"],
                        "started_at": datetime.now().isoformat(timespec="seconds")})
                    self.events.put({"type": "started", "index": index, "name": spec["name"],
                                     "output_dir": str(run_dir)})
                    status = self._run_one(index, spec, run_dir)
                    cancelled = cancelled or status == "cancelled"
                    worker_status = {}
                    try:
                        worker_status = saving.read_json(run_dir / "status.json", encoding="utf-8")
                    except (OSError, ValueError):
                        pass
                    self._write_json(run_dir / "status.json", {**worker_status,
                        "status": status, "name": spec["name"], "method": spec["method"],
                        "finished_at": datetime.now().isoformat(timespec="seconds")})
                    self.events.put({"type": "finished", "index": index, "status": status,
                                     "output_dir": str(run_dir)})
                except Exception as error:
                    queue_error = f"{type(error).__name__}: {error}"
                    if run_dir is not None:
                        failure_status = {}
                        try:
                            existing_status = saving.read_json(
                                run_dir / "status.json", encoding="utf-8")
                            if isinstance(existing_status, dict):
                                failure_status = existing_status
                        except (OSError, ValueError, TypeError):
                            pass
                        failure_status.update({
                            "status": "failed",
                            "name": spec["name"],
                            "method": spec["method"],
                            "error": queue_error,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                        })
                        try:
                            self._write_json(run_dir / "status.json", failure_status)
                        except Exception as status_error:
                            queue_error += (
                                "; failed to save failed status: "
                                f"{type(status_error).__name__}: {status_error}"
                            )

                    self.events.put({"type": "log", "index": index,
                                     "text": f"QueueRunner error: {queue_error}"})
                    if run_dir is None:
                        failed_output_dir = None
                    else:
                        failed_output_dir = str(run_dir)
                    self.events.put({"type": "finished", "index": index,
                                     "name": spec["name"], "status": "failed",
                                     "output_dir": failed_output_dir})
                    break
        except Exception as error:
            queue_error = f"{type(error).__name__}: {error}"
            self.events.put({"type": "log", "index": max(index, 0),
                             "text": f"QueueRunner error: {queue_error}"})
        finally:
            with self._lock:
                self._process = None
                self._accepting = False
                skipped_appends = []
                if (queue_error is not None and self._specs is specs
                        and self._initial_spec_count is not None):
                    first_skipped = max(next_index, self._initial_spec_count)
                    skipped_appends = list(enumerate(
                        specs[first_skipped:], start=first_skipped))
                    for skipped_index, skipped_spec in skipped_appends:
                        self.events.put({"type": "finished", "index": skipped_index,
                                         "name": skipped_spec["name"],
                                         "status": "cancelled", "output_dir": None})
                stopped = (cancelled or bool(skipped_appends)
                           or self._stop_requested.is_set())
                self.events.put({"type": "queue_done", "cancelled": stopped, "stopped": stopped,
                                 "error": queue_error})
                self._running = False
                self._specs = None
                self._initial_spec_count = None

    # 在接收鎖內取出下一筆；抵達批次尾端時同時關閉 append。
    def _next_spec(self, specs: list[dict], index: int):
        with self._lock:
            if index < len(specs):
                return index, specs[index]
            self._accepting = False
            return None

    # 保存取消狀態並送出該場實驗的結束事件。
    def _finish_cancelled(self, index: int, spec: dict, run_dir: Path):
        self._write_json(run_dir / "status.json", {
            "status": "cancelled", "name": spec["name"], "method": spec["method"],
            "finished_at": datetime.now().isoformat(timespec="seconds")})
        self.events.put({"type": "finished", "index": index, "status": "cancelled",
                         "output_dir": str(run_dir)})

    # 啟動單場訓練子程序，保存日誌並轉送進度事件。
    def _run_one(self, index: int, spec: dict, run_dir: Path) -> str:
        progress_path = run_dir / "progress.jsonl"
        log_path = run_dir / "training.log"
        command = [sys.executable, "-X", "utf8", "-B", "-u"]
        if self.worker_path is None:
            command.extend(["-m", "gui.experiment_worker"])
        else:
            command.append(str(self.worker_path))
        command.extend(["--config", str(run_dir / "config.json"),
                        "--output-dir", str(run_dir)])
        process = None
        waited = False
        try:
            process = self._popen_factory(command, cwd=str(Path(__file__).resolve().parent.parent),
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace", bufsize=1)
            with self._lock:
                self._process = process
                should_stop = self._stop_requested.is_set()
            if should_stop and process.poll() is None:
                self._ensure_watchdog(process)
            with saving.open_log(log_path, errors="replace") as log_file, \
                    saving.open_log(progress_path, errors="strict") as progress_file:
                assert process.stdout is not None
                for line in iter(process.stdout.readline, ""):
                    saving.write_log(log_file, line)
                    if line.startswith("@@EXPERIMENT "):
                        raw = line[len("@@EXPERIMENT "):].strip()
                        try:
                            event = json.loads(raw, parse_float=_progress_float,
                                               parse_constant=lambda _value: None)
                        except (ValueError, TypeError):
                            event = None
                        if isinstance(event, dict):
                            event["type"] = "progress"
                            event["index"] = index
                            event["name"] = spec["name"]
                            event["output_dir"] = str(run_dir)
                            saving.write_json_line(progress_file, event, safe=True)
                            self.events.put(event)
                            continue
                    self.events.put({"type": "log", "index": index, "text": line.rstrip("\r\n"),
                                     "output_dir": str(run_dir)})
            try:
                # 取消時只等終止寬限時間；正常訓練則等待子程序自然結束。
                if self._stop_requested.is_set():
                    return_code = process.wait(timeout=self._terminate_timeout)
                else:
                    return_code = process.wait(timeout=None)
                waited = True
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
                waited = True
            if return_code == 0:
                return "completed"
            if self._stop_requested.is_set():
                return "cancelled"
            return "failed"
        except Exception as error:
            with saving.open_log(log_path, append=True, errors="strict") as log_file:
                saving.write_log(log_file, f"QueueRunner error: {error}\n")
            self.events.put({"type": "log", "index": index,
                             "text": f"QueueRunner error: {error}",
                             "output_dir": str(run_dir)})
            if self._stop_requested.is_set():
                return "cancelled"
            return "failed"
        finally:
            if process is not None:
                try:
                    if process.poll() is None and not waited:
                        self._ensure_watchdog(process)
                    process.wait()
                finally:
                    if process.stdout is not None:
                        try:
                            process.stdout.close()
                        except OSError:
                            pass
                    with self._lock:
                        if self._process is process:
                            self._process = None
