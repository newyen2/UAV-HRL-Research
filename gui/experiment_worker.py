"""Isolated child process that applies one experiment config and runs its trainer."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from datetime import datetime
from pathlib import Path

from runtime import fileIO as saving

from runtime.configuration import METHODS, validate_experiment
from runtime import record as report


PROGRESS_PREFIX = "@@EXPERIMENT "
# 以暫存檔替換方式保存實驗狀態。
def _write_status(path: Path, status: dict):
    saving.write_json(path, status, safe=True, newline=True, atomic=True)


# 訓練模組匯入前套用設定，指定本場獨立結果目錄。
def apply_settings(method: str, settings: dict, utils_module, output_dir) -> Path:
    """Validate and apply settings before creating this run's result directory."""
    output_dir = Path(output_dir).resolve()
    results_dir = output_dir / "results"
    # utils.apply_settings validates the entire snapshot before changing a namespace.
    utils_module.apply_settings(method, settings)
    saving.make_directory(results_dir, parents=True, exist_ok=True)
    # Each run owns its result tree, even if the config omitted RECORD fields.
    utils_module.RECORD_SETTING.PATH = results_dir
    return results_dir


# 載入指定方法的 train 入口。
def _load_trainer(project_root: Path, method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    details = METHODS[method]
    module_path = project_root / details["module"]
    module_name = f"_experiment_method_{method.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        trainer = getattr(module, "train", None)
        if not callable(trainer):
            raise TypeError(f"Training entry is not callable: {module_path}")
        return trainer
    except Exception:
        # 匯入或入口驗證失敗時，移除尚未成功載入的模組。
        sys.modules.pop(module_name, None)
        raise


# 將進度事件轉成帶固定前綴的 JSON 日誌。
def _progress_callback(event):
    progress_module = importlib.import_module("runtime.progress")
    progress_module.forward_event(event, prefix=PROGRESS_PREFIX)


# 驗證並套用單場設定，執行訓練與保存完成或失敗狀態。
def run_experiment(config: dict, output_dir, *, project_root=None, utils_module=None):
    """Apply settings, install progress forwarding, load the method, and run it."""
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    status_path = output_dir / "status.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    original_sys_path = sys.path[:]
    progress_module = None
    base_status = {"output_dir": str(output_dir), "started_at": started_at}
    try:
        config = validate_experiment(config)
        base_status.update(name=config["name"], method=config["method"])
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if utils_module is None:
            utils_module = importlib.import_module("utils")
        results_dir = apply_settings(config["method"], config["settings"],
                                     utils_module, output_dir)
        progress_module = importlib.import_module("runtime.progress")
        progress_module.set_callback(_progress_callback)
        trainer = _load_trainer(project_root, config["method"])
        trainer()
        _write_status(status_path, {**base_status, "status": "completed",
                                    "results_dir": str(results_dir),
                                    "finished_at": datetime.now().isoformat(timespec="seconds")})
        return 0
    except BaseException as error:
        details = traceback.format_exc()
        try:
            saving.make_directory(output_dir, parents=True, exist_ok=True)
            _write_status(status_path, {**base_status, "status": "failed",
                                        "error": f"{type(error).__name__}: {error}",
                                        "traceback": details,
                                        "finished_at": datetime.now().isoformat(timespec="seconds")})
        except Exception:
            pass
        report.exception()
        return 1
    finally:
        if progress_module is not None:
            try:
                progress_module.set_callback(None)
            except Exception:
                pass
        sys.path[:] = original_sys_path


# 讀取命令列指定的實驗 JSON 並執行單場訓練。
def main(argv=None):
    import utils
    args = utils.parse_args("experiment_worker", argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        config = saving.read_json(args.config, encoding="utf-8")
    except Exception as error:
        saving.make_directory(args.output_dir, parents=True, exist_ok=True)
        _write_status(args.output_dir / "status.json", {
            "status": "failed", "error": f"Could not read config: {error}",
            "finished_at": datetime.now().isoformat(timespec="seconds")})
        report.exception()
        return 1
    return run_experiment(config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
