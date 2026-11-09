"""訓練進度狀態、事件與子程序傳輸。"""
from contextlib import contextmanager
import json
import math
import time


_callback = None
_state = {"stage": "initializing", "episode": 0, "phase": "train"}
_last_step_time = 0.0
_HRL_METHODS = ("HRL", "HRL_PPO")


# 設定接收函式並重設階段與節流狀態。
def set_callback(callback):
    global _callback, _last_step_time
    _callback = callback
    _last_step_time = 0.0
    _state.update(stage="initializing", episode=0, phase="train")


# 將目前階段與額外欄位組成進度事件。
def emit(event_type, **fields):
    if _callback is not None:
        _callback({"type": "progress", "event": event_type, **_state, **fields})


# 切換目前階段並送出階段事件。
def set_stage(stage, *, phase=None):
    _state["stage"] = stage
    # 階段入口同步 phase，讓回合外的初始化也顯示正確階段。
    if phase is not None:
        _state["phase"] = phase
    emit("stage")


# 暫時切換階段，離開區塊時恢復原階段。
@contextmanager
def stage(name):
    previous = _state["stage"]
    set_stage(name)
    try:
        yield
    finally:
        set_stage(previous)


# 記錄回合與訓練階段，送出回合開始事件。
def begin_episode(episode, phase="train"):
    _state.update(episode=int(episode), phase=phase, stage="training")
    emit("episode_start", step=0)


# 回報步數與指標，保留起終步並節流中間事件。
def step(step, total_steps, reward=None, mean_aoi=None):
    global _last_step_time
    if _callback is None:
        return
    now = time.monotonic()
    # 限制管線與介面的事件量，同時保留第一步與最後一步。
    if step not in (1, total_steps) and now - _last_step_time < 0.2:
        return
    _last_step_time = now
    fields = {"step": int(step), "total_steps": int(total_steps)}
    if reward is not None:
        fields["reward"] = float(reward)
    if mean_aoi is not None:
        fields["mean_aoi"] = float(mean_aoi)
    emit("step", **fields)


# 送出回合結束事件及彙整指標。
def end_episode(**metrics):
    emit("episode_end", **metrics)


# 將巢狀資料中的非有限浮點數轉成 JSON 空值。
def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


# 將進度事件轉成帶固定前綴的 JSON 日誌。
def forward_event(event, *, prefix):
    event = dict(event)
    event["type"] = "progress"
    event.setdefault("event", "update")
    print(prefix + json.dumps(_safe_json(event), ensure_ascii=False,
                              allow_nan=False), flush=True)


class Progress:
    """只讀取方法的執行狀態，向 GUI 發送進度。"""

    def __init__(self, method, training):
        if method not in ("DQN", "DQN_service", *_HRL_METHODS, "MA-CQR"):
            raise ValueError(f"Unsupported progress method: {method!r}")
        self.method = method
        self.training = training

    def set_stage(self, stage):
        if stage == "navigation_pretrain":
            # 預訓練在回合外開始，需同步目前的訓練階段。
            set_stage(stage, phase=self.training.phase)
        else:
            set_stage(stage)

    def episode_start(self):
        begin_episode(self.training.episode, self.training.phase)

    def step(self):
        training = self.training
        if self.method in _HRL_METHODS:
            reward = training.environment_return
        else:
            reward = training.episode_reward
        step(training.env.steps, training.env.max_steps, reward,
             training.aoi_sum / training.env.steps)

    def episode_end(self):
        training = self.training
        if self.method in _HRL_METHODS:
            end_episode(reward=training.environment_return,
                        learning_reward=training.upper_episode_reward,
                        mean_aoi=training.aoi_sum / training.env.steps)
        elif self.method == "MA-CQR":
            import numpy as np
            end_episode(reward=training.episode_reward,
                        mean_aoi=training.aoi_sum / training.env.steps,
                        loss=float(np.mean(training.losses)))
        else:
            end_episode(reward=training.episode_reward,
                        mean_aoi=training.aoi_sum / training.env.steps)
