"""Experience replay buffers used by the project agents.

Each buffer documents and preserves its own sample tuple schema. In particular,
duration and option-completion fields describe hierarchical transitions; they
are not interchangeable with an environment ``done`` flag.
"""

from collections import namedtuple
from collections.abc import Sequence
from operator import index as integer_index
import random

import numpy as np
import torch
from utils import TRAIN_SETTING, HRL_SETTING


class _ReplayMemory(Sequence):
    """Fixed-capacity FIFO storage with constant-time logical indexing."""

    __slots__ = ("_items", "_maxlen", "_start")

    def __init__(self, maxlen):
        self._items = []
        self._maxlen = maxlen
        self._start = 0

    @property
    def maxlen(self):
        return self._maxlen

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        index = integer_index(index)
        size = len(self._items)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("deque index out of range")
        return self._items[(self._start + index) % self._maxlen]

    def __iter__(self):
        size = len(self._items)
        for index in range(size):
            yield self._items[(self._start + index) % self._maxlen]

    def append(self, value):
        if len(self._items) < self._maxlen:
            self._items.append(value)
        else:
            self._items[self._start] = value
            self._start = (self._start + 1) % self._maxlen

    def clear(self):
        self._items.clear()
        self._start = 0


class ReplayBuffer:
    """一般離散動作 replay：``(state, action, reward, next_state)``。

    state/next_state/reward 取樣為 float32，action 取樣為 int64；action
    是離散動作索引。此 schema 沒有環境 done 標記。
    """

    # 建立固定容量的經驗池。
    def __init__(self):
        self.memory = _ReplayMemory(maxlen=TRAIN_SETTING.buffer_size)
        self.batch_size = TRAIN_SETTING.batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state"]
        )

    # 保存一筆單步狀態轉移。
    def add(self, state, action, reward, next_state):
        self.memory.append(self.experience(state, action, reward, next_state))

    # 隨機抽取一批經驗，轉成指定裝置上的 tensor。
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # 狀態保留向量欄位；單一動作與獎勵堆成 (batch, 1)。
        states = torch.as_tensor(
            np.stack([e.state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        actions = torch.as_tensor(
            np.vstack([e.action for e in experiences]),
            dtype=torch.long, device=TRAIN_SETTING.device,
        )
        rewards = torch.as_tensor(
            np.vstack([e.reward for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_states = torch.as_tensor(
            np.stack([e.next_state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        return states, actions, rewards, next_states

    # 回傳目前保存的經驗數量。
    def __len__(self):
        return len(self.memory)


class DurationReplayBuffer(ReplayBuffer):
    """跨步離散動作 replay：``(state, action, reward, next_state, duration)``。

    action 是 int64 離散索引；其餘取樣欄位為 float32。duration 是此決策
    持續的步數，用於 ``gamma ** duration``，不代表環境 done。
    """

    # 建立包含決策持續步數的經驗池。
    def __init__(self):
        super().__init__()
        self.experience = namedtuple(
            "DurationExperience", ["state", "action", "reward", "next_state", "duration"]
        )

    # 保存一筆跨步決策經驗及持續步數。
    def add(self, state, action, reward, next_state, duration):
        self.memory.append(self.experience(state, action, reward, next_state, duration))

    # 抽取跨步經驗，連同持續步數回傳。
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.as_tensor(
            np.stack([e.state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        actions = torch.as_tensor(
            np.vstack([e.action for e in experiences]),
            dtype=torch.long, device=TRAIN_SETTING.device,
        )
        rewards = torch.as_tensor(
            np.vstack([e.reward for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_states = torch.as_tensor(
            np.stack([e.next_state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        durations = torch.as_tensor(
            np.vstack([e.duration for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        return states, actions, rewards, next_states, durations


class ServiceReplayBuffer(ReplayBuffer):
    """服務決策 replay：``(state, target, action, reward, next_state, next_forced_service)``。

    target 是鎖定裝置的 int64 索引，action 是實際執行的 int64 服務選擇（含強制服務）；其他欄位
    取樣為 float32。next_forced_service 表示下一步是否因移動後距離而被迫服務。
    """

    # 建立保存鎖定目標與服務動作的經驗池。
    def __init__(self):
        super().__init__()
        self.experience = namedtuple(
            "ServiceExperience", ["state", "target", "action", "reward", "next_state", "next_forced_service"]
        )

    # 保存指定目標下的一筆服務決策經驗。
    def add(self, state, target, action, reward, next_state, next_forced_service):
        self.memory.append(self.experience(state, target, action, reward, next_state, next_forced_service))

    # 抽取服務經驗，回傳目標索引與服務動作。
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.as_tensor(
            np.stack([e.state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        targets = torch.as_tensor(
            np.vstack([e.target for e in experiences]),
            dtype=torch.long, device=TRAIN_SETTING.device,
        )
        actions = torch.as_tensor(
            np.vstack([e.action for e in experiences]),
            dtype=torch.long, device=TRAIN_SETTING.device,
        )
        rewards = torch.as_tensor(
            np.vstack([e.reward for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_states = torch.as_tensor(
            np.stack([e.next_state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_forced_service = torch.as_tensor(
            np.vstack([e.next_forced_service for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        return states, targets, actions, rewards, next_states, next_forced_service


class HRLUpperReplayBuffer(ReplayBuffer):
    """HRL 上層 replay：``(state, action, reward, next_state, duration, completed)``。

    action 是 int64 上層離散目標索引；其他欄位為 float32。duration 是目標
    決策跨越的環境步數；completed 表示該目標任務完成，用來決定是否重選
    目標。completed 不是環境 done 標記。
    """

    # 建立保存上層目標、持續步數與完成狀態的經驗池。
    def __init__(self):
        super().__init__()
        self.experience = namedtuple(
            "HRLUpperExperience",
            ["state", "action", "reward", "next_state", "duration", "completed"],
        )

    # 保存一筆上層目標片段及其完成狀態。
    def add(self, state, action, reward, next_state, duration, completed=True):
        self.memory.append(self.experience(state, action, reward, next_state, duration, completed))

    # 抽取上層片段，回傳持續步數與目標完成標記。
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.as_tensor(
            np.stack([e.state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        actions = torch.as_tensor(
            np.vstack([e.action for e in experiences]),
            dtype=torch.long, device=TRAIN_SETTING.device,
        )
        rewards = torch.as_tensor(
            np.vstack([e.reward for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_states = torch.as_tensor(
            np.stack([e.next_state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        durations = torch.as_tensor(
            np.vstack([e.duration for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        completed = torch.as_tensor(
            np.vstack([e.completed for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        return states, actions, rewards, next_states, durations, completed


class HRLLowerReplayBuffer(ReplayBuffer):
    """HRL 下層 SAC／DSAC 系列 replay：``(state, action, reward, next_state, option_end, teacher)``。

    action 與 teacher 是連續控制向量；六個取樣欄位均為 float32。option_end
    表示服務目標完成並停止 bootstrap，不表示環境 episode 結束；teacher 是
    示範控制向量。
    """

    # 建立保存下層控制、選項結束標記與示範的經驗池。
    def __init__(self):
        self.memory = _ReplayMemory(maxlen=TRAIN_SETTING.buffer_size)
        self.batch_size = HRL_SETTING.lower_batch_size
        self.experience = namedtuple(
            "HRLLowerExperience",
            ["state", "action", "reward", "next_state", "option_end", "teacher"],
        )

    # 保存一筆下層控制經驗及對應示範動作。
    def add(self, state, action, reward, next_state, option_end, teacher):
        self.memory.append(self.experience(state, action, reward, next_state, option_end, teacher))

    # 抽取下層離策略經驗，將所有欄位轉成 float32 tensor。
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.as_tensor(
            np.stack([e.state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        actions = torch.as_tensor(
            np.stack([e.action for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        rewards = torch.as_tensor(
            np.vstack([e.reward for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        next_states = torch.as_tensor(
            np.stack([e.next_state for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        option_ends = torch.as_tensor(
            np.vstack([e.option_end for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        teachers = torch.as_tensor(
            np.stack([e.teacher for e in experiences]),
            dtype=torch.float32, device=TRAIN_SETTING.device,
        )
        return states, actions, rewards, next_states, option_ends, teachers


class JointReplayBuffer(ReplayBuffer):
    """多 UAV 聯合 replay：``(global_state, joint_action, global_reward, next_state)``。

    採用 ReplayBuffer 相同的 dtype 與 shape 規則；joint_action 是每個 UAV
    的離散動作索引向量，global_reward 每筆只記錄一次。沒有 done 欄位。
    """
