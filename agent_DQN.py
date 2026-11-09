import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import DDQN
from utils import DQN_SETTING, TRAIN_SETTING


class DQNAgent:
    # 依設定建立 Q 網路、目標網路與 optimizer。
    def __init__(self):
        self.state_size = DQN_SETTING.state_size
        self.action_size = DQN_SETTING.action_size
        self.hidden_size = DQN_SETTING.hidden_size
        self.device = TRAIN_SETTING.device

        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=self.hidden_size).to(self.device)
        self.target_net = DDQN(state_size=self.state_size,
                               action_size=self.action_size,
                               layer_size=self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=DQN_SETTING.lr)

    # 以 epsilon-greedy 選擇離散動作。
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state_tensor)
            self.network.train()

            action = np.argmax(action_values.cpu().numpy(), axis=1)
        else:
            action = np.asarray(random.choices(np.arange(self.action_size), k=1), dtype=np.int64)
        return action

    # 以下一狀態的目標 Q 值建立單步 TD 目標。
    def _build_targets(self, experiences):
        states, actions, rewards, next_states = experiences
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1).values.unsqueeze(1)
            td_targets = rewards + DQN_SETTING.gamma * max_next_q_values
        return states, actions, td_targets

    # 更新 Q 網路、裁剪梯度並同步目標網路。
    def learn(self, experiences):
        states, actions, td_targets = self._build_targets(experiences)
        action_q_values = self.network(states)
        # 沿動作維度取出每筆經驗實際選擇的 Q(s, a)。
        predicted_q_values = action_q_values.gather(1, actions)
        # 0.5 * mean(TD error²)：微分後消去平方項的係數 2。
        loss = 0.5 * F.mse_loss(predicted_q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        # 將整個 Q 網路的梯度視為一個向量，限制其 L2 norm。
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()

    # 按設定比例混合線上與目標網路的權重。
    def soft_update(self):
        if self.device == "cuda":
            local_values = [parameter.data for parameter in self.network.parameters()]
            target_values = [parameter.data for parameter in self.target_net.parameters()]
            local_updates = torch._foreach_mul(local_values, DQN_SETTING.tau)
            target_updates = torch._foreach_mul(target_values, 1.0 - DQN_SETTING.tau)
            updated_values = torch._foreach_add(local_updates, target_updates)
            torch._foreach_copy_(target_values, updated_values)
        else:
            for target_param, local_param in zip(
                self.target_net.parameters(), self.network.parameters()
            ):
                target_param.data.copy_(DQN_SETTING.tau * local_param.data +
                                        (1.0 - DQN_SETTING.tau) * target_param.data)
