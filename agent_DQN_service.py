import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import ServiceDQN
from utils import DQN_SERVICE_SETTING, TRAIN_SETTING


class DQNAgent:
    # 依 service 設定建立雙頭網路、目標網路與 optimizer。
    def __init__(self):
        self.state_size = DQN_SERVICE_SETTING.state_size
        self.action_size = DQN_SERVICE_SETTING.action_size
        self.hidden_size = DQN_SERVICE_SETTING.hidden_size
        self.device = TRAIN_SETTING.device

        self.network = ServiceDQN(state_size=self.state_size,
                                  action_size=self.action_size,
                                  layer_size=self.hidden_size).to(self.device)
        self.target_net = ServiceDQN(state_size=self.state_size,
                                     action_size=self.action_size,
                                     layer_size=self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=DQN_SERVICE_SETTING.lr)

    # 以 epsilon-greedy 選擇目標裝置或懸停。
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

    # 依鎖定目標選擇繼續移動或立即服務。
    def get_service_action(self, state, target, epsilon):
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            target_tensor = torch.tensor([target], dtype=torch.long, device=self.device)
            
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state_tensor, target_tensor)
            self.network.train()
            
            action =  np.argmax(action_values.cpu().numpy(), axis=1)
        else:
            action = np.asarray([random.randrange(2)], dtype=np.int64)
        return action

    # 依目標持續步數折扣，建立目標選擇頭的 TD 目標。
    def _build_targets(self, experiences):
        states, actions, rewards, next_states, durations = experiences
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1).values.unsqueeze(1)
            # 每筆經驗跨越不同步數，因此以 gamma 的 duration 次方折扣。
            td_targets = rewards + DQN_SERVICE_SETTING.gamma ** durations * max_next_q_values
        return states, actions, td_targets

    # 用跨步經驗更新目標選擇頭並同步目標網路。
    def learn(self, experiences):
        states, actions, td_targets = self._build_targets(experiences)
        action_q_values = self.network(states)
        # gather 沿動作維度取出每筆經驗實際選擇的 Q(s, a)。
        predicted_q_values = action_q_values.gather(1, actions)
        loss = 0.5 * F.mse_loss(predicted_q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()

    # 依本步是否已服務及下一步是否強制服務，建立服務頭的 TD 目標。
    def _build_targets_service(self, experiences):
        states, targets, actions, rewards, next_states, next_forced_service = experiences
        with torch.no_grad():
            next_service_q_values = self.target_net(next_states, targets)
            # 已服務則重新選目標；未服務則依下一步是否強制服務取原目標的 Q。
            next_step_q_values = torch.where(
                actions == 1,
                self.target_net(next_states).max(dim=1, keepdim=True).values,
                torch.where(
                    next_forced_service.bool(),
                    next_service_q_values[:, 1:2],  # 索引 1 為服務，保留 (batch, 1)。
                    next_service_q_values.max(dim=1, keepdim=True).values,
                ),
            )
            td_targets = rewards + DQN_SERVICE_SETTING.gamma * next_step_q_values
        return states, targets, actions, td_targets

    # 用單步服務經驗更新服務頭並同步目標網路。
    def learn_service(self, experiences):
        states, targets, actions, td_targets = self._build_targets_service(experiences)
        action_q_values = self.network(states, targets)
        predicted_q_values = action_q_values.gather(1, actions)
        loss = 0.5 * F.mse_loss(predicted_q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()

    # 按設定比例同步目標網路的權重。
    def soft_update(self):
        if self.device == "cuda":
            local_values = [parameter.data for parameter in self.network.parameters()]
            target_values = [parameter.data for parameter in self.target_net.parameters()]
            local_updates = torch._foreach_mul(local_values, DQN_SERVICE_SETTING.tau)
            target_updates = torch._foreach_mul(target_values, 1.0 - DQN_SERVICE_SETTING.tau)
            updated_values = torch._foreach_add(local_updates, target_updates)
            torch._foreach_copy_(target_values, updated_values)
        else:
            for target_param, local_param in zip(
                self.target_net.parameters(), self.network.parameters()
            ):
                target_param.data.copy_(DQN_SERVICE_SETTING.tau * local_param.data +
                                        (1.0 - DQN_SERVICE_SETTING.tau) * target_param.data)
