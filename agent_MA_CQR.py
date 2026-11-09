import random

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from networks import CQR_DQN
from utils import MA_CQR_SETTING, TRAIN_SETTING


class CQRAgent:
    # eta 是 lower-tail CVaR 的質量比例；網路始終學習完整回報分布。
    def __init__(self):
        self.state_size = MA_CQR_SETTING.state_size
        self.action_size = MA_CQR_SETTING.action_size
        self.hidden_size = MA_CQR_SETTING.hidden_size
        self.device = TRAIN_SETTING.device
        quantile_indices = torch.arange(
            MA_CQR_SETTING.N, device=self.device, dtype=torch.float32
        )
        self.quantile_tau = (quantile_indices + 0.5) / MA_CQR_SETTING.N
        self.network = CQR_DQN(state_size=self.state_size,
                               action_size=self.action_size,
                               layer_size=self.hidden_size,
                               N=MA_CQR_SETTING.N).to(self.device)
        self.target_network = CQR_DQN(state_size=self.state_size,
                                      action_size=self.action_size,
                                      layer_size=self.hidden_size,
                                      N=MA_CQR_SETTING.N).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.requires_grad_(False)

    # 排序分位數後累加完整原子與一個邊界原子的部分質量，計算 lower-tail CVaR。
    def lower_tail_cvar(self, quantiles):
        if float(MA_CQR_SETTING.eta) == 1.:
            return quantiles.mean(dim=1)

        ordered = torch.sort(quantiles, dim=1).values
        tail_mass = float(MA_CQR_SETTING.eta) * MA_CQR_SETTING.N
        full_atom_count = int(tail_mass)
        if full_atom_count == 0:
            # 尾端只有最小分位數的部分質量，正規化後就是該最小值。
            return ordered[:, 0]

        fractional_atom_mass = tail_mass - full_atom_count

        # 先將各原子除以尾端質量再加總，避免大分位數的中間總和溢位。
        full_atom_contribution = (
            ordered[:, :full_atom_count] / tail_mass
        ).sum(dim=1)
        boundary_atom_contribution = (
            fractional_atom_mass / tail_mass
        ) * ordered[:, full_atom_count]
        return full_atom_contribution + boundary_atom_contribution

    # 選擇最低回報尾端 CVaR 估值最大的動作。
    def risk_action(self, quantiles):
        return self.lower_tail_cvar(quantiles).argmax(dim=1)

    # 探索與貪婪分支均回傳 shape=(1,) 的整數 ndarray。
    def get_action(self, state, epsilon=0.):
        if random.random() < epsilon:
            return np.asarray([random.randrange(self.action_size)], dtype=np.int64)
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.risk_action(self.network(state_tensor))
        return action.cpu().numpy()


class CentralQRTrainer:
    # 所有 UAV 共用一個持續存在的 optimizer，支援任意 UAV 數量。
    def __init__(self, agents):
        assert agents
        self.agents = agents
        self.parameters = [p for agent in agents for p in agent.network.parameters()]
        self.optimizer = optim.Adam(self.parameters, lr=MA_CQR_SETTING.lr)

    # 聯合經驗必須來自同一步；全域 reward 僅加入一次，不乘 UAV 數量。
    def joint_quantiles(self, experiences):
        states, actions, rewards, next_states = experiences
        expected_quantiles, target_quantiles = [], []
        for agent_index, agent in enumerate(self.agents):
            # 擴展動作索引以沿最後一軸 gather，保留每個分位數的選定動作值。
            action_indices = actions[:, agent_index:agent_index + 1].unsqueeze(1)
            action_indices = action_indices.expand(-1, MA_CQR_SETTING.N, -1)
            current_action_quantiles = agent.network(states).gather(2, action_indices)
            expected_quantiles.append(current_action_quantiles.squeeze(-1))

            with torch.no_grad():
                # Double DQN：online 網路選動作，target 網路估分位數。
                next_action_values = agent.network(next_states)
                next_actions = agent.risk_action(next_action_values).unsqueeze(1)
                next_action_indices = next_actions.unsqueeze(1).expand(
                    -1, MA_CQR_SETTING.N, -1
                )
                next_action_quantiles = agent.target_network(next_states).gather(
                    2, next_action_indices
                )
                target_quantiles.append(next_action_quantiles.squeeze(-1))

        joint_expected_quantiles = torch.stack(expected_quantiles).sum(dim=0)
        joint_next_quantiles = torch.stack(target_quantiles).sum(dim=0)
        td_targets = rewards + MA_CQR_SETTING.gamma * joint_next_quantiles
        return joint_expected_quantiles, td_targets

    # 標準 Quantile Huber Loss；tau 對齊預測分位數 i，而非目標分位數 j。
    def Loss_Calc(self, expected, target):
        td_error = target.detach().unsqueeze(1) - expected.unsqueeze(2)
        huber_loss = torch.where(
            td_error.abs() <= MA_CQR_SETTING.huber_kappa,
            .5 * td_error.square(),
            MA_CQR_SETTING.huber_kappa * td_error.abs() - .5 * MA_CQR_SETTING.huber_kappa ** 2,
        )
        quantile_levels = self.agents[0].quantile_tau.view(1, -1, 1)
        # td_error 的形狀為 (batch, 預測分位數 i, 目標分位數 j)。
        quantile_weights = (quantile_levels - (td_error < 0).float()).abs()
        # 對目標分位數取平均、預測分位數加總，再對 batch 取平均。
        return (quantile_weights * huber_loss).mean(dim=2).sum(dim=1).mean()

    # 一次聯合反向傳播更新所有 UAV，不包含 CQL 或其他保守正則項。
    def learn(self, experiences):
        expected, target = self.joint_quantiles(experiences)
        loss = self.Loss_Calc(expected, target)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        self.soft_update()
        return float(loss.detach())

    # Soft update 不參與反向傳播。
    def soft_update(self):
        local_params = []
        target_params = []
        for agent in self.agents:
            local_params.extend(agent.network.parameters())
            target_params.extend(agent.target_network.parameters())
        with torch.no_grad():
            torch._foreach_mul_(target_params, 1. - MA_CQR_SETTING.tau)
            torch._foreach_add_(target_params, local_params, alpha=MA_CQR_SETTING.tau)
