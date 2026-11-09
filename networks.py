import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    # 建立輸出各離散動作 Q 值的全連接網路。
    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.action_size = action_size
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    # 由觀測計算各動作的 Q 值。
    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        return self.ff_2(x)


class ServiceDQN(DDQN):
    # 建立目標選擇頭與目標條件服務頭。
    def __init__(self, state_size, action_size, layer_size):
        super().__init__(state_size, action_size, layer_size)
        self.service_head = nn.Sequential(
            nn.Linear(layer_size + action_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 2),
        )

    # 回傳目標 Q 值，或指定目標下的服務 Q 值。
    def forward(self, input, targets=None):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        if targets is None:
            return self.ff_2(x)
        # 將鎖定目標編成 one-hot，與狀態特徵一起交給服務頭。
        targets = F.one_hot(targets.long().reshape(-1), self.action_size).to(x.dtype)
        return self.service_head(torch.cat((x, targets), dim=1))

# 通用連續動作策略網路；不在網路層內決定分布或動作縮放。
class GaussianActor(nn.Module):
    # 輸入與動作維度由 agent 建構時指定。
    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.actor = nn.Linear(layer_size, action_size * 2)

    # 回傳未經 tanh 的平均值及 log 標準差。
    def forward(self, state):
        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))
        # 輸出前、後兩半分別代表平均值與 log 標準差。
        return self.actor(x).chunk(2, dim=-1)


# 通用 Q(s,a) 網路，每個實例具有獨立參數。
class QCritic(nn.Module):
    # 將狀態及連續動作串接後，估計一個 Q 值。
    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.head_1 = nn.Linear(state_size + action_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)

    # 支援批次狀態與動作，不依賴具體 UAV 或裝置數量。
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.head_1(x))
        x = torch.relu(self.ff_1(x))
        return self.ff_2(x)


class StateValueCritic(nn.Module):
    # 由狀態估計 scalar V(s)。
    def __init__(self, state_size, layer_size):
        super().__init__()
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.value = nn.Linear(layer_size, 1)

    def forward(self, state):
        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))
        return self.value(x)


class GaussianReturnCritic(nn.Module):
    # 對連續 (s,a) 輸出 scalar 回報分布的平均值及原始標準差參數。
    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.head_1 = nn.Linear(state_size + action_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.mean = nn.Linear(layer_size, 1)
        self.raw_std = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.head_1(x))
        x = torch.relu(self.ff_1(x))
        return self.mean(x), self.raw_std(x)


class CQR_DQN(nn.Module):
    # 建立每個動作輸出 N 個分位數的 Q 網路。
    def __init__(self, state_size, action_size, layer_size, N):
        super().__init__()
        self.action_size = action_size
        self.N = N
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size * N)

    # 輸出 (batch, quantile, action)，每個動作有 N 個分位數。
    def forward(self, state):
        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))
        return self.ff_2(x).view(state.shape[0], self.N, self.action_size)


class SharedDeviceQNetwork(nn.Module):
    """Permutation-equivariant Q values for devices and one current-position action.

    Input rows use the HRL upper-policy layout: focal UAV xy, all UAV xy, all
    device xy, then device AoI. The first ``M`` outputs follow device order
    and output ``M`` is the current-position candidate. If physical candidates
    have tied Q values, argmax may select either tied index; equivariance
    guarantees their values, not tie-breaking.
    """

    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device_count = action_size - 1
        remaining_state = state_size - 2 - 3 * self.device_count
        if remaining_state < 0 or remaining_state % 2:
            raise ValueError(
                "HRL state_size must equal 2 + 2 * UAV_count + 3 * device_count"
            )
        self.uav_count = remaining_state // 2
        if self.device_count < 1 or self.uav_count < 1:
            raise ValueError("SharedDeviceQNetwork requires at least one UAV and device")

        self.candidate_encoder = nn.Sequential(
            nn.Linear(7, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.uav_encoder = nn.Sequential(
            nn.Linear(2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.focal_encoder = nn.Sequential(
            nn.Linear(2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.candidate_uav_encoder = nn.Sequential(
            nn.Linear(3, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(3 * layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(3 * layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1),
        )

    def forward(self, state):
        unbatched = state.ndim == 1
        if unbatched:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected HRL state shape (batch, {self.state_size}) or "
                f"({self.state_size},), got {tuple(state.shape)}"
            )

        batch_size = state.shape[0]
        focal_xy = state[:, :2]
        uav_start = 2
        uav_end = uav_start + 2 * self.uav_count
        device_xy_end = uav_end + 2 * self.device_count
        uav_xy = state[:, uav_start:uav_end].reshape(
            batch_size, self.uav_count, 2
        )
        device_xy = state[:, uav_end:device_xy_end].reshape(
            batch_size, self.device_count, 2
        )
        device_age = state[:, device_xy_end:].reshape(
            batch_size, self.device_count, 1
        )

        relative_device_xy = device_xy - focal_xy.unsqueeze(1)
        device_distance = torch.linalg.vector_norm(
            relative_device_xy, dim=-1, keepdim=True
        )
        device_current_position_flag = state.new_zeros(
            (batch_size, self.device_count, 1)
        )
        device_features = torch.cat(
            (
                device_xy,
                relative_device_xy,
                device_distance,
                device_age,
                device_current_position_flag,
            ),
            dim=-1,
        )

        # The current-position action is excluded from device pooling so its
        # representation is invariant to device order.
        current_position_features = torch.cat(
            (
                focal_xy.unsqueeze(1),
                state.new_zeros((batch_size, 1, 4)),
                state.new_ones((batch_size, 1, 1)),
            ),
            dim=-1,
        )
        candidate_features = torch.cat(
            (device_features, current_position_features), dim=1
        )
        candidate_embeddings = self.candidate_encoder(candidate_features)
        pooled_device_embeddings = candidate_embeddings[:, :self.device_count].mean(dim=1)

        relative_uav_xy = uav_xy - focal_xy.unsqueeze(1)
        pooled_uav_embeddings = self.uav_encoder(relative_uav_xy).mean(dim=1)
        focal_embedding = self.focal_encoder(focal_xy)
        context = self.context_encoder(
            torch.cat(
                (pooled_device_embeddings, pooled_uav_embeddings, focal_embedding),
                dim=-1,
            )
        )

        candidate_xy = candidate_features[..., :2]
        relative_candidate_uav_xy = (
            candidate_xy.unsqueeze(2) - uav_xy.unsqueeze(1)
        )
        candidate_uav_distance = torch.linalg.vector_norm(
            relative_candidate_uav_xy, dim=-1, keepdim=True
        )
        candidate_uav_pair_features = torch.cat(
            (relative_candidate_uav_xy, candidate_uav_distance), dim=-1
        )
        candidate_uav_pair_embeddings = self.candidate_uav_encoder(
            candidate_uav_pair_features
        )
        if self.uav_count > 1:
            relative_candidate_focal_xy = candidate_xy - focal_xy.unsqueeze(1)
            candidate_focal_distance = torch.linalg.vector_norm(
                relative_candidate_focal_xy, dim=-1, keepdim=True
            )
            candidate_focal_pair_features = torch.cat(
                (relative_candidate_focal_xy, candidate_focal_distance), dim=-1
            )
            candidate_focal_pair_embeddings = self.candidate_uav_encoder(
                candidate_focal_pair_features
            )
            # Subtract one focal-pair embedding, rather than a UAV index, so
            # pooling stays permutation-invariant and removes only one UAV
            # even when another UAV is co-located with the focal UAV.
            candidate_other_uav_context = (
                candidate_uav_pair_embeddings.sum(dim=2)
                - candidate_focal_pair_embeddings
            ) / (self.uav_count - 1)
        else:
            candidate_other_uav_context = state.new_zeros(
                (
                    batch_size,
                    self.action_size,
                    self.candidate_uav_encoder[0].out_features,
                )
            )

        candidate_context = context.unsqueeze(1).expand(
            -1, self.action_size, -1
        )
        q_values = self.scorer(
            torch.cat(
                (candidate_embeddings, candidate_context, candidate_other_uav_context),
                dim=-1,
            )
        ).squeeze(-1)
        if unbatched:
            return q_values.squeeze(0)
        return q_values


class HRLPPOActor(nn.Module):
    """Feedforward Gaussian policy for the HRL-PPO lower controller.

    The network predicts an unsquashed action mean. A single state-independent
    log standard deviation is learned for each action dimension; the agent
    applies the configured bounds and tanh transform when sampling actions.
    """

    def __init__(self, state_size, action_size, layer_size):
        super().__init__()
        self.head_1 = nn.Linear(state_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.mean = nn.Linear(layer_size, action_size)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

        nn.init.orthogonal_(self.head_1.weight, gain=2 ** 0.5)
        nn.init.zeros_(self.head_1.bias)
        nn.init.orthogonal_(self.ff_1.weight, gain=2 ** 0.5)
        nn.init.zeros_(self.ff_1.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def forward(self, state):
        x = torch.tanh(self.head_1(state))
        x = torch.tanh(self.ff_1(x))
        return self.mean(x)
