import numpy as np
import random
import math
import operator
import utils
from runtime.communication import (
    channel_power_gains,
    schedule_adaptive,
    schedule_hybrid,
    schedule_subchannels,
)

# 依環境設定產生裝置座標，可另傳入目前場景的有效性判斷。
def generate_unique_coords(is_valid=None):
    L = np.array([0., 0.], dtype=np.float32)
    H = np.array([utils.ENV_SETTING.map_size, utils.ENV_SETTING.map_size], dtype=np.float32)

    coords = []
    attempts = 0 # 總嘗試次數

    while (len(coords) < utils.ENV_SETTING.M
           and attempts < utils.ENV_SETTING.max_coord_attempts):
        attempts += 1

        candidate = np.random.uniform(low=L, high=H, size=2)

        if is_valid is not None and not is_valid(candidate):
            continue

        if coords:
            # 新座標必須與每個已接受的裝置保持最低距離。
            distances = np.linalg.norm(np.array(coords) - candidate, axis=1)
            if not np.all(distances >= utils.ENV_SETTING.min_device_distance):
                continue
        coords.append(candidate)

    if len(coords) != utils.ENV_SETTING.M:
        # 採樣可能耗盡；執行失敗不能依賴會被 python -O 關閉的 assert。
        raise RuntimeError(
            f"Could not place all devices after {attempts} attempts "
            f"({len(coords)}/{utils.ENV_SETTING.M} placed); "
            "check device spacing and available area outside risk regions"
        )
    
    return np.asarray(coords, dtype=float).reshape(utils.ENV_SETTING.M, 2)

class Environment():
    # 讀取環境設定，初始化場域與物理參數。
    def __init__(self):
        # 每次建立環境時讀取物理設定，允許在建構前調整，不變更原功耗公式。
        self.CHANNEL_GAIN = utils.ENV_SETTING.channel_gain
        self.H = utils.ENV_SETTING.altitude
        self.CELL_DISTANCE = utils.ENV_SETTING.cell_distance
        self.B = utils.ENV_SETTING.bandwidth
        self.PACKET_SIZE = utils.ENV_SETTING.packet_size
        self.SIGMA = utils.ENV_SETTING.noise_power
        self.M = utils.ENV_SETTING.M
        self.U = utils.ENV_SETTING.U
        self.map_size = utils.ENV_SETTING.map_size
        self.max_mov = utils.ENV_SETTING.max_mov
        self.max_steps = utils.TRAIN_SETTING.max_steps
        self.energy_weight = utils.ENV_SETTING.energy_weight
        self.penalty = utils.ENV_SETTING.penalty
        self.prob_of_risk = utils.ENV_SETTING.prob_of_risk
        self.arrival_model = utils.ENV_SETTING.arrival_model
        self.arrival_interval = utils.ENV_SETTING.arrival_interval
        self.arrival_probability = utils.ENV_SETTING.arrival_probability
        self.arrival_rate = utils.ENV_SETTING.arrival_rate
        self.communication_model = utils.ENV_SETTING.communication_model
        self.max_transmit_power = utils.ENV_SETTING.max_transmit_power
        self.fixed_transmit_power = utils.ENV_SETTING.fixed_transmit_power
        self.subchannel_count = utils.ENV_SETTING.subchannel_count

        self.device_coord = np.array([])
        self.risk_region = np.array([])
        self.UAV_coord = np.array([])

        self.L_corner = [0.0, 0.0]
        self.H_corner = [self.map_size, self.map_size]
        
    # 重置環境
    def reset(self):        
        if utils.FUNCTIONAL_SETTING.risk:
            self.risk_region = np.array(utils.FIXED_SETTING.fixed_risk_region, dtype=float, copy=True).reshape(-1, 4)
        else:
            self.risk_region = np.empty((0, 4))

        if utils.FUNCTIONAL_SETTING.random_device_coord:
            self.device_coord = generate_unique_coords(
                is_valid=lambda coord: not self.is_coord_risky(coord)
            )
        else:
            self.device_coord = np.array(utils.FIXED_SETTING.fixed_device_coord, dtype=float, copy=True)

        if utils.FUNCTIONAL_SETTING.random_UAV_coord:
            max_attempts = utils.ENV_SETTING.max_coord_attempts
            self.UAV_coord = np.empty((self.U, 2), dtype=float)
            for i in range(self.U):
                for _ in range(max_attempts):
                    coord = np.random.uniform(self.L_corner, self.H_corner, size=2)
                    if not self.is_coord_risky(coord):
                        self.UAV_coord[i] = coord
                        break
                else:
                    raise RuntimeError(
                        f"Could not place UAV {i} outside risk regions after "
                        f"{max_attempts} attempts"
                    )
        else:
            self.UAV_coord = np.array(utils.FIXED_SETTING.fixed_UAV_coord, dtype=float, copy=True)


        self.source_AOI = np.ones(self.M)  # 每個裝置目前等待服務的封包年齡。
        self.AOI = np.ones(self.M) # 初始化collector AoI

        self.steps = 0
        self.power = 0.0  # 本步所有 UAV 的加權傳輸成本與實際風險懲罰總和。
        self.rewards = [0] * self.U
        self.total_reward = 0.0
        # 輸出用分項紀錄，不改變 reward 或抽樣流程。
        self.transmission_power = np.zeros(self.U)
        self.transmission_peak_power = np.zeros(self.U)
        self.transmission_time = np.zeros(self.U)
        self.service_success = np.zeros(self.U, dtype=bool)
        self.communication_scheme = "idle"
        self.risk_penalty_counts = np.zeros(self.U, dtype=int)
    
    # 回傳原始觀測：[UAV 座標、裝置座標、AoI]，長度 2U+3M。
    def get_state(self):
        return np.concatenate((np.asarray(self.UAV_coord).reshape(-1),
                               np.asarray(self.device_coord).reshape(-1), self.AOI), axis=None)

    # 服務有效裝置時，collector AoI 取目前封包年齡；M 表示 idle。
    def AOI_Reset(self, device):
        if 0 <= device < self.M:
            self.AOI[device] = self.source_AOI[device]

    # 判斷座標是否位於任一風險區內。
    def is_coord_risky(self, coord):
        x, y = coord

        for region in self.risk_region:
            xmin, ymin, xmax, ymax = region

            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True

        return False

    # 限制移動距離與地圖邊界，回傳更新後的座標。
    def Update_Location(self, coord, vec):
        vec = np.asarray(vec, dtype=float)
        distance = np.linalg.norm(vec)
        if distance > self.max_mov:
            # 保持方向，將位移長度限制在每步的最大移動距離。
            vec = vec * (self.max_mov / distance)

        next_coord = np.clip(
            np.asarray(coord, dtype=float) + vec,
            self.L_corner,
            self.H_corner
        )

        next_coord = np.asarray(next_coord).reshape(-1)

        return next_coord
        
    # 必要的傳輸功率
    def Transmission_Power_Calc(self, device, UAV):
        if device >= self.M:
            return 0
        horizontal_distance = self.CELL_DISTANCE * math.dist(
            self.UAV_coord[UAV], self.device_coord[device]
        )
        # 路徑距離平方乘上所需訊雜比，再按雜訊與通道增益換算功率。
        return ((horizontal_distance ** 2 + self.H ** 2)
                * (2 ** (self.PACKET_SIZE / self.B) - 1)
                * self.SIGMA / self.CHANNEL_GAIN)

    # 回傳加權傳輸成本；風險懲罰另由 step() 加入。
    def Power_Calc(self, device, UAV):
        return self.Transmission_Power_Calc(device, UAV) * self.energy_weight
    
    # power 是傳入的成本，可包含風險懲罰；AoI 對所有裝置取平均。
    def Reward_Calc(self, AOI, power):
        return -power - np.sum(AOI) / self.M
    
    # 風險機率  
    def Risk_prob(self, coord):
        if self.is_coord_risky(coord):
            return self.prob_of_risk
        return 0

    # SINR 模式先完成所有移動與頻道排程，再依 UAV 索引抽取風險事件。
    def _step_sinr(self, actions, uav_costs):
        action_targets = [actions[u][0] for u in range(self.U)]
        target_indices = [None] * self.U

        for u in range(self.U):
            _action_select, action_move = actions[u]
            self.UAV_coord[u] = self.Update_Location(self.UAV_coord[u], action_move)

        gains = channel_power_gains(
            self.device_coord,
            self.UAV_coord,
            self.CELL_DISTANCE,
            self.H,
            self.CHANNEL_GAIN,
        )

        # Duplicate requests are represented by the receiver with the strongest
        # direct link. Iteration order resolves an exact gain tie by UAV index.
        best_receiver_by_device = {}
        for u, action_select in enumerate(action_targets):
            try:
                device = operator.index(action_select)
            except TypeError:
                continue
            if device < 0 or device >= self.M:
                continue
            target_indices[u] = device
            current_receiver = best_receiver_by_device.get(device)
            if current_receiver is None:
                best_receiver_by_device[device] = u
            elif gains[u, device] > gains[current_receiver, device]:
                best_receiver_by_device[device] = u

        requests = []
        for device, uav in best_receiver_by_device.items():
            # Both arrays already include this slot's aging and packet arrivals.
            potential_reduction = float(self.AOI[device] - self.source_AOI[device])
            requests.append((uav, device, potential_reduction))
        requests.sort(key=lambda request: (-request[2], request[0]))

        power_cap = float(self.max_transmit_power)
        if self.communication_model == "sinr_hybrid":
            (peak_power, active_time, success,
             self.communication_scheme) = schedule_hybrid(
                requests, gains, self.SIGMA, self.B, self.PACKET_SIZE,
                power_cap, self.U)
        elif self.communication_model == "sinr_adaptive":
            fixed_power = float(self.fixed_transmit_power)
            (peak_power, active_time, success,
             self.communication_scheme) = schedule_adaptive(
                requests, gains, self.SIGMA, self.B, self.PACKET_SIZE,
                fixed_power, power_cap, self.U)
        elif self.communication_model == "sinr_subchannels":
            subchannel_count = int(self.subchannel_count)
            (peak_power, active_time, success,
             self.communication_scheme) = schedule_subchannels(
                requests, gains, self.SIGMA, self.B, self.PACKET_SIZE,
                power_cap, subchannel_count, self.U)
        else:
            raise ValueError(
                f"Unsupported communication model: {self.communication_model!r}")

        self.transmission_peak_power = peak_power
        self.transmission_time = active_time
        self.service_success = success
        # One slot is exactly one second, so average watts equal delivered joules.
        self.transmission_power = self.transmission_peak_power * self.transmission_time

        # Keep the per-UAV Python RNG draw sequence stable, including idle or
        # deferred UAVs. Risk cost remains independent of communication success.
        for u in range(self.U):
            uav_cost = self.transmission_power[u] * self.energy_weight
            if utils.FUNCTIONAL_SETTING.risk:
                risk_prob = self.Risk_prob(self.UAV_coord[u])
                if risk_prob > 0:
                    sample_prob = random.random()
                    if sample_prob < risk_prob:
                        uav_cost += self.penalty
                        self.risk_penalty_counts[u] = 1
            uav_costs[u] = uav_cost
            self.power += uav_cost

        # The channel schedule is complete before any collector AoI is reset.
        for u in range(self.U):
            if self.service_success[u]:
                self.AOI_Reset(target_indices[u])
    
    # 執行環境
    def step(self, actions):
        # 來源封包與 collector AoI 都先經過一個環境步。
        self.source_AOI += 1
        self.AOI += 1

        if self.arrival_model == "bernoulli":
            arrivals = np.random.random(self.M) < self.arrival_probability
            self.source_AOI[arrivals] = 1
        elif self.arrival_model == "poisson":
            # slotted Poisson 只需判斷此步是否至少到達一個封包。
            arrival_probability = -np.expm1(-self.arrival_rate)
            arrivals = np.random.random(self.M) < arrival_probability
            self.source_AOI[arrivals] = 1
        elif self.arrival_model == "always":
            # steps 尚未遞增；在第 n、2n、3n 槽同步產生資料，週期 1 保留舊行為且不抽樣。
            if (self.steps + 1) % self.arrival_interval == 0:
                self.source_AOI.fill(1)

        self.power = 0.0

        self.rewards = [0] * self.U
        uav_costs = [0.0] * self.U
        self.transmission_power = np.zeros(self.U)
        self.transmission_peak_power = np.zeros(self.U)
        self.transmission_time = np.zeros(self.U)
        self.service_success = np.zeros(self.U, dtype=bool)
        self.communication_scheme = "idle"
        self.risk_penalty_counts = np.zeros(self.U, dtype=int)

        if self.communication_model == "snr":
            # Preserve the legacy independent-link sequence and equations exactly.
            for u in range(self.U):
                action_select, action_move = actions[u]

                self.UAV_coord[u] = self.Update_Location(self.UAV_coord[u], action_move)

                self.transmission_power[u] = self.Transmission_Power_Calc(action_select, u)
                self.transmission_peak_power[u] = self.transmission_power[u]
                if action_select < self.M:
                    self.transmission_time[u] = 1.0
                if 0 <= action_select < self.M:
                    self.service_success[u] = True
                uav_cost = self.transmission_power[u] * self.energy_weight

                if utils.FUNCTIONAL_SETTING.risk:
                    risk_prob = self.Risk_prob(self.UAV_coord[u])
                    if risk_prob > 0:
                        sample_prob = random.random()
                        if sample_prob < risk_prob:
                            uav_cost += self.penalty
                            self.risk_penalty_counts[u] = 1

                self.AOI_Reset(action_select)

                uav_costs[u] = uav_cost

                self.power += uav_cost
            if np.any(self.transmission_time > 0):
                self.communication_scheme = "snr"
        else:
            self._step_sinr(actions, uav_costs)
        
        # 各 UAV 共用服務後的平均 AoI；total_reward 等於各 UAV reward 的平均。
        self.rewards = [self.Reward_Calc(self.AOI, cost) for cost in uav_costs]
        self.total_reward = self.Reward_Calc(self.AOI, self.power/self.U)

        self.steps += 1
