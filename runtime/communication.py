"""Deterministic slot schedulers for the optional SINR communication models."""

import math

import numpy as np


def channel_power_gains(device_coord, uav_coord, cell_distance, altitude, channel_gain):
    """Return gain[receiver UAV, transmitting device] using the legacy path law."""
    device_coord = np.asarray(device_coord, dtype=float)
    uav_coord = np.asarray(uav_coord, dtype=float)
    horizontal = cell_distance * np.linalg.norm(
        uav_coord[:, None, :] - device_coord[None, :, :], axis=2)
    path_distance_squared = horizontal ** 2 + altitude ** 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gains = channel_gain / path_distance_squared
    return gains


def _requested_sinr(packet_size, bandwidth, duration):
    """Return 2**(packet_size/(bandwidth*duration))-1, or None if unrepresentable."""
    if (not math.isfinite(packet_size) or not math.isfinite(bandwidth)
            or not math.isfinite(duration) or packet_size < 0
            or bandwidth <= 0 or duration <= 0):
        return None
    exponent = (packet_size / bandwidth) / duration
    if not math.isfinite(exponent) or exponent < 0:
        return None
    log_gamma = exponent * math.log(2.0)
    if log_gamma > math.log(np.finfo(float).max):
        return None
    gamma = math.expm1(log_gamma)
    if not math.isfinite(gamma) or gamma < 0:
        return None
    return gamma


def _minimum_power_for_group(requests, gains, noise_power, bandwidth,
                             packet_size, duration, power_cap):
    """Solve the minimum-power cochannel problem for one equal-bandwidth group."""
    count = len(requests)
    if count == 0:
        return np.empty(0, dtype=float)
    gamma = _requested_sinr(packet_size, bandwidth, duration)
    if gamma is None:
        return None
    if (not math.isfinite(noise_power) or noise_power <= 0
            or not math.isfinite(power_cap) or power_cap < 0):
        return None

    uav_indices = np.asarray([request[0] for request in requests], dtype=int)
    device_indices = np.asarray([request[1] for request in requests], dtype=int)
    desired = gains[uav_indices, device_indices]
    if (not np.all(np.isfinite(desired)) or np.any(desired <= 0)):
        return None

    # Matrix rows are receivers and columns are the devices transmitting in this group.
    cross = gains[np.ix_(uav_indices, device_indices)].copy()
    if not np.all(np.isfinite(cross)) or np.any(cross < 0):
        return None
    np.fill_diagonal(cross, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        interference_factor = gamma * (cross / desired[:, None])
        noise_term = gamma * (noise_power / desired)
    if (not np.all(np.isfinite(interference_factor))
            or not np.all(np.isfinite(noise_term))):
        return None

    # A nonnegative power solution exists only when the interference coupling is stable.
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(interference_factor))))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(spectral_radius) or spectral_radius >= 1.0:
        return None

    try:
        powers = np.linalg.solve(np.eye(count) - interference_factor, noise_term)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(powers)):
        return None
    if np.any(powers < -1e-12):
        return None
    powers = np.maximum(powers, 0.0)
    if np.any(powers > power_cap):
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        interference = cross @ powers
        sinr = powers * desired / (noise_power + interference)
    if not np.all(np.isfinite(sinr)):
        return None
    # Allow only small floating-point error from solving the linear system.
    if np.any(sinr < gamma * (1.0 - 1e-9)):
        return None
    return powers


def _tdma_powers(requests, gains, noise_power, bandwidth, packet_size,
                 duration, power_cap):
    """Compute independent link powers for equal-share, full-slot TDMA."""
    if not requests:
        return np.empty(0, dtype=float)
    gamma = _requested_sinr(packet_size, bandwidth, duration)
    if (gamma is None or not math.isfinite(noise_power) or noise_power <= 0
            or not math.isfinite(power_cap) or power_cap < 0):
        return None
    uav_indices = np.asarray([request[0] for request in requests], dtype=int)
    device_indices = np.asarray([request[1] for request in requests], dtype=int)
    desired = gains[uav_indices, device_indices]
    if not np.all(np.isfinite(desired)) or np.any(desired <= 0):
        return None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        powers = gamma * (noise_power / desired)
    if not np.all(np.isfinite(powers)) or np.any(powers < 0):
        return None
    if np.any(powers > power_cap):
        return None
    return powers


def schedule_hybrid(requests, gains, noise_power, bandwidth, packet_size,
                    power_cap, uav_count):
    """Greedily admit by priority, choosing minimum-energy simultaneous or TDMA."""
    powers_by_uav = np.zeros(uav_count, dtype=float)
    times_by_uav = np.zeros(uav_count, dtype=float)
    success = np.zeros(uav_count, dtype=bool)
    admitted = []
    chosen_mode = "idle"

    for request in requests:
        trial = admitted + [request]
        simultaneous = _minimum_power_for_group(
            trial, gains, noise_power, bandwidth, packet_size, 1.0, power_cap)
        tdma_duration = 1.0 / len(trial)
        tdma = _tdma_powers(
            trial, gains, noise_power, bandwidth, packet_size,
            tdma_duration, power_cap)

        options = []
        if simultaneous is not None:
            options.append((float(np.sum(simultaneous)), 0, "simultaneous", simultaneous, 1.0))
        if tdma is not None:
            options.append((float(np.sum(tdma) * tdma_duration), 1, "tdma", tdma, tdma_duration))
        if not options:
            # Keep considering later requests; a bad link does not block the rest.
            continue

        _, _, chosen_mode, chosen_powers, chosen_duration = min(
            options, key=lambda option: (option[0], option[1]))
        admitted = trial
        for (uav_index, _device_index, _priority), power in zip(admitted, chosen_powers):
            powers_by_uav[uav_index] = power
            times_by_uav[uav_index] = chosen_duration
            success[uav_index] = True

    return powers_by_uav, times_by_uav, success, chosen_mode


def _adaptive_group_duration(requests, gains, noise_power, bandwidth,
                             packet_size, fixed_power):
    """Return the group's common duration at fixed power, or None if unusable."""
    if not requests or fixed_power <= 0:
        return None
    if (not math.isfinite(noise_power) or noise_power <= 0
            or not math.isfinite(bandwidth) or bandwidth <= 0
            or not math.isfinite(packet_size) or packet_size <= 0):
        return None
    durations = []
    for index, (uav_index, device_index, _priority) in enumerate(requests):
        desired_gain = float(gains[uav_index, device_index])
        if not math.isfinite(desired_gain) or desired_gain <= 0:
            return None
        interference = noise_power
        for other_index, (_other_uav, other_device, _other_priority) in enumerate(requests):
            if other_index == index:
                continue
            cross_gain = float(gains[uav_index, other_device])
            if not math.isfinite(cross_gain) or cross_gain < 0:
                return None
            if cross_gain > 0:
                interference += fixed_power * cross_gain
                if not math.isfinite(interference):
                    return None
        sinr = (fixed_power * desired_gain) / interference
        if not math.isfinite(sinr) or sinr <= 0:
            return None
        rate = math.log1p(sinr) / math.log(2.0)
        duration = (packet_size / bandwidth) / rate
        if not math.isfinite(duration) or duration <= 0:
            return None
        durations.append(duration)
    return max(durations)


def schedule_adaptive(requests, gains, noise_power, bandwidth, packet_size,
                      fixed_power, power_cap, uav_count):
    """Place each priority-ranked request into a duration-minimizing serial group."""
    powers_by_uav = np.zeros(uav_count, dtype=float)
    times_by_uav = np.zeros(uav_count, dtype=float)
    success = np.zeros(uav_count, dtype=bool)
    if (not math.isfinite(fixed_power) or fixed_power <= 0
            or not math.isfinite(power_cap) or fixed_power > power_cap):
        return powers_by_uav, times_by_uav, success, "idle"

    groups = []
    group_durations = []
    for request in requests:
        choices = []
        for group_index, group in enumerate(groups):
            duration = _adaptive_group_duration(
                group + [request], gains, noise_power, bandwidth,
                packet_size, fixed_power)
            if duration is not None:
                updated_total = math.fsum(
                    duration if index == group_index else old_duration
                    for index, old_duration in enumerate(group_durations))
                if updated_total <= 1.0:
                    choices.append((updated_total, group_index, duration, group + [request]))

        duration = _adaptive_group_duration(
            [request], gains, noise_power, bandwidth, packet_size, fixed_power)
        if duration is not None:
            updated_total = math.fsum(group_durations + [duration])
            if updated_total <= 1.0:
                # Existing groups win exact ties; singleton is considered last.
                choices.append((updated_total, len(groups), duration, [request]))

        if not choices:
            continue
        _total, group_index, duration, selected_group = min(
            choices, key=lambda option: (option[0], option[1]))
        if group_index == len(groups):
            groups.append(selected_group)
            group_durations.append(duration)
        else:
            groups[group_index] = selected_group
            group_durations[group_index] = duration

    for group, duration in zip(groups, group_durations):
        for uav_index, _device_index, _priority in group:
            powers_by_uav[uav_index] = fixed_power
            times_by_uav[uav_index] = duration
            success[uav_index] = True
    if not np.any(success):
        scheme = "idle"
    elif len(groups) == 1 and len(groups[0]) == 1:
        scheme = "adaptive_single"
    elif len(groups) == 1:
        scheme = "adaptive_reuse"
    elif all(len(group) == 1 for group in groups):
        scheme = "adaptive_tdma"
    else:
        scheme = "adaptive_mixed"
    return powers_by_uav, times_by_uav, success, scheme


def _priority_signature(requests, success):
    """Encode the served priority prefix for deterministic equal-count tie breaks."""
    served_uavs = {request[0] for request in requests if success[request[0]]}
    return tuple(1 if request[0] in served_uavs else 0 for request in requests)


def schedule_subchannels(requests, gains, noise_power, bandwidth, packet_size,
                         power_cap, subchannel_count, uav_count):
    """Greedily assign requests to equal subbands and choose the best slot result."""
    empty_powers = np.zeros(uav_count, dtype=float)
    empty_times = np.zeros(uav_count, dtype=float)
    empty_success = np.zeros(uav_count, dtype=bool)
    if not requests or subchannel_count <= 0:
        return empty_powers, empty_times, empty_success, "idle"

    best_result = None
    max_channels = min(len(requests), int(subchannel_count))
    for channel_count in range(1, max_channels + 1):
        groups = [[] for _ in range(channel_count)]
        group_powers = [np.empty(0, dtype=float) for _ in range(channel_count)]

        for request in requests:
            choices = []
            subbandwidth = bandwidth / channel_count
            subnoise = noise_power / channel_count
            for channel_index, group in enumerate(groups):
                trial = group + [request]
                powers = _minimum_power_for_group(
                    trial, gains, subnoise, subbandwidth,
                    packet_size, 1.0, power_cap)
                if powers is None:
                    continue
                old_total = float(np.sum(group_powers[channel_index]))
                added_power = float(np.sum(powers)) - old_total
                if added_power < 0 and added_power > -1e-10:
                    added_power = 0.0
                choices.append((added_power, channel_index, powers, trial))

            if choices:
                _added_power, channel_index, powers, trial = min(
                    choices, key=lambda choice: (choice[0], choice[1]))
                groups[channel_index] = trial
                group_powers[channel_index] = powers

        powers_by_uav = np.zeros(uav_count, dtype=float)
        times_by_uav = np.zeros(uav_count, dtype=float)
        success = np.zeros(uav_count, dtype=bool)
        for group, group_powers_for_band in zip(groups, group_powers):
            for request, power in zip(group, group_powers_for_band):
                uav_index = request[0]
                powers_by_uav[uav_index] = power
                times_by_uav[uav_index] = 1.0
                success[uav_index] = True

        count = int(np.count_nonzero(success))
        signature = _priority_signature(requests, success)
        energy = float(np.sum(powers_by_uav * times_by_uav))
        candidate = (count, signature, energy, powers_by_uav, times_by_uav,
                     success, channel_count)
        if (best_result is None
                or count > best_result[0]
                or (count == best_result[0] and signature > best_result[1])
                or (count == best_result[0] and signature == best_result[1]
                    and energy < best_result[2])):
            best_result = candidate

    if best_result is None or best_result[0] == 0:
        return empty_powers, empty_times, empty_success, "idle"
    scheme = f"subchannels_{best_result[6]}"
    return best_result[3], best_result[4], best_result[5], scheme

