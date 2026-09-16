"""Shared routing calculations used by heuristics and ACO."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution, route_distance


EPSILON = 1e-12


def calculate_urgency(
    available: Mapping[int, float],
    capacities: Mapping[int, float],
    risk_probability: Mapping[int, float],
    probability_weight: float,
    fill_weight: float,
) -> dict[int, float]:
    return {
        int(node): float(
            probability_weight * float(risk_probability.get(node, 0.0))
            + fill_weight * min(1.0, float(available[node]) / float(capacities[node]))
        )
        for node in available
    }


def active_demands(
    available: Mapping[int, float],
    capacities: Mapping[int, float],
    risk_probability: Mapping[int, float],
    risk_threshold: float,
    fill_threshold: float,
    use_risk: bool,
) -> dict[int, float]:
    active: dict[int, float] = {}
    for node, amount in available.items():
        fill = float(amount) / float(capacities[node])
        risk_active = use_risk and float(risk_probability.get(node, 0.0)) >= risk_threshold
        if float(amount) > 0 and (risk_active or fill >= fill_threshold):
            active[int(node)] = float(amount)
    return active


def calculate_objective(
    remaining: Mapping[int, float],
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    energy_used: Mapping[int, float],
    drones: Sequence[DroneState],
    weight_remaining: float,
    weight_energy: float,
) -> float:
    weighted_demand = sum(float(urgency.get(node, 0.0)) * float(value) for node, value in demand.items())
    weighted_remaining = sum(
        float(urgency.get(node, 0.0)) * float(remaining.get(node, 0.0))
        for node in demand
    )
    remaining_norm = weighted_remaining / (weighted_demand + EPSILON)
    battery_total = sum(float(drone.battery_max) for drone in drones)
    energy_norm = sum(float(value) for value in energy_used.values()) / (battery_total + EPSILON)
    return float(weight_remaining * remaining_norm + weight_energy * energy_norm)


def idle_solution(
    drones: Sequence[DroneState], demand: Mapping[int, float], central_node: int = 0
) -> RoutingSolution:
    return RoutingSolution(
        routes={drone.id: [central_node, central_node] for drone in drones},
        pickups={drone.id: {} for drone in drones},
        route_distance={drone.id: 0.0 for drone in drones},
        energy_used={drone.id: 0.0 for drone in drones},
        remaining_goods={int(node): float(value) for node, value in demand.items()},
        objective_value=0.0 if not demand else 1.0,
    )


def finalize_solution(
    routes: dict[int, list[int]],
    pickups: dict[int, dict[int, float]],
    remaining: Mapping[int, float],
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    weight_remaining: float,
    weight_energy: float,
) -> RoutingSolution:
    distances = {
        drone.id: route_distance(routes[drone.id], distance_matrix, node_to_index)
        for drone in drones
    }
    energy = {drone.id: distances[drone.id] * drone.energy_per_km for drone in drones}
    objective = calculate_objective(
        remaining,
        demand,
        urgency,
        energy,
        drones,
        weight_remaining,
        weight_energy,
    )
    return RoutingSolution(
        routes=routes,
        pickups=pickups,
        route_distance=distances,
        energy_used=energy,
        remaining_goods={int(node): float(value) for node, value in remaining.items()},
        objective_value=objective,
    )
