"""Deterministic routing baselines used for training state and evaluation."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import finalize_solution, idle_solution
from .solution_validator import validate_solution


def _construct_greedy(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    selector: Callable[[int, list[int]], int],
    weight_remaining: float,
    weight_energy: float,
    central_node: int = 0,
) -> RoutingSolution:
    if not demand:
        return idle_solution(drones, demand, central_node)
    remaining = {int(node): float(value) for node, value in demand.items()}
    routes: dict[int, list[int]] = {}
    pickups: dict[int, dict[int, float]] = {}

    for drone in sorted(drones, key=lambda item: item.id):
        current = central_node
        route = [central_node]
        drone_pickups: dict[int, float] = {}
        load = 0.0
        energy_to_current = 0.0
        visited: set[int] = set()

        while load < drone.payload_capacity - 1e-12:
            feasible: list[int] = []
            for node, amount in remaining.items():
                if amount <= 1e-12 or node in visited:
                    continue
                travel = distance_matrix[node_to_index[current], node_to_index[node]]
                return_distance = distance_matrix[node_to_index[node], node_to_index[central_node]]
                required = energy_to_current + drone.energy_per_km * (travel + return_distance)
                if required <= float(drone.available_battery) + 1e-12:
                    feasible.append(node)
            if not feasible:
                break
            node = int(selector(current, feasible))
            travel = distance_matrix[node_to_index[current], node_to_index[node]]
            energy_to_current += drone.energy_per_km * travel
            amount = min(remaining[node], drone.payload_capacity - load)
            drone_pickups[node] = drone_pickups.get(node, 0.0) + amount
            remaining[node] -= amount
            load += amount
            route.append(node)
            visited.add(node)
            current = node

        route.append(central_node)
        routes[drone.id] = route
        pickups[drone.id] = drone_pickups

    solution = finalize_solution(
        routes,
        pickups,
        remaining,
        demand,
        urgency,
        drones,
        distance_matrix,
        node_to_index,
        weight_remaining,
        weight_energy,
    )
    validate_solution(solution, drones, demand, distance_matrix, node_to_index, central_node)
    return solution


def nearest_first(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    weight_remaining: float = 0.7,
    weight_energy: float = 0.3,
    central_node: int = 0,
) -> RoutingSolution:
    def selector(current: int, feasible: list[int]) -> int:
        return min(
            feasible,
            key=lambda node: (distance_matrix[node_to_index[current], node_to_index[node]], node),
        )

    return _construct_greedy(
        demand, urgency, drones, distance_matrix, node_to_index, selector,
        weight_remaining, weight_energy, central_node,
    )


def fullest_first(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    capacities: Mapping[int, float],
    weight_remaining: float = 0.7,
    weight_energy: float = 0.3,
    central_node: int = 0,
) -> RoutingSolution:
    def selector(current: int, feasible: list[int]) -> int:
        return max(
            feasible,
            key=lambda node: (
                float(demand[node]) / float(capacities[node]),
                -distance_matrix[node_to_index[current], node_to_index[node]],
                -node,
            ),
        )

    return _construct_greedy(
        demand, urgency, drones, distance_matrix, node_to_index, selector,
        weight_remaining, weight_energy, central_node,
    )
