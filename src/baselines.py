"""Deterministic single-drone route baselines."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import EPSILON, build_route_solution, idle_solution
from .solution_validator import validate_solution


def _construct_greedy(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    selector: Callable[[int, list[int]], int],
    unvisited_penalty: float,
    distance_weight: float,
    central_node: int = 0,
) -> RoutingSolution:
    if not active_goods:
        return idle_solution(drones, active_goods, central_node)
    if len(drones) != 1:
        raise ValueError("Single-drone baselines require exactly one drone.")

    drone = drones[0]
    unvisited = set(int(node) for node in active_goods)
    current = central_node
    distance_to_current = 0.0
    route = [central_node]

    while unvisited:
        feasible: list[int] = []
        for node in sorted(unvisited):
            travel = distance_matrix[node_to_index[current], node_to_index[node]]
            return_distance = distance_matrix[node_to_index[node], node_to_index[central_node]]
            projected_distance = distance_to_current + travel + return_distance
            projected_energy = projected_distance * float(drone.energy_per_km)
            if (
                projected_distance <= float(drone.max_route_distance_km) + EPSILON
                and projected_energy <= float(drone.available_battery) + EPSILON
            ):
                feasible.append(node)
        if not feasible:
            break
        node = int(selector(current, feasible))
        distance_to_current += distance_matrix[node_to_index[current], node_to_index[node]]
        route.append(node)
        unvisited.remove(node)
        current = node

    route.append(central_node)
    solution = build_route_solution(
        route=route,
        active_goods=active_goods,
        active_warehouses=sorted(active_goods),
        drones=drones,
        distance_matrix=distance_matrix,
        node_to_index=node_to_index,
        unvisited_penalty=unvisited_penalty,
        distance_weight=distance_weight,
        central_node=central_node,
    )
    validate_solution(solution, drones, active_goods, distance_matrix, node_to_index, central_node)
    return solution


def nearest_first(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    unvisited_penalty: float = 1000.0,
    distance_weight: float = 1.0,
    central_node: int = 0,
) -> RoutingSolution:
    def selector(current: int, feasible: list[int]) -> int:
        return min(
            feasible,
            key=lambda node: (distance_matrix[node_to_index[current], node_to_index[node]], node),
        )

    return _construct_greedy(
        active_goods, drones, distance_matrix, node_to_index, selector,
        unvisited_penalty, distance_weight, central_node,
    )


def fullest_first(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    capacities: Mapping[int, float],
    unvisited_penalty: float = 1000.0,
    distance_weight: float = 1.0,
    central_node: int = 0,
) -> RoutingSolution:
    def selector(current: int, feasible: list[int]) -> int:
        return max(
            feasible,
            key=lambda node: (
                float(active_goods[node]) / float(capacities[node]),
                -distance_matrix[node_to_index[current], node_to_index[node]],
                -node,
            ),
        )

    return _construct_greedy(
        active_goods, drones, distance_matrix, node_to_index, selector,
        unvisited_penalty, distance_weight, central_node,
    )
