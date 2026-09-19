"""Exact constrained route reference for one drone and up to four warehouses."""

from __future__ import annotations

from itertools import permutations
from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import build_route_solution, idle_solution, route_is_feasible
from .solution_validator import validate_solution


def solve_exact_small_instance(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    unvisited_penalty: float = 1000.0,
    distance_weight: float = 1.0,
    central_node: int = 0,
) -> RoutingSolution:
    if not active_goods:
        return idle_solution(drones, active_goods, central_node)
    if len(drones) != 1:
        raise ValueError("The exact reference solver requires exactly one drone.")
    nodes = sorted(int(node) for node in active_goods)
    if len(nodes) > 4:
        raise ValueError("The exact reference solver is intentionally limited to four warehouses.")

    drone = drones[0]
    best = build_route_solution(
        route=[central_node, central_node],
        active_goods=active_goods,
        active_warehouses=nodes,
        drones=drones,
        distance_matrix=distance_matrix,
        node_to_index=node_to_index,
        unvisited_penalty=unvisited_penalty,
        distance_weight=distance_weight,
        central_node=central_node,
    )
    for length in range(1, len(nodes) + 1):
        for order in permutations(nodes, length):
            route = [central_node, *order, central_node]
            if not route_is_feasible(route, drone, distance_matrix, node_to_index):
                continue
            candidate = build_route_solution(
                route=route,
                active_goods=active_goods,
                active_warehouses=nodes,
                drones=drones,
                distance_matrix=distance_matrix,
                node_to_index=node_to_index,
                unvisited_penalty=unvisited_penalty,
                distance_weight=distance_weight,
                central_node=central_node,
            )
            if candidate.objective_value < best.objective_value:
                best = candidate

    validate_solution(best, drones, active_goods, distance_matrix, node_to_index, central_node)
    return best
