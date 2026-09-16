"""Multi-drone Ant Colony Optimisation with payload and return-battery checks."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import EPSILON, finalize_solution, idle_solution
from .solution_validator import validate_solution


def _construct_ant_solution(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    pheromone: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    weight_remaining: float,
    weight_energy: float,
    central_node: int,
) -> RoutingSolution:
    remaining = {int(node): float(value) for node, value in demand.items()}
    routes: dict[int, list[int]] = {}
    pickups: dict[int, dict[int, float]] = {}

    for drone_index in rng.permutation(len(drones)):
        drone = drones[int(drone_index)]
        current = central_node
        route = [central_node]
        drone_pickups: dict[int, float] = {}
        load = 0.0
        energy_to_current = 0.0
        visited: set[int] = set()

        while load < drone.payload_capacity - EPSILON:
            feasible: list[int] = []
            for node, amount in remaining.items():
                if amount <= EPSILON or node in visited:
                    continue
                travel = distance_matrix[node_to_index[current], node_to_index[node]]
                return_distance = distance_matrix[node_to_index[node], node_to_index[central_node]]
                required = energy_to_current + drone.energy_per_km * (travel + return_distance)
                if required <= float(drone.available_battery) + EPSILON:
                    feasible.append(node)
            if not feasible:
                break

            scores = []
            for node in feasible:
                distance = distance_matrix[node_to_index[current], node_to_index[node]]
                eta = (EPSILON + float(urgency.get(node, 0.0))) / (EPSILON + distance)
                tau = pheromone[node_to_index[current], node_to_index[node]]
                scores.append(max(EPSILON, tau) ** alpha * max(EPSILON, eta) ** beta)
            probabilities = np.asarray(scores, dtype=float)
            probabilities /= probabilities.sum()
            node = int(rng.choice(np.asarray(feasible, dtype=int), p=probabilities))

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

    # Drone iteration is random, but the returned mappings always contain the full fleet.
    for drone in drones:
        routes.setdefault(drone.id, [central_node, central_node])
        pickups.setdefault(drone.id, {})
    return finalize_solution(
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


def optimize_aco(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    aco_config: Mapping,
    seed: int,
    central_node: int = 0,
) -> RoutingSolution:
    if not demand:
        return idle_solution(drones, demand, central_node)

    rng = np.random.default_rng(seed)
    pheromone = np.full(distance_matrix.shape, float(aco_config["initial_pheromone"]), dtype=float)
    np.fill_diagonal(pheromone, 0.0)
    global_best: RoutingSolution | None = None
    stalled = 0

    for _ in range(int(aco_config["iterations"])):
        iteration_best: RoutingSolution | None = None
        for _ in range(int(aco_config["ants"])):
            candidate = _construct_ant_solution(
                demand=demand,
                urgency=urgency,
                drones=drones,
                distance_matrix=distance_matrix,
                node_to_index=node_to_index,
                pheromone=pheromone,
                rng=rng,
                alpha=float(aco_config["alpha"]),
                beta=float(aco_config["beta"]),
                weight_remaining=float(aco_config["weight_remaining"]),
                weight_energy=float(aco_config["weight_energy"]),
                central_node=central_node,
            )
            validate_solution(
                candidate, drones, demand, distance_matrix, node_to_index, central_node
            )
            if iteration_best is None or candidate.objective_value < iteration_best.objective_value:
                iteration_best = candidate

        assert iteration_best is not None
        improved = global_best is None or iteration_best.objective_value < global_best.objective_value - 1e-12
        if improved:
            global_best = iteration_best
            stalled = 0
        else:
            stalled += 1

        evaporation = float(aco_config["evaporation"])
        pheromone *= 1.0 - evaporation
        assert global_best is not None
        deposit = float(aco_config["deposit_scale"]) / (global_best.objective_value + EPSILON)
        for route in global_best.routes.values():
            for left, right in zip(route[:-1], route[1:]):
                if left == right:
                    continue
                left_index, right_index = node_to_index[left], node_to_index[right]
                pheromone[left_index, right_index] += deposit
                pheromone[right_index, left_index] += deposit
        pheromone = np.clip(
            pheromone,
            float(aco_config["pheromone_min"]),
            float(aco_config["pheromone_max"]),
        )
        np.fill_diagonal(pheromone, 0.0)
        if stalled >= int(aco_config["stall_iterations"]):
            break

    assert global_best is not None
    validate_solution(global_best, drones, demand, distance_matrix, node_to_index, central_node)
    return global_best
