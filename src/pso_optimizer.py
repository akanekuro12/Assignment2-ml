"""Random-key Particle Swarm Optimisation for multi-drone routing."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import EPSILON, finalize_solution, idle_solution
from .solution_validator import validate_solution


def _decode_particle(
    position: np.ndarray,
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    weight_remaining: float,
    weight_energy: float,
    central_node: int,
) -> RoutingSolution:
    """Convert continuous random keys into feasible routes and pickups.

    Encoding:
      - D keys choose drone order.
      - D keys choose a visit limit from zero to W for each drone.
      - D x W keys rank warehouses independently for every drone.
    """
    nodes = sorted(int(node) for node in demand)
    drone_count, warehouse_count = len(drones), len(nodes)
    order_keys = position[:drone_count]
    limit_keys = position[drone_count:2 * drone_count]
    priority_keys = position[2 * drone_count:].reshape(drone_count, warehouse_count)
    remaining = {node: float(demand[node]) for node in nodes}
    routes: dict[int, list[int]] = {}
    pickups: dict[int, dict[int, float]] = {}

    for drone_index in np.argsort(order_keys, kind="stable"):
        drone = drones[int(drone_index)]
        visit_limit = min(
            warehouse_count,
            int(np.floor(float(limit_keys[drone_index]) * (warehouse_count + 1))),
        )
        ranked_node_indices = np.argsort(-priority_keys[drone_index], kind="stable")
        current = central_node
        route = [central_node]
        drone_pickups: dict[int, float] = {}
        energy_to_current = 0.0
        load = 0.0
        visits = 0

        for node_index in ranked_node_indices:
            if visits >= visit_limit or load >= drone.payload_capacity - EPSILON:
                break
            node = nodes[int(node_index)]
            if remaining[node] <= EPSILON:
                continue
            travel = distance_matrix[node_to_index[current], node_to_index[node]]
            return_distance = distance_matrix[node_to_index[node], node_to_index[central_node]]
            required = energy_to_current + drone.energy_per_km * (travel + return_distance)
            if required > float(drone.available_battery) + EPSILON:
                continue
            amount = min(remaining[node], drone.payload_capacity - load)
            energy_to_current += drone.energy_per_km * travel
            drone_pickups[node] = drone_pickups.get(node, 0.0) + amount
            remaining[node] -= amount
            load += amount
            visits += 1
            route.append(node)
            current = node

        route.append(central_node)
        routes[drone.id] = route
        pickups[drone.id] = drone_pickups

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


def optimize_pso(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    pso_config: Mapping,
    objective_config: Mapping,
    seed: int,
    central_node: int = 0,
) -> RoutingSolution:
    """Minimize the same normalized operational objective used by ACO."""
    if not demand:
        return idle_solution(drones, demand, central_node)

    rng = np.random.default_rng(seed)
    particle_count = int(pso_config["particles"])
    iterations = int(pso_config["iterations"])
    dimensions = 2 * len(drones) + len(drones) * len(demand)
    positions = rng.random((particle_count, dimensions))
    velocity_limit = float(pso_config["velocity_limit"])
    velocities = rng.uniform(-velocity_limit, velocity_limit, size=(particle_count, dimensions))
    personal_best_positions = positions.copy()
    personal_best_values = np.full(particle_count, np.inf)
    global_best_position: np.ndarray | None = None
    global_best_solution: RoutingSolution | None = None
    stalled = 0

    def evaluate(position: np.ndarray) -> RoutingSolution:
        solution = _decode_particle(
            position=position,
            demand=demand,
            urgency=urgency,
            drones=drones,
            distance_matrix=distance_matrix,
            node_to_index=node_to_index,
            weight_remaining=float(objective_config["weight_remaining"]),
            weight_energy=float(objective_config["weight_energy"]),
            central_node=central_node,
        )
        validate_solution(
            solution, drones, demand, distance_matrix, node_to_index, central_node
        )
        return solution

    for _ in range(iterations):
        improved_global = False
        for particle_index in range(particle_count):
            solution = evaluate(positions[particle_index])
            value = solution.objective_value
            if value < personal_best_values[particle_index] - EPSILON:
                personal_best_values[particle_index] = value
                personal_best_positions[particle_index] = positions[particle_index].copy()
            if global_best_solution is None or value < global_best_solution.objective_value - EPSILON:
                global_best_solution = solution
                global_best_position = positions[particle_index].copy()
                improved_global = True

        if improved_global:
            stalled = 0
        else:
            stalled += 1
        if stalled >= int(pso_config["stall_iterations"]):
            break

        assert global_best_position is not None
        random_personal = rng.random((particle_count, dimensions))
        random_global = rng.random((particle_count, dimensions))
        velocities = (
            float(pso_config["inertia"]) * velocities
            + float(pso_config["cognitive"]) * random_personal
            * (personal_best_positions - positions)
            + float(pso_config["social"]) * random_global
            * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions = np.clip(positions + velocities, 0.0, 1.0)

    assert global_best_solution is not None
    validate_solution(
        global_best_solution, drones, demand, distance_matrix, node_to_index, central_node
    )
    return global_best_solution
