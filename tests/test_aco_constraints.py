from __future__ import annotations

import unittest

import numpy as np

from src.aco_optimizer import optimize_aco
from src.domain import DroneState
from src.solution_validator import validate_solution


ACO_CONFIG = {
    "ants": 8,
    "iterations": 15,
    "alpha": 1.0,
    "beta": 2.0,
    "evaporation": 0.1,
    "initial_pheromone": 1.0,
    "pheromone_min": 0.0001,
    "pheromone_max": 1000.0,
    "deposit_scale": 1.0,
    "stall_iterations": 5,
    "unvisited_penalty": 1000.0,
    "distance_weight": 1.0,
}


class ACOConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.distance = np.asarray(
            [
                [0.0, 3.0, 6.0],
                [3.0, 0.0, 4.0],
                [6.0, 4.0, 0.0],
            ]
        )
        self.index = {0: 0, 1: 1, 2: 2}

    def test_payload_and_return_battery_are_enforced(self) -> None:
        drones = [DroneState(1, payload_capacity=20, battery_max=10, energy_per_km=1)]
        demand = {1: 100.0, 2: 100.0}
        solution = optimize_aco(
            demand, drones, self.distance, self.index, ACO_CONFIG, seed=7,
        )
        self.assertLessEqual(solution.total_pickup, 20.0)
        self.assertNotIn(2, solution.routes[1])  # round trip requires 12 energy
        self.assertEqual(
            validate_solution(solution, drones, demand, self.distance, self.index), []
        )

    def test_insufficient_fleet_leaves_feasible_remaining_goods(self) -> None:
        drones = [DroneState(1, payload_capacity=5, battery_max=20, energy_per_km=1)]
        demand = {1: 20.0}
        solution = optimize_aco(
            demand, drones, self.distance, self.index, ACO_CONFIG, seed=3
        )
        self.assertGreater(solution.remaining_goods[1], 0)
        self.assertTrue(solution.feasible)

    def test_route_energy_matches_all_route_legs(self) -> None:
        drones = [DroneState(1, payload_capacity=20, battery_max=20, energy_per_km=2)]
        demand = {1: 5.0, 2: 5.0}
        solution = optimize_aco(
            demand, drones, self.distance, self.index, ACO_CONFIG, seed=9,
        )
        route = solution.routes[1]
        expected_distance = sum(
            self.distance[self.index[left], self.index[right]]
            for left, right in zip(route[:-1], route[1:])
        )
        self.assertAlmostEqual(solution.energy_used[1], 2.0 * expected_distance)

    def test_sufficient_fleet_collects_all_demand(self) -> None:
        drones = [DroneState(1, payload_capacity=20, battery_max=20, energy_per_km=1)]
        solution = optimize_aco(
            {1: 8.0}, drones, self.distance, self.index, ACO_CONFIG, seed=5,
        )
        self.assertAlmostEqual(solution.remaining_goods[1], 0.0)

    def test_explicit_maximum_route_distance_is_enforced(self) -> None:
        drones = [
            DroneState(
                1,
                payload_capacity=20,
                battery_max=100,
                energy_per_km=1,
                max_route_distance_km=10,
            )
        ]
        solution = optimize_aco(
            {1: 5.0, 2: 5.0}, drones, self.distance, self.index, ACO_CONFIG, seed=4
        )
        self.assertLessEqual(solution.route_distance[1], 10.0)
        self.assertNotIn(2, solution.routes[1])


if __name__ == "__main__":
    unittest.main()
