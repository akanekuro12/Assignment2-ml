from __future__ import annotations

import unittest

import numpy as np

from src.aco_optimizer import optimize_aco
from src.domain import DroneState
from src.exact_solver import solve_exact_small_instance
from tests.test_aco_constraints import ACO_CONFIG


class ExactReferenceTests(unittest.TestCase):
    def test_aco_matches_exact_objective_on_small_full_service_instance(self) -> None:
        distance = np.asarray(
            [
                [0.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, 2.0, 3.0],
                [3.0, 2.0, 0.0, 2.0],
                [4.0, 3.0, 2.0, 0.0],
            ]
        )
        index = {0: 0, 1: 1, 2: 2, 3: 3}
        demand = {1: 5.0, 2: 5.0, 3: 5.0}
        drone = [DroneState(1, payload_capacity=20, battery_max=30, energy_per_km=1)]

        aco = optimize_aco(
            demand, drone, distance, index, ACO_CONFIG, seed=42
        )
        exact = solve_exact_small_instance(
            demand, drone, distance, index,
            unvisited_penalty=ACO_CONFIG["unvisited_penalty"],
            distance_weight=ACO_CONFIG["distance_weight"],
        )

        self.assertAlmostEqual(aco.total_pickup, 15.0)
        self.assertAlmostEqual(exact.total_pickup, 15.0)
        self.assertAlmostEqual(aco.objective_value, exact.objective_value)


if __name__ == "__main__":
    unittest.main()
