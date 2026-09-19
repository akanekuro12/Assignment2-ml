from __future__ import annotations

import unittest

import numpy as np

from src.aco_optimizer import optimize_aco
from src.domain import DroneState
from tests.test_aco_constraints import ACO_CONFIG


class ACODeterminismTests(unittest.TestCase):
    def test_same_seed_produces_same_solution(self) -> None:
        distance = np.asarray(
            [[0.0, 2.0, 3.0], [2.0, 0.0, 2.0], [3.0, 2.0, 0.0]]
        )
        index = {0: 0, 1: 1, 2: 2}
        demand = {1: 8.0, 2: 8.0}
        def run():
            drones = [DroneState(1, 10, 20, 1)]
            return optimize_aco(
                demand, drones, distance, index, ACO_CONFIG, seed=42
            )

        first, second = run(), run()
        self.assertEqual(first.routes, second.routes)
        self.assertEqual(first.pickups, second.pickups)
        self.assertAlmostEqual(first.objective_value, second.objective_value)


if __name__ == "__main__":
    unittest.main()
