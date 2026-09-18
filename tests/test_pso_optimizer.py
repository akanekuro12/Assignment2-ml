from __future__ import annotations

import unittest

import numpy as np

from src.domain import DroneState
from src.pso_optimizer import optimize_pso
from src.solution_validator import validate_solution


PSO_CONFIG = {
    "particles": 16,
    "iterations": 30,
    "inertia": 0.7,
    "cognitive": 1.5,
    "social": 1.5,
    "velocity_limit": 0.2,
    "stall_iterations": 8,
}
OBJECTIVE_CONFIG = {"weight_remaining": 0.7, "weight_energy": 0.3}


class PSOOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.distance = np.asarray(
            [[0.0, 3.0, 6.0], [3.0, 0.0, 4.0], [6.0, 4.0, 0.0]]
        )
        self.index = {0: 0, 1: 1, 2: 2}

    def _run(self, seed: int):
        drones = [DroneState(1, 10, 10, 1), DroneState(2, 10, 10, 1)]
        solution = optimize_pso(
            demand={1: 8.0, 2: 8.0},
            urgency={1: 1.0, 2: 0.8},
            drones=drones,
            distance_matrix=self.distance,
            node_to_index=self.index,
            pso_config=PSO_CONFIG,
            objective_config=OBJECTIVE_CONFIG,
            seed=seed,
        )
        return drones, solution

    def test_pso_solution_satisfies_payload_and_return_battery(self) -> None:
        drones, solution = self._run(4)
        self.assertNotIn(2, solution.routes[1])
        self.assertNotIn(2, solution.routes[2])
        self.assertEqual(
            validate_solution(solution, drones, {1: 8.0, 2: 8.0}, self.distance, self.index),
            [],
        )

    def test_same_seed_is_deterministic(self) -> None:
        _, first = self._run(42)
        _, second = self._run(42)
        self.assertEqual(first.routes, second.routes)
        self.assertEqual(first.pickups, second.pickups)
        self.assertAlmostEqual(first.objective_value, second.objective_value)


if __name__ == "__main__":
    unittest.main()
