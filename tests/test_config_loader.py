from __future__ import annotations

from copy import deepcopy
import unittest

from src.config_loader import load_config, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_single_drone_configuration_is_required(self) -> None:
        config = deepcopy(load_config())
        config["drones"]["fleet"].append(
            {
                "id": 2,
                "payload_capacity": 10,
                "battery_max": 10,
                "energy_per_km": 1,
                "max_route_distance_km": 10,
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly one drone"):
            validate_config(config)

    def test_per_interval_battery_reset_is_required(self) -> None:
        config = deepcopy(load_config())
        config["simulation"]["battery_reset_policy"] = "daily"
        with self.assertRaisesRegex(ValueError, "per_interval"):
            validate_config(config)

    def test_maximum_route_distance_must_be_positive(self) -> None:
        config = deepcopy(load_config())
        config["drones"]["fleet"][0]["max_route_distance_km"] = 0
        with self.assertRaisesRegex(ValueError, "max_route_distance_km"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
