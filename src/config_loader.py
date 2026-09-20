"""Load and validate the single experiment configuration file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def _int_key_dict(values: Dict[Any, Any], cast_value=float) -> Dict[int, Any]:
    return {int(key): cast_value(value) for key, value in values.items()}


def warehouse_capacities(config: Dict[str, Any]) -> Dict[int, int]:
    return _int_key_dict(config["warehouses"]["capacities"], int)


def warehouse_coordinates(config: Dict[str, Any]) -> Dict[int, tuple[float, float]]:
    raw = config["warehouses"]["coordinates_km"]
    return {int(key): (float(value[0]), float(value[1])) for key, value in raw.items()}


def validate_config(config: Dict[str, Any]) -> None:
    required = {
        "data", "mlp", "warehouses", "drones", "aco",
        "objective", "simulation", "outputs",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    warehouse_ids = [int(value) for value in config["data"]["warehouse_ids"]]
    capacities = warehouse_capacities(config)
    if set(warehouse_ids) != set(capacities):
        raise ValueError("warehouse_ids and warehouse capacity keys must match.")
    if any(value <= 0 for value in capacities.values()):
        raise ValueError("All warehouse capacities must be positive.")

    coordinates = warehouse_coordinates(config)
    central_node = int(config["simulation"]["central_node"])
    if central_node not in coordinates or not set(warehouse_ids).issubset(coordinates):
        raise ValueError("Coordinates must include the central node and every warehouse.")

    fleet = config["drones"]["fleet"]
    if len(fleet) != 1:
        raise ValueError("The single-drone experiment requires exactly one drone.")
    if len({int(item["id"]) for item in fleet}) != len(fleet):
        raise ValueError("Drone ids must be unique.")
    for item in fleet:
        if float(item["payload_capacity"]) <= 0 or float(item["battery_max"]) <= 0:
            raise ValueError("Drone payload and battery must be positive.")
        if float(item["energy_per_km"]) <= 0:
            raise ValueError("Drone energy_per_km must be positive.")
        if float(item["max_route_distance_km"]) <= 0:
            raise ValueError("Drone max_route_distance_km must be positive.")

    if int(config["mlp"]["horizon_intervals"]) <= 0:
        raise ValueError("MLP prediction horizon must be positive.")
    for name in ("congestion_threshold", "risk_threshold"):
        value = float(config["mlp"][name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"mlp.{name} must be in [0, 1].")
    evaluation_seeds = [int(value) for value in config["mlp"].get("evaluation_seeds", [])]
    if len(evaluation_seeds) != len(set(evaluation_seeds)):
        raise ValueError("mlp.evaluation_seeds must not contain duplicates.")
    for name in ("false_negative_cost", "false_positive_cost"):
        if float(config["mlp"].get(name, 0.0)) < 0:
            raise ValueError(f"mlp.{name} must be non-negative.")
    for threshold in config["simulation"].get("threshold_sweep", []):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("simulation.threshold_sweep values must be in [0, 1].")
    if str(config["simulation"]["battery_reset_policy"]) != "per_interval":
        raise ValueError("The single-drone experiment requires per_interval battery reset.")


def resolve_project_path(value: str | Path, config: Dict[str, Any] | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if config and config.get("_config_path"):
        candidate = Path(config["_config_path"]).resolve().parents[1] / path
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return PROJECT_ROOT / path
