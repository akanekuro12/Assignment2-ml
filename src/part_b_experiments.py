"""Small validation experiments that support the design choices in report Part B."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .aco_optimizer import optimize_aco
from .config_loader import resolve_project_path
from .domain import DroneState, euclidean_distance_matrix
from .exact_solver import solve_exact_small_instance
from .feature_builder import NUMERIC_FEATURES, feature_names
from .result_reporting import _binary_metrics
from .train_mlp_aco import _fit_mlp, prepare_training_data


def _architecture_label(hidden_units: list[int]) -> str:
    return "-".join(str(value) for value in hidden_units)


def _prepare_validation_arrays(
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    model_table, (train_mask, validation_mask, _) = prepare_training_data(config)
    warehouses = [int(node) for node in config["data"]["warehouse_ids"]]
    features = feature_names(warehouses)
    train_x = model_table.loc[train_mask, features].copy()
    validation_x = model_table.loc[validation_mask, features].copy()
    train_y = model_table.loc[train_mask, "congestion"].to_numpy(np.int8)
    validation_y = model_table.loc[validation_mask, "congestion"].to_numpy(np.int8)

    train_x[NUMERIC_FEATURES] = train_x[NUMERIC_FEATURES].astype(float)
    validation_x[NUMERIC_FEATURES] = validation_x[NUMERIC_FEATURES].astype(float)
    scaler = StandardScaler()
    train_x[NUMERIC_FEATURES] = scaler.fit_transform(train_x[NUMERIC_FEATURES])
    validation_x[NUMERIC_FEATURES] = scaler.transform(validation_x[NUMERIC_FEATURES])
    counts = pd.Series(train_y).value_counts().sort_index()
    class_weights = {
        int(label): len(train_y) / (len(counts) * int(count))
        for label, count in counts.items()
    }
    return (
        train_x.to_numpy(np.float32),
        validation_x.to_numpy(np.float32),
        train_y,
        validation_y,
        class_weights,
    )


def run_architecture_comparison(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare small MLP architectures using validation data only."""
    import tensorflow as tf

    settings = config.get("part_b_experiments", {})
    candidates = [
        [int(value) for value in architecture]
        for architecture in settings.get("architectures", [[16], [32, 16], [64, 32]])
    ]
    seeds = [int(value) for value in settings.get("architecture_seeds", [42, 43, 44])]
    if not candidates or not seeds:
        raise ValueError("Part B architecture candidates and seeds must not be empty.")

    train_array, validation_array, train_y, validation_y, class_weights = (
        _prepare_validation_arrays(config)
    )
    threshold = float(config["mlp"]["risk_threshold"])
    rows: list[dict[str, Any]] = []
    for hidden_units in candidates:
        candidate_config = deepcopy(config)
        candidate_config["mlp"]["hidden_units"] = hidden_units
        label = _architecture_label(hidden_units)
        for seed in seeds:
            started = perf_counter()
            model, history = _fit_mlp(
                tf=tf,
                train_x=train_array,
                train_y=train_y.astype(np.float32),
                validation_x=validation_array,
                validation_y=validation_y.astype(np.float32),
                config=candidate_config,
                class_weights=class_weights,
                seed=seed,
                verbose=0,
            )
            runtime = perf_counter() - started
            probabilities = np.asarray(
                model(validation_array, training=False)
            ).reshape(-1)
            predictions = (probabilities >= threshold).astype(np.int8)
            metrics = _binary_metrics(validation_y, predictions, probabilities)
            val_loss = np.asarray(history.history["val_loss"], dtype=float)
            rows.append(
                {
                    "architecture": label,
                    "hidden_units": json.dumps(hidden_units),
                    "seed": seed,
                    "parameters": int(model.count_params()),
                    "epochs_ran": int(len(val_loss)),
                    "best_epoch": int(np.argmin(val_loss) + 1),
                    "best_validation_loss": float(np.min(val_loss)),
                    "runtime_seconds": float(runtime),
                    **metrics,
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["architecture", "parameters"], sort=False)
        .agg(
            runs=("seed", "size"),
            validation_pr_auc_mean=("pr_auc", "mean"),
            validation_pr_auc_std=("pr_auc", "std"),
            validation_f1_mean=("f1", "mean"),
            validation_f1_std=("f1", "std"),
            best_validation_loss_mean=("best_validation_loss", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    output = resolve_project_path(config["outputs"]["model_directory"], config)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "architecture_comparison_runs.csv", index=False)
    summary.to_csv(output / "architecture_comparison_summary.csv", index=False)
    _plot_architecture_comparison(summary, figures / "architecture_comparison.png")
    return runs, summary


def _plot_architecture_comparison(summary: pd.DataFrame, path: Path) -> None:
    labels = summary["architecture"].tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].errorbar(
        x - 0.06,
        summary["validation_pr_auc_mean"],
        yerr=summary["validation_pr_auc_std"].fillna(0),
        fmt="o",
        capsize=4,
        label="PR-AUC",
    )
    axes[0].errorbar(
        x + 0.06,
        summary["validation_f1_mean"],
        yerr=summary["validation_f1_std"].fillna(0),
        fmt="s",
        capsize=4,
        label="F1 at 0.70",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set(xlabel="Hidden units", ylabel="Validation score", title="Architecture validation results")
    axes[0].legend()
    axes[1].bar(x, summary["parameters"], color="steelblue")
    axes[1].set_xticks(x, labels)
    axes[1].set(xlabel="Hidden units", ylabel="Trainable parameters", title="Model size")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_optimizer_comparison(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Adam and SGD on the configured architecture using validation data."""
    import tensorflow as tf

    settings = config.get("part_b_experiments", {})
    candidates = settings.get(
        "optimizers",
        [
            {"name": "Adam", "optimizer": "adam", "learning_rate": 0.001},
            {"name": "SGD", "optimizer": "sgd", "learning_rate": 0.01},
        ],
    )
    seeds = [int(value) for value in settings.get("architecture_seeds", [42, 43, 44])]
    if not candidates or not seeds:
        raise ValueError("Part B optimiser candidates and seeds must not be empty.")
    train_array, validation_array, train_y, validation_y, class_weights = (
        _prepare_validation_arrays(config)
    )
    threshold = float(config["mlp"]["risk_threshold"])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_config = deepcopy(config)
        candidate_config["mlp"]["optimizer"] = str(candidate["optimizer"]).lower()
        candidate_config["mlp"]["learning_rate"] = float(candidate["learning_rate"])
        label = str(candidate["name"])
        for seed in seeds:
            started = perf_counter()
            model, history = _fit_mlp(
                tf=tf,
                train_x=train_array,
                train_y=train_y.astype(np.float32),
                validation_x=validation_array,
                validation_y=validation_y.astype(np.float32),
                config=candidate_config,
                class_weights=class_weights,
                seed=seed,
                verbose=0,
            )
            runtime = perf_counter() - started
            probabilities = np.asarray(model(validation_array, training=False)).reshape(-1)
            predictions = (probabilities >= threshold).astype(np.int8)
            metrics = _binary_metrics(validation_y, predictions, probabilities)
            val_loss = np.asarray(history.history["val_loss"], dtype=float)
            rows.append(
                {
                    "optimizer": label,
                    "learning_rate": float(candidate["learning_rate"]),
                    "seed": seed,
                    "epochs_ran": int(len(val_loss)),
                    "best_epoch": int(np.argmin(val_loss) + 1),
                    "best_validation_loss": float(np.min(val_loss)),
                    "runtime_seconds": float(runtime),
                    **metrics,
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["optimizer", "learning_rate"], sort=False)
        .agg(
            runs=("seed", "size"),
            validation_pr_auc_mean=("pr_auc", "mean"),
            validation_pr_auc_std=("pr_auc", "std"),
            validation_f1_mean=("f1", "mean"),
            validation_f1_std=("f1", "std"),
            best_validation_loss_mean=("best_validation_loss", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    output = resolve_project_path(config["outputs"]["model_directory"], config)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "optimizer_comparison_runs.csv", index=False)
    summary.to_csv(output / "optimizer_comparison_summary.csv", index=False)
    _plot_optimizer_comparison(summary, figures / "optimizer_comparison.png")
    return runs, summary


def _plot_optimizer_comparison(summary: pd.DataFrame, path: Path) -> None:
    labels = summary["optimizer"].tolist()
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.bar(
        x - width / 2,
        summary["validation_pr_auc_mean"],
        width,
        yerr=summary["validation_pr_auc_std"].fillna(0),
        capsize=4,
        label="PR-AUC",
    )
    axis.bar(
        x + width / 2,
        summary["validation_f1_mean"],
        width,
        yerr=summary["validation_f1_std"].fillna(0),
        capsize=4,
        label="F1 at 0.70",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.94, 1.002)
    axis.set(ylabel="Validation score", title="Optimiser comparison for the 32–16 MLP")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_synthetic_routing_scenarios(config: dict) -> list[dict[str, Any]]:
    """Create reproducible multi-active routing cases with two to four warehouses."""
    settings = config.get("part_b_experiments", {})
    cases_per_size = int(settings.get("synthetic_cases_per_size", 5))
    scenario_seed = int(settings.get("synthetic_scenario_seed", 2026))
    if cases_per_size <= 0:
        raise ValueError("synthetic_cases_per_size must be positive.")
    rng = np.random.default_rng(scenario_seed)
    scenarios: list[dict[str, Any]] = []
    scenario_index = 0
    for active_count in (2, 3, 4):
        for case_index in range(cases_per_size):
            scenario_index += 1
            route_limit = float((18.0, 22.0, 26.0, 30.0)[case_index % 4])
            coordinates: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
            for node in range(1, active_count + 1):
                angle = 2.0 * np.pi * (node - 1) / active_count + rng.uniform(-0.28, 0.28)
                radius = rng.uniform(2.5, min(8.5, route_limit / 2.0 - 0.25))
                coordinates[node] = (
                    float(radius * np.cos(angle)),
                    float(radius * np.sin(angle)),
                )
            goods = {
                node: float(rng.integers(5, 41)) for node in range(1, active_count + 1)
            }
            scenarios.append(
                {
                    "scenario_id": f"S{scenario_index:02d}",
                    "active_warehouses": active_count,
                    "case_index": case_index + 1,
                    "route_limit": route_limit,
                    "coordinates": coordinates,
                    "active_goods": goods,
                }
            )
    return scenarios


def run_synthetic_aco_benchmark(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare ACO with exact routing on deterministic multi-active scenarios."""
    settings = config.get("part_b_experiments", {})
    seeds = [
        int(value)
        for value in settings.get(
            "synthetic_aco_seeds", config["simulation"].get("aco_seeds", [1])
        )
    ]
    if not seeds:
        raise ValueError("Synthetic ACO seeds must not be empty.")
    objective = config["objective"]
    rows: list[dict[str, Any]] = []
    for scenario in build_synthetic_routing_scenarios(config):
        distance_matrix, node_to_index, _ = euclidean_distance_matrix(
            scenario["coordinates"]
        )
        drone = DroneState(
            id=1,
            payload_capacity=float(config["drones"]["fleet"][0]["payload_capacity"]),
            battery_max=float(scenario["route_limit"]),
            energy_per_km=1.0,
            max_route_distance_km=float(scenario["route_limit"]),
        )
        started = perf_counter()
        exact = solve_exact_small_instance(
            active_goods=scenario["active_goods"],
            drones=[drone.copy()],
            distance_matrix=distance_matrix,
            node_to_index=node_to_index,
            unvisited_penalty=float(objective["unvisited_penalty"]),
            distance_weight=float(objective["distance_weight"]),
        )
        exact_runtime = perf_counter() - started
        exact_route = exact.routes[1]
        for seed in seeds:
            started = perf_counter()
            aco = optimize_aco(
                active_goods=scenario["active_goods"],
                drones=[drone.copy()],
                distance_matrix=distance_matrix,
                node_to_index=node_to_index,
                aco_config=config["aco"],
                seed=seed,
                objective_config=objective,
            )
            aco_runtime = perf_counter() - started
            gap = float(aco.objective_value - exact.objective_value)
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "active_warehouses": int(scenario["active_warehouses"]),
                    "case_index": int(scenario["case_index"]),
                    "seed": seed,
                    "route_limit": float(scenario["route_limit"]),
                    "coordinates": json.dumps(scenario["coordinates"], sort_keys=True),
                    "active_goods": json.dumps(scenario["active_goods"], sort_keys=True),
                    "exact_route": "-".join(str(value) for value in exact_route),
                    "aco_route": "-".join(str(value) for value in aco.routes[1]),
                    "exact_objective": float(exact.objective_value),
                    "aco_objective": float(aco.objective_value),
                    "objective_gap": gap,
                    "relative_gap": gap / max(abs(float(exact.objective_value)), 1e-12),
                    "optimal_hit": bool(abs(gap) <= 1e-8),
                    "exact_runtime_seconds": float(exact_runtime),
                    "aco_runtime_seconds": float(aco_runtime),
                    "constraint_violations": int(len(aco.violations)),
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("active_warehouses", sort=True)
        .agg(
            scenarios=("scenario_id", "nunique"),
            aco_runs=("seed", "size"),
            optimal_hit_rate=("optimal_hit", "mean"),
            mean_objective_gap=("objective_gap", "mean"),
            maximum_objective_gap=("objective_gap", "max"),
            mean_relative_gap=("relative_gap", "mean"),
            mean_aco_runtime_seconds=("aco_runtime_seconds", "mean"),
            mean_exact_runtime_seconds=("exact_runtime_seconds", "mean"),
            constraint_violations=("constraint_violations", "sum"),
        )
        .reset_index()
    )

    output = resolve_project_path(config["outputs"]["experiment_directory"], config)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / "aco_exact_benchmark_runs.csv", index=False)
    summary.to_csv(tables / "aco_exact_benchmark_summary.csv", index=False)
    _plot_aco_exact_benchmark(summary, figures / "aco_exact_benchmark.png")
    return runs, summary


def _plot_aco_exact_benchmark(summary: pd.DataFrame, path: Path) -> None:
    labels = [str(value) for value in summary["active_warehouses"]]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].bar(x, summary["optimal_hit_rate"] * 100.0, color="seagreen")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 105)
    axes[0].set(
        xlabel="Active warehouses",
        ylabel="Optimal solutions found (\%)",
        title="ACO agreement with exact solver",
    )
    width = 0.36
    axes[1].bar(
        x - width / 2,
        summary["mean_exact_runtime_seconds"] * 1000.0,
        width,
        color="steelblue",
        label="Exact",
    )
    axes[1].bar(
        x + width / 2,
        summary["mean_aco_runtime_seconds"] * 1000.0,
        width,
        color="darkorange",
        label="ACO",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Active warehouses",
        ylabel="Mean runtime (ms, log scale)",
        title="Runtime on the synthetic cases",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_part_b_experiments(config: dict) -> dict[str, Any]:
    architecture_runs, architecture_summary = run_architecture_comparison(config)
    optimizer_runs, optimizer_summary = run_optimizer_comparison(config)
    routing_runs, routing_summary = run_synthetic_aco_benchmark(config)
    return {
        "architecture_runs": int(len(architecture_runs)),
        "architectures": architecture_summary.to_dict(orient="records"),
        "optimizer_runs": int(len(optimizer_runs)),
        "optimizers": optimizer_summary.to_dict(orient="records"),
        "routing_runs": int(len(routing_runs)),
        "routing_summary": routing_summary.to_dict(orient="records"),
    }
