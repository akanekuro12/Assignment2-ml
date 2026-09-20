"""Tabular reports and static figures for prediction and routing results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


POLICY_LABELS = {
    "nearest_first": "Nearest-first",
    "fullest_first": "Fullest-first",
    "aco_current": "ACO-current",
    "mlp_aco": "MLP–ACO",
}


def _binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = np.unique(labels).size == 2
    result: dict[str, float | int] = {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "negative_samples": int(labels.size - labels.sum()),
        "predicted_positive": int(predictions.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if has_both_classes
            else float("nan")
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        result["pr_auc"] = (
            float(average_precision_score(labels, scores))
            if labels.sum() > 0
            else float("nan")
        )
        result["roc_auc"] = (
            float(roc_auc_score(labels, scores)) if has_both_classes else float("nan")
        )
    return result


def save_prediction_reports(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    operational_threshold: float,
    congestion_threshold: float,
    output_directory: str | Path,
) -> dict[str, float | int]:
    """Save fixed-threshold MLP tables and figures."""
    output = Path(output_directory)
    figure_directory = output / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    labels = predictions["congestion"].to_numpy(np.int8)
    probabilities = predictions["predicted_probability"].to_numpy(float)
    mlp_predictions = predictions["predicted_congestion"].to_numpy(np.int8)

    overall = _binary_metrics(labels, mlp_predictions, probabilities)
    pd.DataFrame([{"model": "mlp", "threshold": operational_threshold, **overall}]).to_csv(
        output / "classification_metrics_at_070.csv", index=False
    )

    matrix = confusion_matrix(labels, mlp_predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(output / "confusion_matrix_at_070.csv")

    warehouse_rows = []
    for warehouse, rows in predictions.groupby("dock", sort=True):
        warehouse_labels = rows["congestion"].to_numpy(np.int8)
        warehouse_probabilities = rows["predicted_probability"].to_numpy(float)
        warehouse_predictions = rows["predicted_congestion"].to_numpy(np.int8)
        warehouse_rows.append(
            {
                "warehouse": int(warehouse),
                "threshold": operational_threshold,
                **_binary_metrics(
                    warehouse_labels, warehouse_predictions, warehouse_probabilities
                ),
            }
        )
    warehouse_metrics = pd.DataFrame(warehouse_rows)
    warehouse_metrics.to_csv(output / "per_warehouse_classification_metrics.csv", index=False)

    baseline_rows = [
        {
            "method": "always_negative",
            "threshold": np.nan,
            **_binary_metrics(labels, np.zeros_like(labels)),
        },
        {
            "method": "current_fill_080",
            "threshold": congestion_threshold,
            **_binary_metrics(
                labels,
                predictions["buffer_fill_ratio"].ge(congestion_threshold).to_numpy(np.int8),
            ),
        },
        {
            "method": "mlp_probability_070",
            "threshold": operational_threshold,
            **overall,
        },
    ]
    pd.DataFrame(baseline_rows).to_csv(
        output / "prediction_baseline_comparison.csv", index=False
    )

    _plot_training_history(history, figure_directory / "training_history.png")
    _plot_precision_recall(
        labels,
        probabilities,
        operational_threshold,
        figure_directory / "precision_recall_curve.png",
    )
    _plot_confusion_matrix(matrix, figure_directory / "confusion_matrix_at_070.png")
    _plot_per_warehouse_metrics(
        warehouse_metrics, figure_directory / "per_warehouse_metrics.png"
    )
    _plot_prediction_timeline(
        predictions,
        operational_threshold,
        figure_directory / "prediction_probability_timeline.png",
    )
    return overall


def _plot_training_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history["epoch"] + 1, history["loss"], label="Train")
    axes[0].plot(history["epoch"] + 1, history["val_loss"], label="Validation")
    axes[0].set(title="Binary cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    metric = "pr_auc" if "pr_auc" in history else "precision"
    axes[1].plot(history["epoch"] + 1, history[metric], label="Train")
    axes[1].plot(history["epoch"] + 1, history[f"val_{metric}"], label="Validation")
    axes[1].set(title=metric.replace("_", " ").upper(), xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_precision_recall(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    predicted = probabilities >= threshold
    point_precision = precision_score(labels, predicted, zero_division=0)
    point_recall = recall_score(labels, predicted, zero_division=0)
    average_precision = average_precision_score(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    axis.plot(recall, precision, label=f"MLP (AP={average_precision:.3f})")
    axis.scatter(
        [point_recall], [point_precision], color="crimson", zorder=3,
        label=f"Threshold {threshold:.2f}",
    )
    axis.axhline(labels.mean(), color="gray", linestyle="--", label="Positive rate")
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision–recall curve")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    axis.set_title("Confusion matrix at threshold 0.70")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_per_warehouse_metrics(metrics: pd.DataFrame, path: Path) -> None:
    columns = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, column in enumerate(columns):
        axis.bar(x + (offset - 1) * width, metrics[column], width, label=column.title())
    axis.set_xticks(x, [f"Warehouse {value}" for value in metrics["warehouse"]])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("MLP performance by warehouse at threshold 0.70")
    for index, row in metrics.reset_index(drop=True).iterrows():
        if int(row["positive_samples"]) == 0:
            axis.axvspan(index - 0.42, index + 0.42, color="lightgray", alpha=0.35)
            axis.text(
                index, 0.48, "No positive\ntest labels", ha="center", va="center",
                color="dimgray", fontsize=9,
            )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_prediction_timeline(
    predictions: pd.DataFrame, threshold: float, path: Path
) -> None:
    warehouses = sorted(predictions["dock"].unique())
    figure, axes = plt.subplots(len(warehouses), 1, figsize=(12, 2.6 * len(warehouses)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, warehouse in zip(axes, warehouses):
        rows = predictions.loc[predictions["dock"].eq(warehouse)].sort_values("time_bin")
        axis.plot(rows["time_bin"], rows["predicted_probability"], linewidth=1, label="Probability")
        positive = rows.loc[rows["congestion"].eq(1)]
        axis.scatter(
            positive["time_bin"], np.ones(len(positive)), s=8, color="black",
            label="Actual congestion", zorder=3,
        )
        axis.axhline(threshold, color="crimson", linestyle="--", linewidth=1)
        axis.set_ylim(-0.03, 1.05)
        axis.set_ylabel(f"W{int(warehouse)}")
    axes[0].legend(loc="lower right", ncol=2)
    axes[-1].set_xlabel("Time bin")
    figure.suptitle("Predicted congestion probability on the test period")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_operational_reports(
    run_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_root: str | Path,
    drone_config: Mapping[str, float],
) -> None:
    """Save comparison, efficiency, constraint, and runtime figures."""
    root = Path(output_root)
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    order = [policy for policy in POLICY_LABELS if policy in set(run_metrics["policy"])]

    constraints = []
    for policy in order:
        rows = run_metrics.loc[run_metrics["policy"].eq(policy)]
        max_distance = float(rows["max_route_distance_interval"].max())
        max_energy = float(rows["max_energy_interval"].max())
        max_payload = float(rows["max_payload_interval"].max())
        constraints.append(
            {
                "policy": policy,
                "max_route_distance_observed": max_distance,
                "route_distance_limit": float(drone_config["max_route_distance_km"]),
                "max_energy_observed": max_energy,
                "battery_limit": float(drone_config["battery_max"]),
                "max_payload_observed": max_payload,
                "payload_limit": float(drone_config["payload_capacity"]),
                "constraint_violations": int(rows["constraint_violations"].sum()),
                "all_constraints_satisfied": bool(
                    max_distance <= float(drone_config["max_route_distance_km"]) + 1e-9
                    and max_energy <= float(drone_config["battery_max"]) + 1e-9
                    and max_payload <= float(drone_config["payload_capacity"]) + 1e-9
                    and int(rows["constraint_violations"].sum()) == 0
                ),
            }
        )
    constraint_table = pd.DataFrame(constraints)
    constraint_table.to_csv(tables / "constraint_summary.csv", index=False)

    key_columns = [
        "policy",
        "total_overflow_units_mean",
        "overflow_rate_mean",
        "total_collected_units_mean",
        "collection_ratio_mean",
        "total_route_distance_mean",
        "total_energy_used_mean",
        "number_of_sorties_mean",
        "active_visit_rate_mean",
        "max_active_warehouses_interval_max",
        "intervals_with_multiple_active_warehouses_mean",
        "constraint_violations_mean",
        "below_threshold_visited_mean",
        "algorithm_runtime_seconds_mean",
    ]
    summary[key_columns].to_csv(tables / "key_results.csv", index=False)

    _plot_operational_comparison(run_metrics, order, figures / "operational_comparison.png")
    _plot_efficiency(run_metrics, order, figures / "efficiency_comparison.png")
    _plot_constraints(
        constraint_table, order, figures / "constraint_usage.png"
    )
    _plot_runtime(run_metrics, order, figures / "algorithm_runtime.png")

    best_overflow = summary.sort_values("total_overflow_units_mean").iloc[0]
    mlp = summary.loc[summary["policy"].eq("mlp_aco")].iloc[0]
    aco_current = summary.loc[summary["policy"].eq("aco_current")].iloc[0]
    overflow_reduction = float(
        aco_current["total_overflow_units_mean"] - mlp["total_overflow_units_mean"]
    )
    manifest = {
        "runs": int(len(run_metrics)),
        "policies": order,
        "best_mean_overflow_policy": str(best_overflow["policy"]),
        "best_mean_overflow_units": float(best_overflow["total_overflow_units_mean"]),
        "mlp_aco_overflow_reduction_vs_aco_current_units": overflow_reduction,
        "mlp_aco_overflow_reduction_vs_aco_current_percent": float(
            100.0 * overflow_reduction / aco_current["total_overflow_units_mean"]
        ),
        "mlp_aco_collection_gain_vs_aco_current_units": float(
            mlp["total_collected_units_mean"] - aco_current["total_collected_units_mean"]
        ),
        "mlp_aco_distance_increase_vs_aco_current": float(
            mlp["total_route_distance_mean"] - aco_current["total_route_distance_mean"]
        ),
        "maximum_active_warehouses_in_any_interval": int(
            run_metrics["max_active_warehouses_interval"].max()
        ),
        "all_constraints_satisfied": bool(constraint_table["all_constraints_satisfied"].all()),
    }
    with (root / "results_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)


def _metric_mean_std(
    metrics: pd.DataFrame, order: list[str], column: str
) -> tuple[np.ndarray, np.ndarray]:
    grouped = metrics.groupby("policy")[column]
    mean = np.asarray([grouped.mean().get(policy, np.nan) for policy in order], dtype=float)
    std = np.asarray([grouped.std().get(policy, np.nan) for policy in order], dtype=float)
    return mean, np.nan_to_num(std)


def _plot_operational_comparison(metrics: pd.DataFrame, order: list[str], path: Path) -> None:
    specifications = [
        ("total_overflow_units", "Total overflow"),
        ("total_collected_units", "Collected goods"),
        ("total_route_distance", "Route distance"),
        ("total_energy_used", "Energy used"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [POLICY_LABELS[value] for value in order]
    for axis, (column, title) in zip(axes.flat, specifications):
        mean, std = _metric_mean_std(metrics, order, column)
        axis.bar(labels, mean, yerr=std, capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Operational comparison over the full test period")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_efficiency(metrics: pd.DataFrame, order: list[str], path: Path) -> None:
    specifications = [
        ("collection_ratio", "Collection ratio"),
        ("energy_per_collected_unit", "Energy / collected unit"),
        ("active_visit_rate", "Active warehouse visit rate"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = [POLICY_LABELS[value] for value in order]
    for axis, (column, title) in zip(axes, specifications):
        mean, std = _metric_mean_std(metrics, order, column)
        axis.bar(labels, mean, yerr=std, capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_constraints(table: pd.DataFrame, order: list[str], path: Path) -> None:
    specifications = [
        ("max_route_distance_observed", "route_distance_limit", "Maximum route distance"),
        ("max_energy_observed", "battery_limit", "Maximum energy"),
        ("max_payload_observed", "payload_limit", "Maximum payload"),
    ]
    labels = [POLICY_LABELS[value] for value in order]
    indexed = table.set_index("policy").loc[order]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (observed, limit, title) in zip(axes, specifications):
        axis.bar(labels, indexed[observed])
        axis.axhline(indexed[limit].iloc[0], color="crimson", linestyle="--", label="Limit")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_runtime(metrics: pd.DataFrame, order: list[str], path: Path) -> None:
    mean, std = _metric_mean_std(metrics, order, "algorithm_runtime_seconds")
    labels = [POLICY_LABELS[value] for value in order]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, mean, yerr=std, capsize=4)
    axis.set(title="Total routing algorithm runtime", ylabel="Seconds")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_route_example(
    route_log: pd.DataFrame,
    coordinates: Mapping[int, tuple[float, float]],
    path: str | Path,
) -> None:
    """Plot the MLP–ACO route that visits the largest number of warehouses."""
    if route_log.empty:
        return
    row = route_log.sort_values(
        ["visited_warehouse_count", "route_distance"], ascending=[False, False]
    ).iloc[0]
    route = [int(node) for node in json.loads(row["route"])]
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    for node, (x, y) in coordinates.items():
        color = "black" if int(node) == 0 else "steelblue"
        axis.scatter([x], [y], s=110, color=color, zorder=3)
        axis.text(x + 0.18, y + 0.18, "Depot" if int(node) == 0 else f"W{node}")
    for left, right in zip(route[:-1], route[1:]):
        x1, y1 = coordinates[left]
        x2, y2 = coordinates[right]
        axis.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.8},
        )
    axis.set(
        title=f"MLP–ACO route example: {route} ({float(row['route_distance']):.2f} km)",
        xlabel="x (km)", ylabel="y (km)",
    )
    axis.grid(alpha=0.25)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180, bbox_inches="tight")
    plt.close(figure)
