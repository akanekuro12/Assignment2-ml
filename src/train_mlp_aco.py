"""Train the MLP on states produced by a fixed drone-only policy."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config_loader import resolve_project_path, warehouse_capacities, warehouse_coordinates
from .data_pipeline import load_interval_table
from .domain import drones_from_config, euclidean_distance_matrix
from .feature_builder import (
    NUMERIC_FEATURES,
    build_model_table,
    chronological_masks,
    feature_names,
)
from .queue_simulator import simulate_historical_queue


def prepare_training_data(config: dict) -> tuple[pd.DataFrame, tuple[pd.Series, pd.Series, pd.Series]]:
    intervals = load_interval_table(config)
    capacities = warehouse_capacities(config)
    coordinates = warehouse_coordinates(config)
    distance_matrix, node_to_index, _ = euclidean_distance_matrix(coordinates)
    drones = drones_from_config(config)
    mlp = config["mlp"]
    objective = config["objective"]
    state_table = simulate_historical_queue(
        intervals=intervals,
        capacities=capacities,
        drones=drones,
        distance_matrix=distance_matrix,
        node_to_index=node_to_index,
        fill_threshold=float(mlp["congestion_threshold"]),
        unvisited_penalty=float(objective["unvisited_penalty"]),
        distance_weight=float(objective["distance_weight"]),
        central_node=int(config["simulation"]["central_node"]),
        battery_reset_policy=str(config["simulation"]["battery_reset_policy"]),
    )
    warehouses = [int(node) for node in config["data"]["warehouse_ids"]]
    model_table = build_model_table(
        state_table,
        capacities,
        warehouses,
        horizon_intervals=int(mlp["horizon_intervals"]),
        congestion_threshold=float(mlp["congestion_threshold"]),
    )
    masks = chronological_masks(
        model_table,
        train_end_day=int(mlp["train_end_day"]),
        validation_end_day=int(mlp["validation_end_day"]),
        horizon_intervals=int(mlp["horizon_intervals"]),
        interval_seconds=int(config["data"]["interval_seconds"]),
    )
    return model_table, masks


def _best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        raise RuntimeError("Validation predictions do not provide a usable threshold.")
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def train(config: dict) -> dict[str, Any]:
    import tensorflow as tf

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model_table, (train_mask, validation_mask, test_mask) = prepare_training_data(config)
    warehouses = [int(node) for node in config["data"]["warehouse_ids"]]
    features = feature_names(warehouses)
    splits = {}
    for name, mask in (
        ("train", train_mask), ("validation", validation_mask), ("test", test_mask)
    ):
        splits[name] = (
            model_table.loc[mask, features].copy(),
            model_table.loc[mask, "congestion"].astype(np.int8).copy(),
            model_table.loc[mask].copy(),
        )
        if splits[name][1].nunique() != 2:
            raise RuntimeError(f"{name} split must contain both target classes.")

    scaler = StandardScaler()
    train_x, train_y, _ = splits["train"]
    validation_x, validation_y, _ = splits["validation"]
    test_x, test_y, test_rows = splits["test"]
    # Convert before assignment so pandas does not place floats into integer columns.
    train_x[NUMERIC_FEATURES] = train_x[NUMERIC_FEATURES].astype(float)
    validation_x[NUMERIC_FEATURES] = validation_x[NUMERIC_FEATURES].astype(float)
    test_x[NUMERIC_FEATURES] = test_x[NUMERIC_FEATURES].astype(float)
    train_x[NUMERIC_FEATURES] = scaler.fit_transform(train_x[NUMERIC_FEATURES])
    validation_x[NUMERIC_FEATURES] = scaler.transform(validation_x[NUMERIC_FEATURES])
    test_x[NUMERIC_FEATURES] = scaler.transform(test_x[NUMERIC_FEATURES])

    counts = train_y.value_counts().sort_index()
    class_weights = {
        int(label): len(train_y) / (len(counts) * int(count)) for label, count in counts.items()
    }
    hidden_units = [int(value) for value in config["mlp"]["hidden_units"]]
    model = tf.keras.Sequential(name="mlp_drone_congestion")
    model.add(tf.keras.layers.Input(shape=(len(features),), name="features"))
    for index, units in enumerate(hidden_units, start=1):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{index}"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="risk_probability"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["mlp"]["learning_rate"])),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(config["mlp"]["early_stopping_patience"]),
        restore_best_weights=True,
    )
    history = model.fit(
        train_x.to_numpy(np.float32),
        train_y.to_numpy(np.float32),
        validation_data=(validation_x.to_numpy(np.float32), validation_y.to_numpy(np.float32)),
        epochs=int(config["mlp"]["max_epochs"]),
        batch_size=int(config["mlp"]["batch_size"]),
        class_weight=class_weights,
        callbacks=[callback],
        verbose=2,
    )

    validation_probability = model.predict(validation_x.to_numpy(np.float32), verbose=0).reshape(-1)
    diagnostic_threshold, validation_f1 = _best_f1_threshold(
        validation_y.to_numpy(np.int8), validation_probability
    )
    operational_threshold = float(config["mlp"]["risk_threshold"])
    test_probability = model.predict(test_x.to_numpy(np.float32), verbose=0).reshape(-1)
    output_directory = resolve_project_path(config["outputs"]["model_directory"], config)
    output_directory.mkdir(parents=True, exist_ok=True)
    model.save(output_directory / "mlp_congestion_aco.keras")
    joblib.dump(scaler, output_directory / "feature_scaler.pkl")
    pd.DataFrame(history.history).rename_axis("epoch").reset_index().to_csv(
        output_directory / "training_history.csv", index=False
    )

    configuration = {
        "feature_names": features,
        "numeric_feature_names": NUMERIC_FEATURES,
        "interval_duration_seconds": int(config["data"]["interval_seconds"]),
        "prediction_horizon_intervals": int(config["mlp"]["horizon_intervals"]),
        "warehouse_capacities": {str(k): int(v) for k, v in warehouse_capacities(config).items()},
        "congestion_threshold": float(config["mlp"]["congestion_threshold"]),
        "operational_prediction_threshold": operational_threshold,
        "validation_optimal_threshold_diagnostic": diagnostic_threshold,
        "validation_f1_at_diagnostic_threshold": validation_f1,
        "training_end_day": int(config["mlp"]["train_end_day"]),
        "validation_end_day": int(config["mlp"]["validation_end_day"]),
        "historical_queue_policy": str(config["simulation"]["historical_queue_policy"]),
        "queue_semantics": "arrivals_minus_single_drone_filtered_route_pickup_no_service_rate",
        "random_seed": seed,
    }
    with (output_directory / "model_configuration.json").open("w", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2, ensure_ascii=False)

    predictions = test_rows[
        ["time_bin", "day", "hour", "dock", "arrivals_15m", "mean_arrivals_60m",
         "available_before_pickup", "buffer_fill_ratio", "congestion"]
    ].copy()
    predictions["predicted_probability"] = test_probability
    predictions["predicted_congestion"] = (
        test_probability >= operational_threshold
    ).astype(np.int8)
    predictions.to_csv(output_directory / "test_predictions.csv", index=False)
    metrics = {
        "test_pr_auc": float(average_precision_score(test_y, test_probability)),
        "test_roc_auc": float(roc_auc_score(test_y, test_probability)),
        "operational_threshold": operational_threshold,
        "validation_optimal_threshold_diagnostic": diagnostic_threshold,
        "validation_f1_at_diagnostic_threshold": validation_f1,
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics
