"""Train the MLP on states produced by a fixed drone-only policy."""

from __future__ import annotations

import json
import random
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from .result_reporting import _binary_metrics, save_prediction_reports


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


def _fit_mlp(
    tf: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict,
    class_weights: dict[int, float],
    seed: int,
    verbose: int = 0,
) -> tuple[Any, Any]:
    """Build and fit one deterministic MLP replicate."""
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    hidden_units = [int(value) for value in config["mlp"]["hidden_units"]]
    model = tf.keras.Sequential(name=f"mlp_drone_congestion_seed_{seed}")
    model.add(tf.keras.layers.Input(shape=(train_x.shape[1],), name="features"))
    for index, units in enumerate(hidden_units, start=1):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{index}"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="risk_probability"))
    optimizer_name = str(config["mlp"].get("optimizer", "adam")).lower()
    learning_rate = float(config["mlp"]["learning_rate"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported MLP optimiser: {optimizer_name}")
    model.compile(
        optimizer=optimizer,
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
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=int(config["mlp"]["max_epochs"]),
        batch_size=int(config["mlp"]["batch_size"]),
        class_weight=class_weights,
        callbacks=[callback],
        verbose=verbose,
    )
    return model, history


def _evaluation_row(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int8)
    return _binary_metrics(np.asarray(labels, dtype=np.int8), predictions, probabilities)


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
    train_array = train_x.to_numpy(np.float32)
    validation_array = validation_x.to_numpy(np.float32)
    test_array = test_x.to_numpy(np.float32)
    train_labels = train_y.to_numpy(np.int8)
    validation_labels = validation_y.to_numpy(np.int8)
    test_labels = test_y.to_numpy(np.int8)
    model, history = _fit_mlp(
        tf,
        train_array,
        train_labels.astype(np.float32),
        validation_array,
        validation_labels.astype(np.float32),
        config,
        class_weights,
        seed,
        verbose=2,
    )

    train_probability = model.predict(train_array, verbose=0).reshape(-1)
    validation_probability = model.predict(validation_array, verbose=0).reshape(-1)
    diagnostic_threshold, validation_f1 = _best_f1_threshold(
        validation_labels, validation_probability
    )
    operational_threshold = float(config["mlp"]["risk_threshold"])
    test_probability = model.predict(test_array, verbose=0).reshape(-1)
    output_directory = resolve_project_path(config["outputs"]["model_directory"], config)
    output_directory.mkdir(parents=True, exist_ok=True)
    model.save(output_directory / "mlp_congestion_aco.keras")
    joblib.dump(scaler, output_directory / "feature_scaler.pkl")
    history_frame = pd.DataFrame(history.history).rename_axis("epoch").reset_index()
    history_frame.to_csv(output_directory / "training_history.csv", index=False)

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
        "optimizer": str(config["mlp"].get("optimizer", "adam")),
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

    split_frames = []
    for split_name, labels, probabilities, split_rows in (
        ("train", train_labels, train_probability, splits["train"][2]),
        ("validation", validation_labels, validation_probability, splits["validation"][2]),
        ("test", test_labels, test_probability, splits["test"][2]),
    ):
        split_frames.append(pd.DataFrame({
            "split": split_name,
            "dock": split_rows["dock"].to_numpy(),
            "actual": labels,
            "probability": probabilities,
            "prediction_at_operational_threshold": (
                probabilities >= operational_threshold
            ).astype(np.int8),
        }))
    split_predictions = pd.concat(split_frames, ignore_index=True)
    split_predictions.to_csv(output_directory / "all_split_predictions.csv", index=False)

    evaluation_seeds = [int(value) for value in config["mlp"].get("evaluation_seeds", [seed])]
    evaluation_seeds = list(dict.fromkeys([seed, *evaluation_seeds]))
    seed_rows = [{"seed": seed, **_evaluation_row(test_labels, test_probability, operational_threshold)}]
    for evaluation_seed in evaluation_seeds:
        if evaluation_seed == seed:
            continue
        replicate, _ = _fit_mlp(
            tf,
            train_array,
            train_labels.astype(np.float32),
            validation_array,
            validation_labels.astype(np.float32),
            config,
            class_weights,
            evaluation_seed,
            verbose=0,
        )
        replicate_probability = replicate.predict(test_array, verbose=0).reshape(-1)
        seed_rows.append({
            "seed": evaluation_seed,
            **_evaluation_row(test_labels, replicate_probability, operational_threshold),
        })
    seed_metrics = pd.DataFrame(seed_rows)

    logistic = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
    )
    # Accelerate-backed NumPy can emit benign matmul overflow warnings inside
    # scikit-learn even for finite, scaled inputs; validate the output explicitly.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        logistic.fit(train_array.astype(np.float64), train_labels)
        logistic_probability = logistic.predict_proba(test_array.astype(np.float64))[:, 1]
    if not np.isfinite(logistic_probability).all():
        raise RuntimeError("Logistic-regression baseline produced non-finite probabilities.")
    numeric_indices = [features.index(value) for value in NUMERIC_FEATURES]
    reduced_model, _ = _fit_mlp(
        tf,
        train_array[:, numeric_indices],
        train_labels.astype(np.float32),
        validation_array[:, numeric_indices],
        validation_labels.astype(np.float32),
        config,
        class_weights,
        seed,
        verbose=0,
    )
    reduced_probability = reduced_model.predict(test_array[:, numeric_indices], verbose=0).reshape(-1)
    ablation_metrics = pd.DataFrame([
        {"model": "MLP full features", **_evaluation_row(test_labels, test_probability, operational_threshold)},
        {"model": "MLP without warehouse ID", **_evaluation_row(test_labels, reduced_probability, operational_threshold)},
        {"model": "Logistic regression", **_evaluation_row(test_labels, logistic_probability, operational_threshold)},
    ])

    fixed_threshold_metrics = save_prediction_reports(
        predictions=predictions,
        history=history_frame,
        operational_threshold=operational_threshold,
        congestion_threshold=float(config["mlp"]["congestion_threshold"]),
        output_directory=output_directory,
        split_predictions=split_predictions,
        diagnostic_threshold=diagnostic_threshold,
        seed_metrics=seed_metrics,
        ablation_metrics=ablation_metrics,
        false_negative_cost=float(config["mlp"].get("false_negative_cost", 5.0)),
        false_positive_cost=float(config["mlp"].get("false_positive_cost", 1.0)),
    )
    metrics = {
        "test_pr_auc": float(average_precision_score(test_y, test_probability)),
        "test_roc_auc": float(roc_auc_score(test_y, test_probability)),
        "test_precision_at_070": float(fixed_threshold_metrics["precision"]),
        "test_recall_at_070": float(fixed_threshold_metrics["recall"]),
        "test_f1_at_070": float(fixed_threshold_metrics["f1"]),
        "test_accuracy_at_070": float(fixed_threshold_metrics["accuracy"]),
        "test_balanced_accuracy_at_070": float(
            fixed_threshold_metrics["balanced_accuracy"]
        ),
        "test_brier_score": float(fixed_threshold_metrics["brier_score"]),
        "test_true_negative_at_070": int(fixed_threshold_metrics["true_negative"]),
        "test_false_positive_at_070": int(fixed_threshold_metrics["false_positive"]),
        "test_false_negative_at_070": int(fixed_threshold_metrics["false_negative"]),
        "test_true_positive_at_070": int(fixed_threshold_metrics["true_positive"]),
        "operational_threshold": operational_threshold,
        "validation_optimal_threshold_diagnostic": diagnostic_threshold,
        "validation_f1_at_diagnostic_threshold": validation_f1,
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "evaluation_seeds": evaluation_seeds,
    }
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics
