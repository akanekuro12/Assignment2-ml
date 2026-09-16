"""Load a saved MLP bundle and expose deployment-time risk probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd


class RiskPredictor:
    def __init__(self, model_directory: str | Path):
        import tensorflow as tf

        directory = Path(model_directory)
        self.model = tf.keras.models.load_model(directory / "mlp_congestion_aco.keras")
        self.scaler = joblib.load(directory / "feature_scaler.pkl")
        with (directory / "model_configuration.json").open("r", encoding="utf-8") as stream:
            self.config = json.load(stream)
        self.feature_names = list(self.config["feature_names"])
        self.numeric_feature_names = list(self.config["numeric_feature_names"])

    @property
    def classification_threshold(self) -> float:
        return float(self.config["selected_prediction_threshold"])

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names).difference(features.columns)
        if missing:
            raise ValueError(f"Prediction features are missing columns: {sorted(missing)}")
        transformed = features[self.feature_names].copy()
        transformed.loc[:, self.numeric_feature_names] = self.scaler.transform(
            transformed[self.numeric_feature_names]
        )
        values = transformed[self.feature_names].to_numpy(dtype=np.float32)
        return self.model.predict(values, verbose=0).reshape(-1)

    def predict_by_warehouse(self, features: pd.DataFrame) -> dict[int, float]:
        probabilities = self.predict_frame(features)
        return {
            int(node): float(probability)
            for node, probability in zip(features.index, probabilities)
        }


class ConstantRiskPredictor:
    """Small deterministic predictor used by tests and non-ML baselines."""

    def __init__(self, probability: float = 0.0):
        self.probability = float(probability)

    def predict_by_warehouse(self, features: pd.DataFrame) -> dict[int, float]:
        return {int(node): self.probability for node in features.index}
