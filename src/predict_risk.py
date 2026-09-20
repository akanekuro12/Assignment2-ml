"""Load a saved MLP bundle and expose deployment-time risk probabilities."""

from __future__ import annotations

import json
from pathlib import Path

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
        if "operational_prediction_threshold" in self.config:
            return float(self.config["operational_prediction_threshold"])
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
        # Calling the model directly avoids the large setup cost of model.predict
        # inside every 15-minute rolling-simulation interval.
        return np.asarray(self.model(values, training=False)).reshape(-1)

    def predict_by_warehouse(self, features: pd.DataFrame) -> dict[int, float]:
        probabilities = self.predict_frame(features)
        return {
            int(node): float(probability)
            for node, probability in zip(features.index, probabilities)
        }
