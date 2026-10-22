import pandas as pd
import numpy as np


class TrainingTableBuilder:
    """
    Builds clean X/y tables for each independent DT residual model.
    """

    TARGET_INFO = {

        "sigma": {
            "residual": "dt_residual_sigma_S_m",
        },

        "her": {
            "residual": "dt_residual_HER_V",
        },

        "oer": {
            "residual": "dt_residual_OER_V",
        },
    }

    def __init__(self, config, feature_schema):
        self.config = config
        self.feature_schema = feature_schema

    def build_training(self, df, target):
        target = target.lower()

        if target not in self.TARGET_INFO:
            raise ValueError(f"Unknown target: {target}")

        feature_cols = self.feature_schema.for_target(target)
        residual_col = self.TARGET_INFO[target]["residual"]

        numeric = df[feature_cols + [residual_col]].apply(pd.to_numeric, errors="coerce")
        usable = np.isfinite(numeric).all(axis=1)

        X = df.loc[usable, feature_cols].astype(float).copy()
        y = df.loc[usable, residual_col].astype(float).copy()

        return {
            "X": X,
            "y": y,
            "row_index": df.index[usable].tolist(),
            "features": feature_cols,
            "target": residual_col,
        }

    def build_prediction(self, df, target):
        """
        Returns rows whose corresponding experimental objective
        is still missing but whose DT features are available.
        """

        target = target.lower()

        if target not in self.config.objective_targets:
            raise ValueError(f"{target} is not an objective in this campaign")

        feature_cols = self.feature_schema.for_target(target)


        measured_col = self.config.objective_column(target)
        pending = pd.to_numeric(df[measured_col], errors="coerce").isna()
        features_ready = np.isfinite(df[feature_cols].apply(pd.to_numeric, errors="coerce")).all(axis=1)

        mask = pending & features_ready

        X = df.loc[mask, feature_cols].astype(float).copy()

        return {
            "X": X,
            "row_index": df.index[mask].tolist(),
            "features": feature_cols,
        }
