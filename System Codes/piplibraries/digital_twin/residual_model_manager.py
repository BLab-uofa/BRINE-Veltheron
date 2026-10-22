import numpy as np
import pandas as pd

from .xgb_ensemble import BootstrapXGBEnsemble


class ResidualModelManager:
    """Fit selected residual models, then predict only their unmeasured rows."""

    TARGETS = {
        "sigma": {"baseline": "dt_sigma_theory_S_m", "residual_prediction": "dt_sigma_residual_pred_S_m",
                  "uncertainty": "dt_sigma_uncertainty_S_m", "final_prediction": "dt_sigma_pred_S_m",
                  "mode": "dt_sigma_model_mode", "n_train": "dt_sigma_training_rows"},
        "her": {"baseline": "dt_E_HER_nernst_vs_SHE_V", "residual_prediction": "dt_HER_residual_pred_V",
                "uncertainty": "dt_HER_uncertainty_V", "final_prediction": "dt_HER_pred_vs_SHE_V",
                "mode": "dt_HER_model_mode", "n_train": "dt_HER_training_rows"},
        "oer": {"baseline": "dt_E_OER_nernst_vs_SHE_V", "residual_prediction": "dt_OER_residual_pred_V",
                "uncertainty": "dt_OER_uncertainty_V", "final_prediction": "dt_OER_pred_vs_SHE_V",
                "mode": "dt_OER_model_mode", "n_train": "dt_OER_training_rows"},
    }

    def __init__(self, config, training_table_builder):
        self.config = config
        self.training_table_builder = training_table_builder
        self.models = {}
        self.summary = {}

    def run(self, df):
        df = df.copy()
        self.models = {}
        self.summary = {}
        for target in self.config.objective_targets:
            df = self._run_target(df, target)
        return df

    def _run_target(self, df, target):
        info = self.TARGETS[target]
        for key in ("residual_prediction", "uncertainty", "final_prediction"):
            df[info[key]] = np.nan
        df[info["mode"]] = "unusable_features"

        training = self.training_table_builder.build_training(df, target)
        prediction = self.training_table_builder.build_prediction(df, target)
        n_train = len(training["X"])
        pred_rows = prediction["row_index"]
        df[info["n_train"]] = n_train
        df.loc[training["row_index"], info["mode"]] = "training_history"

        # Fit even when every row is completed. Pending rows control prediction only.
        ensemble = None
        if n_train >= self.config.xgb_min_training_rows:
            ensemble = BootstrapXGBEnsemble(self.config, target)
            ensemble.fit(training["X"], training["y"])
            self.models[target] = ensemble

        measured_col = self.config.objective_column(target)
        self.summary[target] = {
            "measurement_column": measured_col,
            "n_labelled": int(pd.to_numeric(df[measured_col], errors="coerce").notna().sum()),
            "n_training_rows": n_train,
            "n_prediction_rows": len(pred_rows),
            "model_fitted": ensemble is not None,
        }
        print(f"[digital_twin] {target} <- {measured_col}: {n_train} usable training rows, "
              f"fitted={ensemble is not None}, {len(pred_rows)} prediction rows.")
        if not pred_rows:
            return df

        if ensemble is None:
            residual_mean = np.zeros(len(pred_rows))
            residual_std = np.full(len(pred_rows), np.nan)
            mode = "theory_only"
        else:
            residual_mean, residual_std = ensemble.predict(prediction["X"])
            mode = "xgb_residual_ensemble"

        baseline = df.loc[pred_rows, info["baseline"]].to_numpy(dtype=float)
        df.loc[pred_rows, info["residual_prediction"]] = residual_mean
        df.loc[pred_rows, info["uncertainty"]] = residual_std
        df.loc[pred_rows, info["final_prediction"]] = baseline + residual_mean
        df.loc[pred_rows, info["mode"]] = mode
        return df
