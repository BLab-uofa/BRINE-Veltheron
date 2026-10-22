import numpy as np
from xgboost import XGBRegressor


class BootstrapXGBEnsemble:
    """
    Bootstrap ensemble of independent XGBRegressor models.

    Mean prediction:
        estimated residual

    Standard deviation:
        ensemble disagreement / practical model uncertainty
    """

    TARGET_SEED_OFFSET = {
        "sigma": 0,
        "her": 10000,
        "oer": 20000,
    }

    def __init__(self, config, target):
        self.config = config
        self.target = target.lower()

        if self.target not in self.TARGET_SEED_OFFSET:
            raise ValueError(f"Unknown target: {target}")

        self.models = []
        self.feature_names = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y lengths do not match.")

        if len(X) < self.config.xgb_min_training_rows:
            raise ValueError(
                f"Not enough training rows for {self.target}: "
                f"{len(X)} < {self.config.xgb_min_training_rows}"
            )

        self.feature_names = list(X.columns)
        self.models = []

        base_seed = (
            int(self.config.xgb_seed)
            + self.TARGET_SEED_OFFSET[self.target]
        )

        rng = np.random.default_rng(base_seed)
        n_rows = len(X)

        for model_index in range(self.config.xgb_ensemble_size):

            # Bootstrap sample: same size as training set,
            # sampled WITH replacement.
            boot_idx = rng.integers(
                low=0,
                high=n_rows,
                size=n_rows,
            )

            X_boot = X.iloc[boot_idx]
            y_boot = y.iloc[boot_idx]

            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                min_child_weight=self.config.xgb_min_child_weight,
                reg_alpha=self.config.xgb_reg_alpha,
                reg_lambda=self.config.xgb_reg_lambda,
                random_state=base_seed + model_index,
                tree_method="hist",
                n_jobs=self.config.xgb_n_jobs,
                verbosity=0,
            )

            model.fit(X_boot, y_boot)
            self.models.append(model)

        return self

    def predict(self, X):
        if not self.models:
            raise RuntimeError("Ensemble has not been fitted.")

        missing = [
            col for col in self.feature_names
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Prediction data missing features: {missing}"
            )

        X_use = X[self.feature_names]

        predictions = np.column_stack(
            [
                model.predict(X_use)
                for model in self.models
            ]
        )

        mean_prediction = predictions.mean(axis=1)
        std_prediction = predictions.std(axis=1, ddof=1)

        return mean_prediction, std_prediction