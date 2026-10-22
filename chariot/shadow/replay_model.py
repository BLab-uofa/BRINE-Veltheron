from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np


@dataclass(frozen=True)
class ReplayPrediction:
    mean: float | None
    raw_uncertainty: float | None
    source: str
    training_points: int
    uncertainty_points: int
    error: str | None = None


def _empirical_residual_uncertainty(
    prior_abs_errors: list[float] | np.ndarray,
    cfg: dict,
) -> tuple[float | None, int, str | None]:
    """Return a causal empirical error scale from PRIOR one-step-ahead errors.

    This is deliberately not called a Gaussian sigma. The replay model uses a
    GP only for the next-point mean prediction. Its uncertainty width comes from
    actual historical out-of-sample prediction errors that were available before
    the current trial.

    The base width is a rolling quantile (median by default) of absolute causal
    prediction errors. CHARIOT's separate calibration layer may then widen this
    raw width to the requested empirical coverage target.
    """
    replay_cfg = cfg.get("replay", {})
    min_points = int(replay_cfg.get("min_residual_points_for_uncertainty", 5))
    window = int(replay_cfg.get("empirical_uncertainty_window", 20))
    base_quantile = float(replay_cfg.get("empirical_uncertainty_base_quantile", 0.50))
    floor = float(cfg.get("triage", {}).get("uncertainty_floor_S_per_m", 0.05))

    arr = np.asarray(prior_abs_errors, dtype=float)
    arr = arr[np.isfinite(arr)]

    if window > 0 and len(arr) > window:
        arr = arr[-window:]

    n = int(len(arr))
    if n < min_points:
        return (
            None,
            n,
            f"Need at least {min_points} prior causal prediction errors to estimate replay uncertainty; {n} available.",
        )

    q = min(max(base_quantile, 0.0), 1.0)
    raw_width = float(np.quantile(arr, q, method="higher"))
    raw_width = max(raw_width, floor)
    return raw_width, n, None


def predict_with_causal_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_next: np.ndarray,
    cfg: dict,
    prior_abs_errors: list[float] | np.ndarray | None = None,
) -> ReplayPrediction:
    """Fit a GP mean model on PRIOR rows only and predict the next row.

    IMPORTANT:
    - The GP is used only for the causal next-point MEAN prediction.
    - sklearn's GP ``return_std`` is intentionally NOT used for the historical
      replay uncertainty because it was severely overconfident on the demo data.
    - Replay uncertainty is instead derived from prior one-step-ahead absolute
      prediction errors, so the current trial never contributes to its own band.

    This remains a retrospective causal replay model. It is not a claim that it
    reproduces Reza's production digital twin. In live operation CHARIOT should
    consume Reza's frozen pre-trial mean and raw uncertainty directly.
    """
    n = int(len(y_train))
    replay_cfg = cfg.get("replay", {})
    min_train = int(replay_cfg.get("min_train_points_for_model", 4))

    if n < min_train:
        return ReplayPrediction(
            None,
            None,
            "causal_replay_gp_mean_empirical_residual_uncertainty",
            n,
            0,
            f"Need at least {min_train} prior points for replay GP mean prediction.",
        )

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.exceptions import ConvergenceWarning

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_next = np.asarray(x_next, dtype=float).reshape(1, -1)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        xs = scaler.transform(x_next)

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * Matern(
                length_scale=np.ones(Xs.shape[1]),
                length_scale_bounds=(1e-2, 1e3),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=0.05,
                noise_level_bounds=(1e-6, 1e2),
            )
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=int(replay_cfg.get("random_state", 0)),
            n_restarts_optimizer=int(replay_cfg.get("gp_optimizer_restarts", 1)),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Xs, y_train)

        # Mean only. Do not use GP return_std for retrospective CHARIOT replay.
        mean = float(gp.predict(xs, return_std=False)[0])

        raw_uncertainty, uncertainty_points, uncertainty_error = (
            _empirical_residual_uncertainty(
                prior_abs_errors if prior_abs_errors is not None else [],
                cfg,
            )
        )

        return ReplayPrediction(
            mean,
            raw_uncertainty,
            "causal_replay_gp_mean_empirical_residual_uncertainty",
            n,
            uncertainty_points,
            uncertainty_error,
        )

    except Exception as exc:
        # A replay failure should make CHARIOT conservative rather than crash.
        return ReplayPrediction(
            None,
            None,
            "causal_replay_gp_mean_empirical_residual_uncertainty",
            n,
            0,
            f"Replay GP failed: {exc}",
        )
