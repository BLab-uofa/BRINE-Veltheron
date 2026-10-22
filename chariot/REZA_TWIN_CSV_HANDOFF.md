# Reza → CHARIOT live CSV handoff

Before BRINE runs trial `n`, the conductivity twin should append exactly one row for trial `n`.

Recommended columns:

```csv
trial_index,prediction_timestamp_utc,training_data_last_trial,x1,x2,x3,x4,predicted_conductivity_S_per_m,raw_uncertainty_S_per_m
46,2026-10-21T15:42:10Z,45,0,1100,1,0,22.4,1.6
```

## Required meaning

- `trial_index`: BRINE trial this prediction belongs to.
- `prediction_timestamp_utc`: timestamp when the prediction was made.
- `training_data_last_trial`: newest experimental trial used to fit/update the twin. It MUST be smaller than `trial_index`.
- `x1..x4`: candidate stock-solution volumes in µL. Strongly recommended because CHARIOT verifies the frozen prediction belongs to the formulation BRINE actually runs.
- `predicted_conductivity_S_per_m`: twin prediction of conductivity only.
- `raw_uncertainty_S_per_m`: twin's raw uncertainty expressed as an **absolute conductivity width in S/m**. CHARIOT does not assume this is a Gaussian sigma; it calibrates the raw width empirically as data accumulate.

The twin does **not** predict HER, OER, LSV stability, or whether LSV will pass.

## Causal rule

For trial `n`, the prediction must be written BEFORE trial `n` is measured, and:

```text
training_data_last_trial < trial_index
```

Example: prediction for trial 46 may use data only through trial 45 or earlier.

## File behavior

Append one row per trial. Do not overwrite earlier prediction rows. CHARIOT rejects duplicate trial predictions and causality violations instead of silently choosing a row.
