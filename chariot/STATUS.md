# BRINE–CHARIOT v0.8-live status

Implemented and tested:

- Live growing BRINE CSV read path.
- External Reza conductivity-twin CSV handoff.
- Causal leakage guard (`training_data_last_trial < trial_index`).
- Duplicate prediction rejection.
- Optional x1–x4 prediction/formulation integrity check.
- Raw uncertainty treated as an uncalibrated width, not a Gaussian sigma.
- Empirical causal uncertainty calibration from prior frozen prediction/EIS pairs.
- Shadow-only RUN / REVIEW / SKIP triage.
- Frozen decision log that is not rewritten when BRINE later fills HER/OER.
- Live watcher that never modifies or stops BRINE.
- Deterministic live pipeline status.
- Gemini reloads BRINE + twin + CHARIOT state before every answer.
- Valency Bond remains explicit-only.
- Historical replay remains separate from the live twin path.

Automated tests: 13 passing.
