# BRINE–CHARIOT v0.8-live

A read-only, continuously updating shadow characterization layer for BRINE.

## Core separation

- **BRINE** runs the experiment.
- **Digital twin** predicts **conductivity only** plus a raw uncertainty before each trial.
- **EIS** measures conductivity.
- **CHARIOT** makes a frozen shadow `RUN_LSV / REVIEW_LSV / SKIP_LSV` recommendation from the conductivity evidence.
- **LSV** independently measures HER/OER; CHARIOT does not predict its outcome.
- **Gemini** is an on-demand read-only scientific interface.
- **Valency Bond** is used only when the scientist explicitly requests literature.

## Live files

```text
BRINE campaign CSV              (external, read only)
Reza twin prediction CSV        (external, read only)
outputs/chariot_decision_log.csv  (CHARIOT frozen shadow decisions)
```

See `REZA_TWIN_CSV_HANDOFF.md` and `LIVE_RUN_GUIDE.md`.

## Test

```powershell
python -m pytest -q
```

Expected: `13 passed`.

## Run live

Terminal 1:

```powershell
python shadow_monitor.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv"
```

Terminal 2:

```powershell
python gemini_chat.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv" --decision-log outputs/chariot_decision_log.csv --bond
```

Gemini rebuilds the current campaign/twin/CHARIOT state before every answer, so the chat may remain open while the dataset grows one trial at a time.
