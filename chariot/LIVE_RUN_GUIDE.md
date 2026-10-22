# Live BRINE + Twin + CHARIOT + Gemini

The live system uses three files:

1. BRINE's growing campaign CSV (read only by CHARIOT/Gemini).
2. Reza's growing `twin_predictions.csv` (read only by CHARIOT/Gemini).
3. `outputs/chariot_decision_log.csv`, written only by CHARIOT as frozen shadow decisions.

CHARIOT remains shadow-only. It never prevents BRINE from running LSV.

## Terminal 1 — CHARIOT monitor

```powershell
python shadow_monitor.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv"
```

If BRINE's CSV already contains old rows and live monitoring should begin at a later trial:

```powershell
python shadow_monitor.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv" --start-trial 46
```

## Terminal 2 — scientist chat

```powershell
python gemini_chat.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv" --decision-log outputs/chariot_decision_log.csv --bond
```

Gemini reloads all current files before every answer. If BRINE grows from 37 to 39 trials while the chat stays open, the next question uses the 39-trial state.

## Optional status check without Gemini

```powershell
python live_campaign_status.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv"
```

## Before the campaign

```powershell
python validate_live_inputs.py --file "PATH_TO_BRINE.csv" --predictions "PATH_TO_TWIN_PREDICTIONS.csv"
```

## Causal sequence

```text
Twin predicts conductivity for Trial n from data through < n
        ↓
Prediction is frozen in twin_predictions.csv
        ↓
BRINE runs Trial n
        ↓
EIS conductivity appears in BRINE CSV
        ↓
CHARIOT reads both files and freezes RUN / REVIEW / SKIP shadow recommendation
        ↓
BRINE continues unchanged; later HER/OER updates do not rewrite the old CHARIOT decision
```
