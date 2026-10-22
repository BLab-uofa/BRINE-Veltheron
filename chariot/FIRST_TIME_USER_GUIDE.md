# First-time guide — v0.7

## A. One-time setup

Open PowerShell in this folder and activate your existing virtual environment, then:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

You should see `9 passed`.

Keep your `.env` local. Do not paste API keys or Bond tokens into chat.

## B. Test the existing 45-row demo

```powershell
python replay_chariot.py --file data/demo_candidates.csv
```

This creates the causal replay, benchmark JSON and sensitivity CSV under `outputs/`.

Then launch the observer:

```powershell
python gemini_chat.py --file data/demo_candidates.csv --decision-log outputs/chariot_replay.csv --bond
```

## C. Live campaign tomorrow

### Before each trial runs

Freeze the actual conductivity twin's prediction and its actual per-prediction raw
uncertainty:

```powershell
python capture_pretrial_prediction.py --trial TRIAL --mean MEAN_S_PER_M --uncertainty UNCERTAINTY_S_PER_M --training-last-trial LAST_TRAINING_TRIAL
```

The script refuses a prediction if the twin claims it was trained through the same
trial being predicted.

### Start CHARIOT watcher

```powershell
python shadow_monitor.py --file "PATH_TO_BRINE_OUTPUT.csv"
```

Leave it running. It will wait until both the frozen twin prediction and EIS are
available, then append one causal shadow decision. It never changes BRINE.

### Talk to the system at any time

In another PowerShell window:

```powershell
python gemini_chat.py --file "PATH_TO_BRINE_OUTPUT.csv" --decision-log outputs/chariot_decision_log.csv --bond
```

Every question reloads the current files, so the user can converse while the dataset
grows point by point.

## D. Important scientific language

Say:

> The conductivity twin provides a frozen prior expectation and raw uncertainty.
> EIS measures actual conductivity. CHARIOT determines whether the conductivity
> evidence is sufficient to justify spending destructive LSV characterization.
> LSV independently measures HER/OER; CHARIOT does not predict its outcome.

Do not call the raw twin uncertainty a sigma unless the twin developer later proves
that interpretation. Do not call the disagreement score a z-score.
