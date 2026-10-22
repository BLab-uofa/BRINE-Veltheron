# BRINE × CHARIOT Live Dashboard

The dashboard is a read-only scientist interface. It does not control BRINE, BO, the digital twin, EIS, LSV, or laboratory hardware.

## Install the one new dependency

From the project root with `.venv` activated:

```powershell
python -m pip install -r requirements.txt
```

## Launch against the test files

```powershell
python -m streamlit run dashboard.py -- --file "data\brine_live_test.csv" --predictions "data\twin_predictions_test.csv" --decision-log "outputs\chariot_live_test.csv" --bond
```

Streamlit prints a local browser address, normally `http://localhost:8501`.

## Launch during the real campaign

```powershell
python -m streamlit run dashboard.py -- --file "PATH_TO_REAL_BRINE.csv" --predictions "PATH_TO_REZA_TWIN_PREDICTIONS.csv" --decision-log "outputs\chariot_decision_log.csv" --bond
```

Keep `shadow_monitor.py` running in a separate PowerShell window. The dashboard does not replace the watcher; it reads the continuously updated files.

## What updates automatically

The overview, CHARIOT counts, pending handoff state, conductivity plot, disagreement plot, and data tables reload from disk every few seconds. The refresh interval is adjustable in the sidebar.

Gemini is called only when the scientist submits a chat question. Each chat answer independently reloads the newest campaign/twin/decision files, so the conversation stays current while BRINE continues adding trials.

Valency Bond is called only for explicit literature/paper requests.

## Tabs

- **Overview** — live campaign counts, current/best conductivity, pending handoffs, twin-vs-EIS trace, latest decision.
- **CHARIOT** — RUN/REVIEW/SKIP counts, model readiness, disagreement history, frozen causal decision table.
- **Campaign data** — current campaign statistics and merged current-data/frozen-decision table.
- **Scientist chat** — below the live tabs; asks Gemini about the current state and optionally literature.

## Safety boundary

The GUI is read-only. CHARIOT remains shadow-only. A GUI crash does not alter BRINE or the twin.
