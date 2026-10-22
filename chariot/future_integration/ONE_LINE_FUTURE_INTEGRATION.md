# Future BRINE integration — only after shadow validation

**Do not enable this for the hackathon shadow run.** BRINE remains untouched in v0.7.

The live shadow system writes a frozen causal CHARIOT action for each trial:

- `run_lsv`
- `review_lsv`
- `skip_lsv`

If shadow validation later justifies control and the BRINE owner approves it, the
only control hook should be at BRINE's existing LSV scheduling point. BRINE should
consume the already-computed CHARIOT action rather than duplicate CHARIOT logic.

Conceptually:

```python
if should_run_lsv_from_chariot_action(chariot_action):
    run_lsv(candidate)
```

`REVIEW` deliberately runs LSV. Only `SKIP` blocks it.

This preserves the current scientific boundary: the conductivity twin predicts
conductivity only; CHARIOT decides whether conductivity evidence is sufficient to
spend destructive LSV characterization; LSV itself independently measures HER/OER.
