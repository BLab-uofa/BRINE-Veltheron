"""Illustrative future BRINE hook. DO NOT connect during shadow validation.

The current project is intentionally shadow-only. If control is ever enabled,
BRINE should consume an already-computed CHARIOT decision rather than recreate
the policy inside BRINE.
"""


def should_run_lsv_from_chariot_action(action: str) -> bool:
    """RUN and REVIEW both perform LSV; only an explicit SKIP blocks it."""
    return str(action).lower() != "skip_lsv"
