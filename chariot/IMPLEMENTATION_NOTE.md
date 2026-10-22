# CHARIOT v0.7 implementation note

## What changed

v0.7 replaces the old fixed EIS threshold shadow policy with a causal,
uncertainty-aware three-state characterization policy while keeping the entire
system shadow-only.

The digital twin still predicts **conductivity only**. CHARIOT does not predict
HER, OER, electrochemical stability, or LSV outcome.

For live trial `n`, the required causal order is:

1. The conductivity twin predicts trial `n` from information available before trial `n`.
2. The prediction and raw uncertainty are frozen in `pretrial_predictions.csv`.
3. BRINE runs trial `n` and EIS produces measured conductivity.
4. CHARIOT compares the measured EIS with the frozen twin expectation.
5. CHARIOT freezes `RUN_LSV`, `REVIEW_LSV`, or `SKIP_LSV` in its own decision log.
6. BRINE continues its normal protocol regardless because CHARIOT remains shadow-only.

Old decisions are never recomputed with later data.

## Why the policy changed

The previous policy was essentially a fixed conductivity threshold. That could
reduce LSV use, but it was not a distinctive decision layer and could not express
uncertainty or model/measurement disagreement.

The new policy is conservative:

- measured EIS at/above the conductivity target -> RUN LSV when evidence is consistent;
- below-target EIS + calibrated twin evidence confidently below target -> SKIP LSV;
- cold start, target-crossing uncertainty band, or strong model/EIS disagreement -> REVIEW.

`REVIEW` counts as performing LSV in all benchmark calculations.

## Twin uncertainty

The twin's uncertainty is **not called a sigma** unless its statistical meaning is
later established. CHARIOT stores it as raw prediction uncertainty and estimates an
empirical scale factor from prior causal prediction errors.

The normalized disagreement score is:

`abs(EIS - twin_mean) / calibrated_prediction_band_half_width`

It is explicitly **not a z-score**.

## Calibration

For prior completed trials with a valid frozen prediction, CHARIOT calculates:

`ratio = abs(EIS - twin_mean) / raw_uncertainty`

The configured empirical quantile of past ratios scales the raw uncertainty for
future decisions. The scale factor is never allowed to shrink the twin's native
uncertainty. Automatic SKIP is disabled during cold start until enough prior
prediction/error pairs exist.

## Live vs retrospective prediction

Live operation should use the real external conductivity twin. Its prediction must
be frozen before EIS exists. `training_data_last_trial` must be smaller than the
candidate trial index; the logger rejects a prediction that violates this guard.

Historical data usually do not contain those frozen predictions. For retrospective
benchmarking only, v0.7 includes a small optimized Gaussian-process **causal replay
model**. It is refit on trials available before each historical trial, predicts the
next trial, then reveals its measured EIS. This model is not represented as Reza's
production digital twin.

## Benchmark framing

The fixed 20 S/m gate necessarily skips at least as many LSVs as CHARIOT because
CHARIOT sends REVIEW cases to LSV. Therefore raw LSV savings are not the headline
comparison.

The intended comparison is:

- Full protocol: safest, most measurements.
- Fixed gate: maximum raw savings, no model/measurement uncertainty handling.
- CHARIOT: intermediate measurement burden, but escalates decision-insufficient
  or model/EIS disagreement cases instead of blindly discarding them.

Reported metrics include LSV reduction versus the full protocol, fraction of the
fixed gate's savings retained, number of below-target cases escalated to REVIEW,
disagreement reviews, qualified-candidate retention, prediction error, and
empirical prediction-band coverage.

## Current demo caveat

The included 45-row demonstration campaign is not a proof of experimental savings.
Its low-conductivity points occur early in the sequence, when a causal replay model
has little trustworthy history. Under the conservative v0.7 defaults this can lead
to REVIEW rather than autonomous SKIP. Do not tune the policy simply to manufacture
a savings result. The final ~100-point campaign and frozen predictions from the
actual twin are the meaningful evaluation.

## Continuously updating behavior

`shadow_monitor.py` watches both the BRINE CSV and the frozen pre-trial prediction
log. As new information arrives it appends new frozen decisions only; it never
rewrites an old decision.

`gemini_chat.py` invokes Gemini only when the scientist asks a question. Before
every answer it reloads the latest BRINE CSV and CHARIOT decision log, so the
conversation always reflects the current campaign state without running the LLM on
every experiment.
