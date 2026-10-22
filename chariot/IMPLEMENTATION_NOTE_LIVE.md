# v0.8-live implementation note

This update converts the validated v0.7 shadow/replay architecture into a live, file-decoupled integration suitable for an ongoing ~100-point BRINE campaign.

The major design choice is that BRINE and Reza's twin remain independent. Reza writes a pre-trial conductivity prediction CSV; BRINE writes/updates its normal campaign CSV; CHARIOT observes both and writes a separate frozen decision log. There is no direct control dependency between CHARIOT/Gemini and BRINE.

CHARIOT verifies causality using `training_data_last_trial < trial_index`. When x1–x4 are included in the twin file, it also verifies that the frozen twin prediction belongs to the formulation BRINE actually executed. Duplicate predictions or composition mismatches block a shadow decision instead of being silently resolved.

The twin predicts conductivity only. CHARIOT never predicts LSV/HER/OER. It decides whether conductivity evidence is sufficiently strong for RUN, REVIEW, or SKIP in shadow mode. The twin's uncertainty is stored as a raw width and empirically calibrated from prior frozen prediction/EIS errors; it is not called a sigma or z-score.

Gemini remains on-demand. It does not run on every experiment. Before each user answer, it reloads the current BRINE CSV, current twin prediction file, and current CHARIOT decision log. Thus a chat can remain open while the campaign grows one point at a time without stale numerical state.
