# v0.9 GUI implementation note

This update adds a Streamlit live dashboard on top of the validated v0.8 live backend. It does not modify the causal CHARIOT policy, frozen-decision logic, BRINE watcher, Gemini evidence rules, or Valency Bond workflow.

The dashboard rebuilds live state from disk every refresh cycle and shows campaign progress, twin handoff state, CHARIOT action counts, model-readiness/disagreement information, frozen decisions, and current campaign data. Gemini chat is intentionally outside the auto-refresh loop: the deterministic dashboard refreshes continuously, while the LLM is called only when the scientist asks a question. Each Gemini turn still reloads the newest files.

The dashboard remains read-only and CHARIOT remains shadow-only.
