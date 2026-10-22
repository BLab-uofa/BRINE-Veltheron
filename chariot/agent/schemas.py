from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel


class PlotRequest(BaseModel):
    plot_type: Literal[
        "conductivity_over_trials",
        "her_oer_over_trials",
        "stability_window_over_trials",
        "input_vs_conductivity",
        "concentration_vs_conductivity",
        "conductivity_vs_stability_window",
        "shadow_actions_over_trials",
    ]
    input_name: Optional[Literal["x1", "x2", "x3", "x4"]] = None


class ObserverResponse(BaseModel):
    answer: str
    plot_request: Optional[PlotRequest] = None
    needs_literature: bool = False
    literature_query: Optional[str] = None
