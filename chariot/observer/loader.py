from __future__ import annotations
from pathlib import Path
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_brine_csv(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Normalize the existing BRINE-style CSV without modifying the source file."""
    path = Path(path)
    raw = pd.read_csv(path)
    cols = cfg["columns"]
    out = pd.DataFrame(index=raw.index)

    trial_col = cols.get("trial_index", "trial_index")
    out["trial_index"] = _num(raw[trial_col]) if trial_col in raw else range(len(raw))

    for logical, source in cols["inputs"].items():
        out[logical] = _num(raw[source]) if source in raw else float("nan")

    outs = cols["outputs"]
    for logical in ("conductivity", "her", "oer"):
        source = outs.get(logical)
        out[logical] = _num(raw[source]) if source and source in raw else float("nan")

    # User-defined concentration conversion. Missing stock concentrations remain NaN.
    total_uL = float(cfg["campaign"]["total_formulation_volume_uL"])
    for x in ("x1", "x2", "x3", "x4"):
        stock = cfg["stocks"].get(x, {})
        name = stock.get("name", x)
        c_stock = stock.get("stock_concentration_M")
        out[f"{x}_solution_name"] = name
        if c_stock is None:
            out[f"{x}_M"] = float("nan")
        else:
            out[f"{x}_M"] = out[x] * float(c_stock) / total_uL

    out["total_solution_volume_uL"] = out[["x1", "x2", "x3", "x4"]].sum(axis=1)
    out["remaining_to_1200_uL"] = total_uL - out["total_solution_volume_uL"]
    out["stability_window_V"] = out["oer"] - out["her"]
    return out
