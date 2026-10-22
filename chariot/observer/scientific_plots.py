from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# Publication-style composition palette based on the BRINE figure language
# shared by the user: blue / gold / magenta / green stacked formulation bars.
_COMPONENT_COLORS = {
    "x1": "#4C78A8",  # NaCl - blue
    "x2": "#D4B72C",  # KCl - gold
    "x3": "#C85A91",  # ZnCl2 - magenta
    "x4": "#54A24B",  # ZnSO4 - green
}

_TWIN_COLOR = "#4C8EDA"
_HER_COLOR = "#3F83C5"
_OER_COLOR = "#D65C69"
_WINDOW_COLOR = "#9AA2AE"


def _theme_ink() -> str:
    """Foreground color inherited from the active dashboard theme."""
    return str(mpl.rcParams.get("text.color", "#111111"))


def _theme_muted() -> str:
    """Muted line color inherited from the active dashboard theme."""
    return str(mpl.rcParams.get("xtick.color", "#777777"))


def _theme_surface() -> str:
    """Axes background used for marker edges/fills in either theme."""
    return str(mpl.rcParams.get("axes.facecolor", "#ffffff"))


@dataclass(frozen=True)
class CompositionPlotData:
    values: pd.DataFrame
    labels: list[str]
    y_label: str
    mode: str


def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _first_numeric_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Return the first existing numeric column from a list of aliases."""
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _composition_data(df: pd.DataFrame, cfg: dict) -> CompositionPlotData:
    """Return composition values for stacked bars.

    If all stock concentrations are configured, use component molarity.
    Otherwise use the actual stock-solution volume in uL. Missing stock
    concentrations are never inferred.
    """
    xcols = ["x1", "x2", "x3", "x4"]
    stocks = cfg.get("stocks", {})

    all_concentrations_configured = all(
        stocks.get(x, {}).get("stock_concentration_M") is not None
        for x in xcols
    )

    labels = [
        str(stocks.get(x, {}).get("name", x))
        for x in xcols
    ]

    if all_concentrations_configured:
        values = pd.DataFrame(
            {
                x: _numeric(df.get(f"{x}_M"), df.index).fillna(0.0)
                for x in xcols
            },
            index=df.index,
        )
        return CompositionPlotData(
            values=values,
            labels=labels,
            y_label="Component concentration (M)",
            mode="molarity",
        )

    values = pd.DataFrame(
        {
            x: _numeric(df.get(x), df.index).fillna(0.0)
            for x in xcols
        },
        index=df.index,
    )

    return CompositionPlotData(
        values=values,
        labels=labels,
        y_label="Stock-solution volume (µL)",
        mode="volume",
    )


def _ordered_campaign(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["trial_index"] = pd.to_numeric(
        d.get("trial_index"),
        errors="coerce",
    )
    d = d[d["trial_index"].notna()].copy()
    d["trial_index"] = d["trial_index"].astype(int)
    return (
        d.sort_values("trial_index", kind="stable")
        .reset_index(drop=True)
    )


def _set_integer_trial_ticks(ax, x: np.ndarray) -> None:
    """Always show experiment number as integers, never decimal ticks."""
    if len(x) == 0:
        return

    xmin = int(np.nanmin(x))
    xmax = int(np.nanmax(x))
    n = xmax - xmin + 1

    # Aim for roughly 10-12 readable tick labels for long campaigns.
    if n <= 12:
        step = 1
    elif n <= 30:
        step = 2
    elif n <= 60:
        step = 5
    else:
        step = 10

    ticks = np.arange(xmin, xmax + 1, step, dtype=int)
    if len(ticks) == 0 or ticks[-1] != xmax:
        ticks = np.append(ticks, xmax)

    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(xmin - 0.75, xmax + 0.75)


def _set_headroom(ax, values, lower_zero: bool = True, frac: float = 0.16) -> None:
    """Give data visible breathing room instead of pinning to plot borders."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    vmax = float(np.max(arr))
    vmin = float(np.min(arr))

    if lower_zero and vmin >= 0:
        top = vmax * (1.0 + frac)
        if top <= 0:
            top = 1.0
        ax.set_ylim(0.0, top)
        return

    span = vmax - vmin
    if span <= 0:
        span = max(abs(vmax), 1.0)

    ax.set_ylim(
        vmin - 0.10 * span,
        vmax + frac * span,
    )


def _style_axis(ax, xlabel: str | None, ylabel: str) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_composition_bars(ax, x: np.ndarray, comp: CompositionPlotData) -> np.ndarray:
    bottoms = np.zeros(len(x), dtype=float)

    for logical, label in zip(
        ["x1", "x2", "x3", "x4"],
        comp.labels,
    ):
        y = comp.values[logical].to_numpy(dtype=float)
        ax.bar(
            x,
            y,
            bottom=bottoms,
            width=0.78,
            color=_COMPONENT_COLORS[logical],
            edgecolor=_theme_surface(),
            linewidth=0.25,
            alpha=0.92,
            label=label,
            zorder=1,
        )
        bottoms += y

    _set_headroom(ax, bottoms, lower_zero=True, frac=0.18)
    return bottoms


def plot_eis_with_composition(df: pd.DataFrame, cfg: dict):
    """Publication-style stacked composition + measured EIS conductivity."""
    d = _ordered_campaign(df)
    comp = _composition_data(d, cfg)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = d["trial_index"].to_numpy(dtype=int)

    totals = _draw_composition_bars(ax, x, comp)
    _style_axis(ax, "Experiment number", comp.y_label)
    _set_integer_trial_ticks(ax, x)

    ax2 = ax.twinx()
    conductivity = _first_numeric_column(
        d,
        ["conductivity", "y1"],
    )
    valid = conductivity.notna()

    if valid.any():
        ax2.plot(
            x[valid.to_numpy()],
            conductivity[valid].to_numpy(dtype=float),
            color=_theme_ink(),
            linestyle=":",
            marker="*",
            markerfacecolor=_theme_ink(),
            markeredgecolor=_theme_surface(),
            markeredgewidth=0.35,
            markersize=7.5,
            linewidth=1.35,
            label="Measured EIS conductivity",
            zorder=6,
        )

    target = float(
        cfg.get("campaign", {})
        .get("conductivity_target_S_per_m", 20.0)
    )
    ax2.axhline(
        target,
        color=_theme_muted(),
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Target ({target:g} S/m)",
        zorder=2,
    )

    right_values = conductivity[valid].tolist() + [target]
    _set_headroom(ax2, right_values, lower_zero=False, frac=0.14)
    _style_axis(ax2, None, "Ionic conductivity (S/m)")

    handles = [
        Patch(facecolor=_COMPONENT_COLORS[k], label=label)
        for k, label in zip(["x1", "x2", "x3", "x4"], comp.labels)
    ]
    handles += [
        Line2D(
            [0], [0],
            color=_theme_ink(),
            linestyle=":",
            marker="*",
            markersize=7,
            label="Measured EIS conductivity",
        ),
        Line2D(
            [0], [0],
            color=_theme_muted(),
            linestyle="--",
            label=f"Target ({target:g} S/m)",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=6,
        frameon=False,
        fontsize=9,
        handlelength=1.7,
        columnspacing=1.2,
    )

    ax.set_title(
        "Electrolyte composition and measured EIS conductivity",
        pad=38,
        fontsize=12,
        fontweight="semibold",
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.915,
        bottom=0.12,
        top=0.82,
    )

    return fig, comp.mode


def plot_twin_vs_eis(df: pd.DataFrame, predictions: pd.DataFrame, cfg: dict):
    """Measured EIS vs frozen pre-trial twin prediction + raw uncertainty."""
    d = _ordered_campaign(df)
    p = predictions.copy()

    if not p.empty:
        p["trial_index"] = pd.to_numeric(
            p.get("trial_index"),
            errors="coerce",
        )
        p = p[p["trial_index"].notna()].copy()
        p["trial_index"] = p["trial_index"].astype(int)
        p = p.sort_values("trial_index", kind="stable")

    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    conductivity = _first_numeric_column(
        d,
        ["conductivity", "y1"],
    )
    valid_meas = conductivity.notna()

    if valid_meas.any():
        xm = d.loc[valid_meas, "trial_index"].to_numpy(dtype=int)
        ym = conductivity[valid_meas].to_numpy(dtype=float)
        ax.plot(
            xm,
            ym,
            color=_theme_ink(),
            linestyle=":",
            marker="*",
            markeredgecolor=_theme_surface(),
            markeredgewidth=0.35,
            markersize=7.5,
            linewidth=1.35,
            label="Measured EIS",
            zorder=6,
        )

    all_y = list(conductivity[valid_meas].dropna().astype(float))

    if not p.empty:
        mu = _first_numeric_column(
            p,
            [
                "twin_predicted_conductivity_S_per_m",
                "predicted_conductivity_S_per_m",
            ],
        )
        unc = _first_numeric_column(
            p,
            [
                "twin_raw_uncertainty_S_per_m",
                "raw_uncertainty_S_per_m",
            ],
        )
        valid_mu = mu.notna()

        if valid_mu.any():
            xp = p.loc[valid_mu, "trial_index"].to_numpy(dtype=int)
            mup = mu[valid_mu].to_numpy(dtype=float)

            ax.plot(
                xp,
                mup,
                color=_TWIN_COLOR,
                linewidth=1.8,
                marker="o",
                markerfacecolor=_theme_surface(),
                markeredgecolor=_TWIN_COLOR,
                markeredgewidth=1.0,
                markersize=4.8,
                label="Twin prediction",
                zorder=5,
            )

            all_y.extend(mup.tolist())

            unc_valid = valid_mu & unc.notna() & (unc >= 0)
            if unc_valid.any():
                xu = p.loc[unc_valid, "trial_index"].to_numpy(dtype=int)
                muu = mu[unc_valid].to_numpy(dtype=float)
                uu = unc[unc_valid].to_numpy(dtype=float)
                order = np.argsort(xu)

                lower = (muu - uu)[order]
                upper = (muu + uu)[order]

                ax.fill_between(
                    xu[order],
                    lower,
                    upper,
                    color=_TWIN_COLOR,
                    alpha=0.18,
                    linewidth=0,
                    label="Twin raw uncertainty",
                    zorder=1,
                )
                all_y.extend(lower.tolist())
                all_y.extend(upper.tolist())

    target = float(
        cfg.get("campaign", {})
        .get("conductivity_target_S_per_m", 20.0)
    )
    ax.axhline(
        target,
        color=_theme_muted(),
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Target ({target:g} S/m)",
        zorder=2,
    )
    all_y.append(target)

    combined_x = []
    if not d.empty:
        combined_x.extend(d["trial_index"].astype(int).tolist())
    if not p.empty:
        combined_x.extend(p["trial_index"].astype(int).tolist())

    if combined_x:
        _set_integer_trial_ticks(ax, np.asarray(combined_x, dtype=int))

    _set_headroom(ax, all_y, lower_zero=False, frac=0.15)
    _style_axis(ax, "Experiment number", "Ionic conductivity (S/m)")

    ax.set_title(
        "Measured EIS vs digital-twin conductivity prediction",
        pad=38,
        fontsize=12,
        fontweight="semibold",
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.3,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.97,
        bottom=0.13,
        top=0.80,
    )

    return fig


def plot_lsv_with_composition(df: pd.DataFrame, cfg: dict):
    """Clear two-panel LSV + composition figure.

    The two LSV endpoint traces are intentionally separated from the stacked
    composition bars. This keeps HER/OER fully legible while preserving the
    common experiment-number axis and direct composition context.
    """
    d = _ordered_campaign(df)
    comp = _composition_data(d, cfg)
    x = d["trial_index"].to_numpy(dtype=int)

    fig, (ax_lsv, ax_comp) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.1),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.25, 1.0],
            "hspace": 0.10,
        },
    )

    # ------------------------------------------------------------------
    # TOP: HER / OER only, with a visible stability-window fill.
    # ------------------------------------------------------------------
    her = _first_numeric_column(d, ["her", "y3"])
    oer = _first_numeric_column(d, ["oer", "y4"])

    valid_her = her.notna()
    valid_oer = oer.notna()

    if valid_her.any():
        ax_lsv.plot(
            x[valid_her.to_numpy()],
            her[valid_her].to_numpy(dtype=float),
            color=_HER_COLOR,
            linestyle="--",
            marker="v",
            markerfacecolor=_theme_surface(),
            markeredgecolor=_HER_COLOR,
            markeredgewidth=1.0,
            markersize=5.4,
            linewidth=1.7,
            label="HER endpoint",
            zorder=6,
        )

    if valid_oer.any():
        ax_lsv.plot(
            x[valid_oer.to_numpy()],
            oer[valid_oer].to_numpy(dtype=float),
            color=_OER_COLOR,
            linestyle="-",
            marker="^",
            markerfacecolor=_theme_surface(),
            markeredgecolor=_OER_COLOR,
            markeredgewidth=1.0,
            markersize=5.4,
            linewidth=1.8,
            label="OER endpoint",
            zorder=6,
        )

    both = valid_her & valid_oer
    if both.any():
        xb = x[both.to_numpy()]
        hb = her[both].to_numpy(dtype=float)
        ob = oer[both].to_numpy(dtype=float)
        order = np.argsort(xb)

        ax_lsv.fill_between(
            xb[order],
            hb[order],
            ob[order],
            color=_WINDOW_COLOR,
            alpha=0.22,
            linewidth=0,
            label="HER–OER stability window",
            zorder=1,
        )

    lsv_values = pd.concat([her, oer], ignore_index=True).dropna()
    _set_headroom(
        ax_lsv,
        lsv_values,
        lower_zero=False,
        frac=0.16,
    )
    _style_axis(ax_lsv, None, "LSV endpoint potential (V)")

    ax_lsv.set_title(
        "LSV endpoints and electrolyte composition",
        pad=18,
        fontsize=12,
        fontweight="semibold",
    )

    ax_lsv.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # BOTTOM: composition only, with deliberate headroom above bars.
    # ------------------------------------------------------------------
    totals = _draw_composition_bars(ax_comp, x, comp)
    _style_axis(ax_comp, "Experiment number", comp.y_label)
    _set_integer_trial_ticks(ax_comp, x)

    comp_handles = [
        Patch(
            facecolor=_COMPONENT_COLORS[k],
            label=label,
        )
        for k, label in zip(
            ["x1", "x2", "x3", "x4"],
            comp.labels,
        )
    ]

    ax_comp.legend(
        handles=comp_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.97,
        bottom=0.10,
        top=0.86,
    )

    return fig, comp.mode
