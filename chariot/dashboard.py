from __future__ import annotations

import argparse
import html
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from cycler import cycler
from dotenv import load_dotenv

from agent.gemini_observer import GeminiObserver
from observer.dashboard_state import build_dashboard_snapshot
from observer.loader import load_brine_csv
from observer.plots import create_plot
from observer.scientific_plots import (
    plot_eis_with_composition,
    plot_twin_vs_eis,
    plot_lsv_with_composition,
)
from shadow.decision_log import load_decision_log
from shadow.view import build_shadow_view


load_dotenv()


# ============================================================================
# CLI
# ============================================================================

def _cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", default="data/demo_candidates.csv")
    parser.add_argument("--predictions", default="data/live_templates/twin_predictions_template.csv")
    parser.add_argument("--decision-log", default="outputs/chariot_replay.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bond", action="store_true")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


ARGS = _cli_args()


# ============================================================================
# PAGE + SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="BRINE × CHARIOT",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PERSISTED_PATHS_FILE = Path("config") / "dashboard_paths.json"


def _first_existing_path(*candidates: str, fallback: str = "") -> str:
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    return fallback


def _load_persisted_paths() -> dict[str, str]:
    try:
        if PERSISTED_PATHS_FILE.exists():
            data = json.loads(PERSISTED_PATHS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        pass
    return {}


def _save_persisted_paths() -> None:
    try:
        PERSISTED_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "campaign_path": str(st.session_state.get("campaign_path", "")),
            "prediction_path": str(st.session_state.get("prediction_path", "")),
            "decision_path": str(st.session_state.get("decision_path", "")),
            "config_path": str(st.session_state.get("config_path", "")),
        }
        PERSISTED_PATHS_FILE.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        # Path persistence should never stop the live dashboard.
        print(f"[Dashboard] Could not persist data-source paths: {exc}")


def _init_state():
    bond_configured = bool(os.getenv("BOND_BEARER_TOKEN"))
    saved = _load_persisted_paths()

    campaign_default = saved.get("campaign_path") or _first_existing_path(
        "data/candidates.csv",
        ARGS.file,
        fallback=ARGS.file,
    )
    prediction_default = saved.get("prediction_path") or _first_existing_path(
        "data/twin_predictions.csv",
        "outputs/twin_predictions.csv",
        ARGS.predictions,
        fallback=ARGS.predictions,
    )
    decision_default = saved.get("decision_path") or _first_existing_path(
        "outputs/chariot_decision_log.csv",
        ARGS.decision_log,
        fallback=ARGS.decision_log,
    )
    config_default = saved.get("config_path") or _first_existing_path(
        "config/system.yaml",
        "config/system.yml",
        fallback=ARGS.config or "",
    )

    defaults = {
        "campaign_path": campaign_default,
        "prediction_path": prediction_default,
        "decision_path": decision_default,
        "config_path": config_default,
        "campaign_path_input": campaign_default,
        "prediction_path_input": prediction_default,
        "decision_path_input": decision_default,
        "config_path_input": config_default,
        "pending_campaign_path": None,
        "pending_prediction_path": None,
        "pending_decision_path": None,
        "pending_config_path": None,
        "bond_enabled": bool(ARGS.bond) or bond_configured,
        "bond_autoconfig_v2": False,
        "dark_mode": True,
        "auto_refresh": True,
        "refresh_seconds": 3,
        "chat_messages": [],
        "observer": None,
        "observer_config_key": None,
        "active_page": "Dashboard",
        "pending_page": None,
        "last_non_chat_page": "Dashboard",
        "compact_chat_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# One-time migration from the old dashboard where Bond defaulted OFF.
if not st.session_state.get("bond_autoconfig_v2", False):
    if os.getenv("BOND_BEARER_TOKEN"):
        st.session_state["bond_enabled"] = True
    st.session_state["bond_autoconfig_v2"] = True

# Programmatic page changes (for chat maximize/minimize) are applied before
# the top navigation widget is instantiated on this run.
pending_page = st.session_state.get("pending_page")
if pending_page:
    st.session_state["active_page"] = pending_page
    st.session_state["pending_page"] = None


# ============================================================================
# THEME
# Streamlit uses a light native base so light mode is truly light.
# This app-level switch applies dark styling when requested and keeps plots in sync.
# ============================================================================

DARK_MODE = bool(st.session_state.get("dark_mode", True))

DARK_PAL = {
    "page": "#0d0d0d",
    "sidebar": "#111111",
    "surface": "#1a1a19",
    "surface_2": "#20201f",
    "input": "#1d1d1c",
    "ink": "#f0efec",
    "ink_2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "border": "rgba(255,255,255,0.12)",
    "primary": "#9085e9",
    "data": "#5aa0ef",
    "run": "#22c55e",
    "review": "#f59e0b",
    "skip": "#ef4444",
    "good": "#19a974",
    "warning": "#fab219",
}

LIGHT_PAL = {
    "page": "#f7f7f4",
    "sidebar": "#f0f1ee",
    "surface": "#ffffff",
    "surface_2": "#f5f5f2",
    "input": "#ffffff",
    "ink": "#161616",
    "ink_2": "#31312f",
    "muted": "#74746f",
    "grid": "#e5e5df",
    "baseline": "#cecec7",
    "border": "rgba(20,20,20,0.12)",
    "primary": "#6658cf",
    "data": "#2f78c8",
    "run": "#16a34a",
    "review": "#d97706",
    "skip": "#dc2626",
    "good": "#11875d",
    "warning": "#d99600",
}

PAL = DARK_PAL if DARK_MODE else LIGHT_PAL

ACTION_COLOR = {
    "run_lsv": PAL["run"],
    "review_lsv": PAL["review"],
    "skip_lsv": PAL["skip"],
    "wait_for_eis": PAL["muted"],
    "wait_for_twin": PAL["muted"],
}

mpl.rcParams.update({
    "figure.facecolor": PAL["surface"],
    "axes.facecolor": PAL["surface"],
    "savefig.facecolor": PAL["surface"],
    "figure.dpi": 130,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": PAL["ink_2"],
    "axes.edgecolor": PAL["baseline"],
    "axes.linewidth": 1.0,
    "axes.titlecolor": PAL["ink"],
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.titlepad": 10,
    "axes.labelcolor": PAL["ink_2"],
    "axes.labelsize": 11,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": PAL["grid"],
    "grid.linewidth": 0.8,
    "xtick.color": PAL["muted"],
    "ytick.color": PAL["muted"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=[
        PAL["data"], "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", PAL["primary"], "#e66767",
    ]),
})

st.markdown(
    f"""
    <style>
      :root {{ color-scheme: {'dark' if DARK_MODE else 'light'}; }}
      footer {{ visibility: hidden; height: 0; }}
      [data-testid="stAppViewContainer"], .stApp {{
          background: {PAL['page']} !important;
          color: {PAL['ink_2']} !important;
      }}
      [data-testid="stHeader"] {{ display:none !important; }}
      [data-testid="stToolbar"] {{ display:none !important; }}
      [data-testid="stSidebar"] > div:first-child {{
          background: {PAL['sidebar']} !important;
          border-right: 1px solid {PAL['border']};
      }}
      .block-container {{ padding-top: 0.75rem; padding-bottom: 5.5rem; max-width: 1500px; }}
      h1, h2, h3, h4 {{ color: {PAL['ink']} !important; letter-spacing: -0.01em; }}
      p, label, [data-testid="stCaptionContainer"] {{ color: {PAL['ink_2']} !important; }}

      div[data-testid="stMetric"] {{
         background: {PAL['surface']} !important; border: 1px solid {PAL['border']} !important;
         border-radius: 14px; padding: 0.72rem 0.92rem;
         box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
      div[data-testid="stMetricLabel"] p {{
         color: {PAL['muted']} !important; font-size: 0.74rem; font-weight: 650;
         text-transform: uppercase; letter-spacing: 0.04em; }}
      div[data-testid="stMetricValue"] {{ color: {PAL['ink']} !important; font-variant-numeric: tabular-nums; }}

      /* Native inputs: explicitly theme both the shell and actual input. */
      div[data-baseweb="input"],
      div[data-baseweb="input"] > div,
      div[data-baseweb="textarea"],
      div[data-baseweb="textarea"] > div,
      div[data-baseweb="select"] > div {{
          background: {PAL['input']} !important;
          border-color: {PAL['border']} !important;
          color: {PAL['ink']} !important;
      }}
      [data-testid="stTextInput"] input,
      [data-testid="stTextArea"] textarea,
      input, textarea {{
          background: {PAL['input']} !important;
          color: {PAL['ink']} !important;
          -webkit-text-fill-color: {PAL['ink']} !important;
          caret-color: {PAL['ink']} !important;
      }}
      input::placeholder, textarea::placeholder {{
          color: {PAL['muted']} !important;
          -webkit-text-fill-color: {PAL['muted']} !important;
      }}
      button[kind="secondary"], button[kind="tertiary"],
      [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"] {{
          background: {PAL['surface']} !important;
          color: {PAL['ink_2']} !important;
          border-color: {PAL['border']} !important;
      }}
      hr {{ border-color: {PAL['border']} !important; }}

      .app-sub {{ color: {PAL['muted']}; font-size: 0.92rem; margin: 0.1rem 0 0.65rem 0; }}
      .live-dot {{ width:9px; height:9px; border-radius:50%; background:{PAL['good']};
                   display:inline-block; margin-right:0.5rem;
                   box-shadow:0 0 0 3px rgba(25,169,116,0.20); }}
      .status-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:0.4rem; }}
      .status-ready {{ background:{PAL['good']}; }}
      .status-off {{ background:{PAL['muted']}; }}

      .chariot-badge {{ display:inline-block; padding:0.24rem 0.64rem; border-radius:999px;
                        font-size:0.72rem; font-weight:650; letter-spacing:0.03em;
                        margin-right:0.4rem; background:rgba(144,133,233,0.14);
                        color:{PAL['primary']}; border:1px solid rgba(144,133,233,0.30); }}
      .pill {{ display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
               font-size:0.72rem; font-weight:700; letter-spacing:0.02em; }}
      .pill-run {{ background:{PAL['run']}; color:#fff; }}
      .pill-review {{ background:{PAL['review']}; color:#0b0b0b; }}
      .pill-skip {{ background:{PAL['skip']}; color:#fff; }}

      /* Shadow-decision explainer cards. */
      .decision-grid {{
          display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:0.8rem; margin:0.35rem 0 0.9rem 0;
      }}
      .decision-card {{
          background:{PAL['surface']}; border:1px solid {PAL['border']}; border-radius:14px;
          padding:0.85rem 0.9rem 0.8rem 0.9rem; min-height:142px; position:relative; overflow:hidden;
      }}
      .decision-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent); }}
      .decision-label {{ font-size:0.72rem; font-weight:800; letter-spacing:0.055em; text-transform:uppercase; color:var(--accent); }}
      .decision-value {{ font-size:2rem; line-height:1.05; margin:0.35rem 0 0.45rem 0; color:{PAL['ink']}; font-variant-numeric:tabular-nums; }}
      .decision-desc {{ font-size:0.76rem; line-height:1.38; color:{PAL['muted']}; }}
      .decision-card.run {{ --accent:{PAL['run']}; }}
      .decision-card.review {{ --accent:{PAL['review']}; }}
      .decision-card.skip {{ --accent:{PAL['skip']}; }}
      .decision-card.ready {{ --accent:{PAL['primary']}; }}
      .decision-card.disagree {{ --accent:#e76f51; }}
      .decision-key {{
          display:flex; flex-wrap:wrap; gap:0.55rem; align-items:center;
          margin:0.15rem 0 1rem 0; color:{PAL['muted']}; font-size:0.78rem;
      }}
      .decision-key .key-item {{
          display:inline-flex; align-items:center; gap:0.35rem; background:{PAL['surface_2']};
          border:1px solid {PAL['border']}; border-radius:999px; padding:0.32rem 0.62rem;
      }}
      .decision-key .dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}

      .flag-pill {{ display:inline-block; padding:0.18rem 0.48rem; border-radius:999px; font-size:0.7rem; font-weight:700; }}
      .flag-ready {{ background:rgba(47,120,200,.16); color:{PAL['run']}; border:1px solid rgba(47,120,200,.28); }}
      .flag-review {{ background:rgba(217,150,0,.16); color:{PAL['review']}; border:1px solid rgba(217,150,0,.30); }}
      .flag-skip {{ background:rgba(17,135,93,.16); color:{PAL['skip']}; border:1px solid rgba(17,135,93,.30); }}
      .flag-disagree {{ background:rgba(231,111,81,.16); color:#e76f51; border:1px solid rgba(231,111,81,.30); }}
      @media (max-width:1100px) {{ .decision-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}

      .stTabs [data-baseweb="tab-list"] {{ gap:0.35rem; }}
      .stTabs [data-baseweb="tab"] {{ color:{PAL['muted']} !important; }}
      .stTabs [aria-selected="true"] {{ color:{PAL['ink']} !important; }}

      /* Theme-safe HTML tables replace native dataframe canvases. */
      .table-shell {{
          border: 1px solid {PAL['border']}; border-radius: 12px; overflow: auto;
          max-height: 500px; background: {PAL['surface']};
      }}
      .theme-table {{ width:100%; border-collapse:collapse; font-size:0.86rem; color:{PAL['ink_2']}; }}
      .theme-table thead th {{
          position:sticky; top:0; z-index:2; background:{PAL['surface_2']}; color:{PAL['ink']};
          padding:0.62rem 0.7rem; text-align:left; border-bottom:1px solid {PAL['border']};
          white-space:nowrap;
      }}
      .theme-table tbody td {{ padding:0.56rem 0.7rem; border-bottom:1px solid {PAL['border']}; white-space:nowrap; }}
      .theme-table tbody tr:hover {{ background:{PAL['surface_2']}; }}

      /* About page cards + architecture diagram. */
      .about-intro {{
          background:{PAL['surface']}; border:1px solid {PAL['border']}; border-radius:16px;
          padding:1rem 1.1rem; margin-bottom:1rem;
      }}
      .arch-scroll {{ overflow-x:auto; padding:0.2rem 0 0.6rem 0; }}
      .arch-canvas {{
          position:relative; min-width:1080px; height:620px; margin:0 auto;
          background:{PAL['surface']}; border:1px solid {PAL['border']}; border-radius:18px;
          overflow:hidden;
      }}
      .arch-lines {{ position:absolute; inset:0; width:100%; height:100%; z-index:1; pointer-events:none; }}
      .arch-card {{
          position:absolute; z-index:2; box-sizing:border-box; border-radius:16px;
          border:1px solid {PAL['border']}; padding:14px 16px; line-height:1.25;
          box-shadow:0 4px 14px rgba(0,0,0,0.06); overflow:hidden;
      }}
      .arch-card h4 {{ margin:0 0 7px 0; font-size:1rem; }}
      .arch-card p {{ margin:0; font-size:0.82rem; color:{PAL['ink_2']} !important; }}
      .arch-brine {{ left:3%; top:8%; width:21%; height:20%; background:{'#163a5f' if DARK_MODE else '#e7f2ff'}; }}
      .arch-twin {{ left:3%; top:42%; width:21%; height:20%; background:{'#173f35' if DARK_MODE else '#e8f8ef'}; }}
      .arch-chariot {{ left:31%; top:24%; width:25%; height:28%; background:{'#513915' if DARK_MODE else '#fff4dc'}; }}
      .arch-dashboard {{ left:64%; top:8%; width:22%; height:20%; background:{'#342b66' if DARK_MODE else '#f0ecff'}; }}
      .arch-gemini {{ left:64%; top:42%; width:22%; height:20%; background:{'#542635' if DARK_MODE else '#ffeaf0'}; }}
      .arch-bond {{ left:88%; top:42%; width:10.5%; height:20%; background:{'#20365f' if DARK_MODE else '#e9efff'}; padding:12px; }}
      .arch-user {{ left:73%; top:76%; width:20%; height:17%; background:{'#26343e' if DARK_MODE else '#eef2f5'}; }}
      .arch-safety {{ left:3%; top:77%; width:53%; height:15%; background:{PAL['surface_2']}; }}
      .arch-note {{ position:absolute; z-index:3; color:{PAL['muted']}; font-size:0.75rem; }}
      .arch-note-bond {{ left:84%; top:36%; }}
      .arch-note-user {{ left:78%; top:69%; }}

      /* ------------------------------------------------------------------
         TOP PRODUCT NAVIGATION
         ------------------------------------------------------------------ */
      .st-key-top_nav_shell {{
          position: sticky !important;
          top: 0 !important;
          z-index: 9990 !important;
          background: {PAL['surface']} !important;
          border: 1px solid {PAL['border']} !important;
          border-radius: 14px !important;
          padding: 0.42rem 0.48rem !important;
          margin: 0 0 1rem 0 !important;
          box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
          backdrop-filter: blur(12px);
      }}
      .st-key-top_nav_shell [data-testid="stHorizontalBlock"] {{ gap:0.35rem !important; }}
      .st-key-top_nav_shell button {{
          min-height:2.55rem !important; border-radius:10px !important;
          font-weight:700 !important; white-space:nowrap !important;
      }}
      .st-key-top_nav_shell button[kind="tertiary"],
      .st-key-top_nav_shell button[kind="secondary"] {{
          background:transparent !important; color:{PAL['ink_2']} !important;
          border-color:transparent !important;
      }}
      .st-key-top_nav_shell button[kind="tertiary"]:hover,
      .st-key-top_nav_shell button[kind="secondary"]:hover {{
          background:{PAL['surface_2']} !important; color:{PAL['ink']} !important;
      }}
      .st-key-top_nav_shell button[kind="primary"] {{
          background:{PAL['primary']} !important; color:#ffffff !important;
          border-color:{PAL['primary']} !important;
          box-shadow:0 4px 12px rgba(102,88,207,0.25) !important;
      }}

      /* ------------------------------------------------------------------
         NON-MODAL SCIENTIST CHAT
         A normal Streamlit container is pinned to the bottom-right.
         ------------------------------------------------------------------ */
      .st-key-chat_launcher {{
          position:fixed !important; right:1.4rem !important; bottom:1.25rem !important;
          width:auto !important; z-index:9997 !important;
      }}
      .st-key-chat_launcher button {{
          border-radius:999px !important; padding:0.78rem 1.15rem !important;
          background:{PAL['primary']} !important; color:#ffffff !important;
          border:1px solid rgba(255,255,255,0.18) !important;
          box-shadow:0 12px 34px rgba(0,0,0,0.26) !important;
          font-weight:800 !important;
      }}
      .st-key-chat_compact_panel {{
          position:fixed !important; right:1.35rem !important; bottom:1.2rem !important;
          width:min(420px, calc(100vw - 2rem)) !important;
          max-height:min(690px, calc(100vh - 3rem)) !important; overflow-y:auto !important;
          z-index:9998 !important; background:{PAL['surface']} !important;
          color:{PAL['ink_2']} !important; border:1px solid {PAL['border']} !important;
          border-radius:20px !important; padding:0.85rem 0.9rem 0.9rem 0.9rem !important;
          box-shadow:0 22px 70px rgba(0,0,0,0.34) !important;
      }}
      .st-key-chat_compact_panel [data-testid="stVerticalBlockBorderWrapper"] {{
          background:transparent !important; border-color:{PAL['border']} !important;
      }}
      .st-key-chat_compact_panel [data-testid="stForm"] {{ background:transparent !important; border:0 !important; }}
      .st-key-chat_compact_panel input,
      .st-key-chat_compact_panel div[data-baseweb="input"],
      .st-key-chat_compact_panel div[data-baseweb="base-input"] {{
          background:{PAL['input']} !important; color:{PAL['ink']} !important;
          -webkit-text-fill-color:{PAL['ink']} !important; border-color:{PAL['border']} !important;
      }}
      .st-key-chat_compact_panel button {{ color:{PAL['ink']} !important; }}
      .chat-history {{ max-height:300px; overflow:auto; padding-right:0.2rem; }}
      .chat-bubble {{ padding:0.62rem 0.72rem; border-radius:12px; margin:0.45rem 0; font-size:0.88rem; }}
      .chat-user {{ background:{PAL['surface_2']}; border:1px solid {PAL['border']}; }}
      .chat-assistant {{ background:rgba(144,133,233,0.10); border:1px solid rgba(144,133,233,0.22); }}
      .chat-meta {{ color:{PAL['muted']}; font-size:0.72rem; margin-top:0.3rem; }}

      .st-key-full_chat_controls {{
          position:sticky !important; top:3.7rem !important; z-index:8500 !important;
          background:{PAL['surface']} !important; border:1px solid {PAL['border']} !important;
          border-radius:14px !important; padding:0.45rem 0.55rem !important;
          margin-bottom:0.85rem !important; box-shadow:0 5px 16px rgba(0,0,0,0.08) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# APPLY PENDING BROWSER SELECTIONS  (before text-input widgets are built)
# ============================================================================

def _apply_pending_path(pending_key, authoritative_key, widget_key):
    selected = st.session_state.get(pending_key)
    if selected:
        st.session_state[authoritative_key] = selected
        st.session_state[widget_key] = selected
        st.session_state[pending_key] = None
        _save_persisted_paths()


_apply_pending_path("pending_campaign_path", "campaign_path", "campaign_path_input")
_apply_pending_path("pending_prediction_path", "prediction_path", "prediction_path_input")
_apply_pending_path("pending_decision_path", "decision_path", "decision_path_input")
_apply_pending_path("pending_config_path", "config_path", "config_path_input")


# ============================================================================
# NATIVE WINDOWS FILE PICKER
# ============================================================================

def _initial_directory(current_value):
    if not current_value:
        return str(Path.cwd())
    try:
        p = Path(current_value).expanduser()
        if p.is_file():
            return str(p.parent)
        if p.parent.exists():
            return str(p.parent)
    except Exception:
        pass
    return str(Path.cwd())


def _pick_existing_file(current_value, title, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askopenfilename(
            title=title, initialdir=_initial_directory(current_value), filetypes=filetypes)
        root.destroy()
        return str(Path(selected).resolve()) if selected else None
    except Exception as exc:
        st.sidebar.error(f"Could not open the Windows file picker: {exc}")
        return None


def _pick_output_file(current_value, title, default_name):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.asksaveasfilename(
            title=title, initialdir=_initial_directory(current_value),
            initialfile=default_name, defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        root.destroy()
        return str(Path(selected).resolve()) if selected else None
    except Exception as exc:
        st.sidebar.error(f"Could not open the Windows file picker: {exc}")
        return None


# ============================================================================
# MANUAL TEXT-INPUT CALLBACKS
# ============================================================================

def _campaign_path_changed():
    st.session_state["campaign_path"] = st.session_state["campaign_path_input"]
    _save_persisted_paths()


def _prediction_path_changed():
    st.session_state["prediction_path"] = st.session_state["prediction_path_input"]
    _save_persisted_paths()


def _decision_path_changed():
    st.session_state["decision_path"] = st.session_state["decision_path_input"]
    _save_persisted_paths()


def _config_path_changed():
    st.session_state["config_path"] = st.session_state["config_path_input"]
    _save_persisted_paths()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("BRINE × CHARIOT")
    st.caption("Live scientist-facing observer")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch the dashboard and scientific figures between dark and light presentation modes.",
    )

    st.divider()
    st.caption("DATA SOURCES")

    st.markdown("**BRINE campaign CSV**")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.text_input("BRINE campaign CSV", key="campaign_path_input",
                      label_visibility="collapsed", on_change=_campaign_path_changed)
    with c2:
        if st.button("Browse", key="browse_brine", use_container_width=True):
            selected = _pick_existing_file(st.session_state["campaign_path"],
                                           "Select BRINE campaign CSV",
                                           [("CSV files", "*.csv"), ("All files", "*.*")])
            if selected:
                st.session_state["pending_campaign_path"] = selected
                st.rerun()

    st.markdown("**Twin predictions CSV**")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.text_input("Twin predictions CSV", key="prediction_path_input",
                      label_visibility="collapsed", on_change=_prediction_path_changed)
    with c2:
        if st.button("Browse", key="browse_twin", use_container_width=True):
            selected = _pick_existing_file(st.session_state["prediction_path"],
                                           "Select digital-twin predictions CSV",
                                           [("CSV files", "*.csv"), ("All files", "*.*")])
            if selected:
                st.session_state["pending_prediction_path"] = selected
                st.rerun()

    st.markdown("**CHARIOT decision log**")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.text_input("CHARIOT decision log", key="decision_path_input",
                      label_visibility="collapsed", on_change=_decision_path_changed)
    with c2:
        if st.button("Open", key="browse_decision", use_container_width=True):
            selected = _pick_existing_file(st.session_state["decision_path"],
                                           "Select existing CHARIOT decision log",
                                           [("CSV files", "*.csv"), ("All files", "*.*")])
            if selected:
                st.session_state["pending_decision_path"] = selected
                st.rerun()
    with c3:
        if st.button("New", key="new_decision", use_container_width=True):
            selected = _pick_output_file(st.session_state["decision_path"],
                                         "Choose CHARIOT decision-log location",
                                         "chariot_decision_log.csv")
            if selected:
                st.session_state["pending_decision_path"] = selected
                st.rerun()

    st.markdown("**Config YAML (optional)**")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.text_input("Config YAML", key="config_path_input",
                      label_visibility="collapsed", on_change=_config_path_changed)
    with c2:
        if st.button("Browse", key="browse_config", use_container_width=True):
            selected = _pick_existing_file(st.session_state["config_path"],
                                           "Select CHARIOT config YAML",
                                           [("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
            if selected:
                st.session_state["pending_config_path"] = selected
                st.rerun()

    st.caption("Selected data-source paths are saved and survive browser refreshes.")

    st.divider()
    st.caption("SYSTEM")

    gemini_configured = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if gemini_configured:
        st.markdown(
            "<span class='status-dot status-ready'></span>**Gemini configured**",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}")
    else:
        st.markdown(
            "<span class='status-dot status-off'></span>**Gemini not configured**",
            unsafe_allow_html=True,
        )
        st.caption("Add GEMINI_API_KEY to the project .env file.")

    bond_configured = bool(os.getenv("BOND_BEARER_TOKEN"))
    if bond_configured:
        st.markdown(
            f"<span class='status-dot status-ready'></span>**Valency Bond ready**",
            unsafe_allow_html=True,
        )
        st.caption("Queried only when you explicitly ask for literature or papers.")
    else:
        st.markdown(
            f"<span class='status-dot status-off'></span>**Valency Bond not configured**",
            unsafe_allow_html=True,
        )
        st.caption("Add BOND_BEARER_TOKEN to the project .env file.")

    st.toggle(
        "Use Valency Bond for literature",
        key="bond_enabled",
        disabled=not bond_configured,
        help="This does not search on ordinary campaign questions. Bond is called only for explicit literature requests.",
    )
    st.toggle("Auto-refresh dashboard", key="auto_refresh")
    st.slider("Refresh interval (s)", 2, 15, key="refresh_seconds")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("**Safety boundary**")
    st.caption("CHARIOT is shadow-only. Gemini is read-only. The twin predicts conductivity only.")


# ============================================================================
# HELPERS
# ============================================================================

def _path(value):
    return Path(value).expanduser()


def _config_value():
    value = st.session_state["config_path"].strip()
    return value or None


def _build_snapshot():
    return build_dashboard_snapshot(
        _path(st.session_state["campaign_path"]),
        _path(st.session_state["prediction_path"]),
        _path(st.session_state["decision_path"]),
        _config_value(),
    )


def _fmt(value, digits=2, suffix=""):
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _trial(value):
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _action_label(action):
    mapping = {"run_lsv": "RUN LSV", "review_lsv": "REVIEW", "skip_lsv": "SKIP LSV",
               "wait_for_eis": "WAIT FOR EIS", "wait_for_twin": "WAIT FOR TWIN"}
    return mapping.get(str(action), str(action or "—").upper())


def _action_pill(action):
    cls = {"run_lsv": "pill-run", "review_lsv": "pill-review", "skip_lsv": "pill-skip"}.get(str(action))
    if not cls:
        return _action_label(action)
    return f"<span class='pill {cls}'>{_action_label(action)}</span>"


def _render_theme_table(df: pd.DataFrame, max_rows: int = 100):
    """Render a compact, theme-safe scrollable table in light or dark mode."""
    if df is None or df.empty:
        st.info("No rows available yet.")
        return
    shown = df.head(max_rows).copy()
    table_html = shown.to_html(index=False, escape=True, classes="theme-table", border=0)
    st.markdown(f"<div class='table-shell'>{table_html}</div>", unsafe_allow_html=True)


def _render_decision_summary(snapshot):
    """Readable CHARIOT status cards for the Shadow Decisions page."""
    run_n = int(snapshot.get("run_lsv", 0) or 0)
    review_n = int(snapshot.get("review_lsv", 0) or 0)
    skip_n = int(snapshot.get("skip_lsv", 0) or 0)
    ready_n = int(snapshot.get("model_ready_counts", {}).get("true", 0) or 0)
    disagreement_n = int(snapshot.get("review_breakdown", {}).get("twin_eis_disagreement", 0) or 0)

    cards = f"""
    <div class='decision-grid'>
      <div class='decision-card run'>
        <div class='decision-label'>RUN LSV</div><div class='decision-value'>{run_n}</div>
        <div class='decision-desc'>Conductivity evidence supports proceeding to the destructive LSV characterization.</div>
      </div>
      <div class='decision-card review'>
        <div class='decision-label'>REVIEW</div><div class='decision-value'>{review_n}</div>
        <div class='decision-desc'>Evidence is not strong enough for an automatic run/skip decision. Escalate rather than discard.</div>
      </div>
      <div class='decision-card skip'>
        <div class='decision-label'>SKIP LSV</div><div class='decision-value'>{skip_n}</div>
        <div class='decision-desc'>EIS is below target and the calibrated twin also supports a confident below-target decision.</div>
      </div>
      <div class='decision-card ready'>
        <div class='decision-label'>MODEL-READY</div><div class='decision-value'>{ready_n}</div>
        <div class='decision-desc'>Trials where enough prior calibration evidence existed for uncertainty-aware automatic triage.</div>
      </div>
      <div class='decision-card disagree'>
        <div class='decision-label'>DISAGREEMENT REVIEWS</div><div class='decision-value'>{disagreement_n}</div>
        <div class='decision-desc'>Twin and measured EIS differed beyond the configured disagreement threshold, so CHARIOT forced REVIEW.</div>
      </div>
    </div>
    <div class='decision-key'>
      <span class='key-item'><span class='dot' style='background:{PAL["run"]}'></span>RUN = proceed to LSV</span>
      <span class='key-item'><span class='dot' style='background:{PAL["review"]}'></span>REVIEW = uncertainty/conflict</span>
      <span class='key-item'><span class='dot' style='background:{PAL["skip"]}'></span>SKIP = avoid LSV in shadow</span>
      <span class='key-item'><span class='dot' style='background:#e76f51'></span>Disagreement = twin vs EIS conflict</span>
    </div>
    """
    st.markdown(cards, unsafe_allow_html=True)


def _flag_explanation(flag):
    mapping = {
        "EIS_QUALIFIED": "EIS meets conductivity target",
        "MODEL_NOT_READY": "Cold start / insufficient calibration",
        "PREDICTION_BAND_CROSSES_TARGET": "Twin uncertainty overlaps the threshold",
        "TWIN_EIS_DISAGREEMENT": "Twin and EIS conflict beyond review threshold",
        "CONFIDENT_BELOW_TARGET": "EIS and calibrated twin both support below-target",
    }
    return mapping.get(str(flag), str(flag).replace("_", " ").title())


def _render_decision_table(decisions: pd.DataFrame, max_rows: int = 100):
    """Decision log with action/flag colors and plain-language flag meaning."""
    if decisions is None or decisions.empty:
        st.info("No frozen decisions yet.")
        return

    preferred = [
        "trial_index", "chariot_shadow_action", "chariot_flag_code",
        "measured_eis_conductivity_S_per_m", "twin_predicted_conductivity_S_per_m",
        "twin_raw_uncertainty_S_per_m", "normalized_disagreement_score",
        "model_ready", "training_data_last_trial", "prediction_composition_verified",
    ]
    cols = [c for c in preferred if c in decisions.columns]
    d = decisions[cols].sort_values("trial_index", ascending=False).head(max_rows).copy()

    headers = {
        "trial_index": "Trial",
        "chariot_shadow_action": "CHARIOT action",
        "chariot_flag_code": "Why",
        "measured_eis_conductivity_S_per_m": "Measured EIS (S/m)",
        "twin_predicted_conductivity_S_per_m": "Twin prediction (S/m)",
        "twin_raw_uncertainty_S_per_m": "Raw uncertainty (S/m)",
        "normalized_disagreement_score": "Disagreement score",
        "model_ready": "Model ready",
        "training_data_last_trial": "Training through",
        "prediction_composition_verified": "Composition verified",
    }

    def cell(col, value):
        if pd.isna(value):
            return "—"
        if col == "chariot_shadow_action":
            action = str(value)
            cls = {"run_lsv":"pill-run", "review_lsv":"pill-review", "skip_lsv":"pill-skip"}.get(action, "")
            label = {"run_lsv":"RUN LSV", "review_lsv":"REVIEW", "skip_lsv":"SKIP LSV"}.get(action, action)
            return f"<span class='pill {cls}'>{html.escape(label)}</span>"
        if col == "chariot_flag_code":
            flag = str(value)
            cls = "flag-review"
            if flag == "EIS_QUALIFIED": cls = "flag-ready"
            elif flag == "CONFIDENT_BELOW_TARGET": cls = "flag-skip"
            elif flag == "TWIN_EIS_DISAGREEMENT": cls = "flag-disagree"
            return f"<span class='flag-pill {cls}'>{html.escape(_flag_explanation(flag))}</span>"
        if col in {"measured_eis_conductivity_S_per_m", "twin_predicted_conductivity_S_per_m", "twin_raw_uncertainty_S_per_m", "normalized_disagreement_score"}:
            try: return f"{float(value):.2f}"
            except Exception: pass
        if col in {"model_ready", "prediction_composition_verified"}:
            return "✓" if bool(value) else "—"
        try:
            if col in {"trial_index", "training_data_last_trial"}: return str(int(float(value)))
        except Exception:
            pass
        return html.escape(str(value))

    out = ["<div class='table-shell'><table class='theme-table'><thead><tr>"]
    for c in cols:
        out.append(f"<th>{html.escape(headers.get(c,c))}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in d.iterrows():
        action = str(row.get("chariot_shadow_action", ""))
        accent = ACTION_COLOR.get(action, PAL["muted"])
        out.append(f"<tr style='border-left:4px solid {accent}'>")
        for c in cols:
            out.append(f"<td>{cell(c, row[c])}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def _architecture_component_html() -> str:
    """Self-contained responsive architecture card used inside an iframe.

    Using components.html avoids Streamlit/Markdown interpreting the HTML as code.
    """
    if DARK_MODE:
        bg = "#171817"
        panel = "#202220"
        ink = "#f3f4f6"
        muted = "#c4c7cc"
        line = "#8f949c"
        cards = {
            "candidate": ("#173b2a", "#63d497"),
            "twin": ("#4b3511", "#f3bf4a"),
            "bond": ("#34255f", "#b49bff"),
            "brine": ("#173958", "#62b0ff"),
            "chariot": ("#57262b", "#ff858f"),
            "gemini": ("#173a58", "#66b5ff"),
            "data": ("#32275a", "#aa95ff"),
            "user": ("#27323a", "#aebcc7"),
        }
    else:
        bg = "#ffffff"
        panel = "#fbfcfe"
        ink = "#172033"
        muted = "#526078"
        line = "#506278"
        cards = {
            "candidate": ("#e7f8ed", "#2b9b57"),
            "twin": ("#fff5db", "#d89c14"),
            "bond": ("#f1ebff", "#7b5bd6"),
            "brine": ("#e9f4ff", "#448fe1"),
            "chariot": ("#fff0ec", "#e46a55"),
            "gemini": ("#eaf5ff", "#368bd3"),
            "data": ("#f0ecff", "#8066d9"),
            "user": ("#eef2f6", "#75879a"),
        }

    def card_style(name):
        fill, border = cards[name]
        return f"background:{fill};border:2px solid {border};"

    return textwrap.dedent(f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{ box-sizing:border-box; }}
        html, body {{ margin:0; padding:0; background:{bg}; font-family:Segoe UI,Arial,sans-serif; color:{ink}; }}
        .frame {{ position:relative; width:100%; height:585px; background:{panel}; border:1px solid rgba(125,125,125,.20); border-radius:18px; overflow:hidden; }}
        .card {{ position:absolute; border-radius:14px; padding:12px 13px; box-shadow:0 2px 6px rgba(0,0,0,.05); overflow:hidden; }}
        .card h3 {{ margin:0 0 7px 0; font-size:15px; line-height:1.15; color:{ink}; }}
        .card p {{ margin:0; font-size:12.2px; line-height:1.38; color:{muted}; }}
        .small h3 {{ font-size:13px; }} .small p {{ font-size:11.5px; }}
        .candidate {{ left:3%; top:8%; width:23%; height:17%; {card_style('candidate')} }}
        .twin {{ left:37%; top:7%; width:24%; height:19%; {card_style('twin')} }}
        .bond {{ right:3%; top:7%; width:25%; height:24%; {card_style('bond')} }}
        .brine {{ left:3%; top:37%; width:25%; height:22%; {card_style('brine')} }}
        .chariot {{ left:37%; top:37%; width:25%; height:22%; {card_style('chariot')} }}
        .gemini {{ right:3%; top:39%; width:25%; height:24%; {card_style('gemini')} }}
        .data {{ left:3%; bottom:7%; width:25%; height:20%; {card_style('data')} }}
        .user {{ right:3%; bottom:6%; width:25%; height:18%; {card_style('user')} }}
        .note {{ position:absolute; font-size:10.5px; color:{muted}; background:{panel}; padding:2px 5px; border-radius:5px; }}
        .bondnote {{ right:15%; top:33%; }}
        .usernote {{ right:13%; bottom:26%; }}
        svg {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
        @media (max-width:720px) {{
          .card p {{ font-size:10.5px; }} .card h3 {{ font-size:13px; }}
        }}
      </style>
    </head>
    <body>
      <div class="frame">
        <svg viewBox="0 0 1000 585" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <marker id="end" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="{line}"/></marker>
            <marker id="start" markerWidth="9" markerHeight="9" refX="1" refY="3" orient="auto"><path d="M8,0 L8,6 L0,3 z" fill="{line}"/></marker>
          </defs>
          <line x1="260" y1="100" x2="365" y2="100" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="145" y1="145" x2="145" y2="215" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="280" y1="285" x2="365" y2="285" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="145" y1="345" x2="145" y2="425" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="490" y1="150" x2="490" y2="215" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="620" y1="285" x2="720" y2="285" stroke="{line}" stroke-width="2.2" marker-end="url(#end)"/>
          <line x1="845" y1="170" x2="845" y2="225" stroke="{line}" stroke-width="2.2" marker-start="url(#start)" marker-end="url(#end)"/>
          <line x1="845" y1="370" x2="845" y2="445" stroke="{line}" stroke-width="2.2" marker-start="url(#start)" marker-end="url(#end)"/>
          <line x1="278" y1="505" x2="725" y2="505" stroke="{line}" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#end)"/>
        </svg>

        <div class="card candidate small"><h3>Composition candidates</h3><p>From BO / campaign plan</p></div>
        <div class="card twin"><h3>Digital twin</h3><p>Conductivity only.<br>Writes a frozen pre-trial prediction + raw uncertainty.</p></div>
        <div class="card bond"><h3>Valency Bond</h3><p>Retrieves relevant papers, ranks evidence, and provides DOI / source links for explicit literature questions.</p></div>
        <div class="card brine"><h3>BRINE</h3><p>Runs the physical electrolyte experiment and writes measured EIS / LSV data.</p></div>
        <div class="card chariot"><h3>CHARIOT</h3><p><b>Shadow-only triage.</b><br>Compares EIS with the frozen twin evidence and records RUN / REVIEW / SKIP.</p></div>
        <div class="card gemini"><h3>Gemini</h3><p>Read-only scientist interface. Explains current results, decisions, trends, and literature context.</p></div>
        <div class="card data"><h3>Experimental data</h3><p>Conductivity, HER / OER, composition, and frozen CHARIOT log.</p></div>
        <div class="card user"><h3>Scientist / user</h3><p>Two-way conversation with Gemini about the live campaign, plots, decisions, or papers.</p></div>
        <div class="note bondnote">explicit literature request ⇅</div>
        <div class="note usernote">scientist conversation ⇅</div>
      </div>
    </body>
    </html>
    """)


# ============================================================================
# CHARTS
# ============================================================================

def _render_conductivity_chart(snapshot):
    fig = plot_twin_vs_eis(snapshot["campaign"], snapshot["predictions"], snapshot["cfg"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_action_chart(snapshot):
    decisions = snapshot["decisions"].copy()
    if decisions.empty:
        st.info("No frozen CHARIOT decisions yet.")
        return
    mapping = {"skip_lsv": 0, "review_lsv": 1, "run_lsv": 2}
    d = decisions.copy()
    d["action_numeric"] = d["chariot_shadow_action"].map(mapping)
    d = d.dropna(subset=["trial_index", "action_numeric"])
    if d.empty:
        st.info("No RUN / REVIEW / SKIP decisions yet.")
        return

    colors = d["chariot_shadow_action"].map(ACTION_COLOR).fillna(PAL["muted"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    # Soft decision bands make the categorical meaning obvious at a glance.
    ax.axhspan(-0.45, 0.45, color=PAL["skip"], alpha=0.08, zorder=0)
    ax.axhspan(0.55, 1.45, color=PAL["review"], alpha=0.08, zorder=0)
    ax.axhspan(1.55, 2.45, color=PAL["run"], alpha=0.08, zorder=0)
    ax.scatter(d["trial_index"], d["action_numeric"], s=72, c=list(colors),
               edgecolors=PAL["surface"], linewidths=0.9, zorder=3)
    ax.set_yticks([0, 1, 2], ["SKIP LSV", "REVIEW", "RUN LSV"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Trial")
    ax.set_title("Frozen CHARIOT decisions by trial")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_disagreement_chart(snapshot):
    decisions = snapshot["decisions"].copy()
    if decisions.empty:
        st.info("No frozen CHARIOT decisions yet.")
        return
    x = pd.to_numeric(decisions["trial_index"], errors="coerce")
    y = pd.to_numeric(decisions["normalized_disagreement_score"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        st.info("No normalized disagreement scores available yet.")
        return
    threshold = float(snapshot["cfg"].get("triage", {}).get("disagreement_score_threshold", 1.0))
    xv, yv = x[valid], y[valid]
    point_colors = ["#e76f51" if v >= threshold else PAL["data"] for v in yv]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axhspan(threshold, max(float(yv.max()) * 1.10, threshold + 0.15), color="#e76f51", alpha=0.07, zorder=0)
    ax.plot(xv, yv, color=PAL["data"], linewidth=1.7, alpha=0.9, zorder=2)
    ax.scatter(xv, yv, s=46, c=point_colors, edgecolors=PAL["surface"], linewidths=0.8, zorder=3)
    ax.axhline(threshold, linestyle=(0, (3, 3)), color=PAL["review"], linewidth=1.35,
               label=f"Review threshold = {threshold:g}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Normalized disagreement")
    ax.set_title("Twin–EIS disagreement: orange points trigger REVIEW")
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_scientific_plots(snapshot):
    st.subheader("1. Composition + EIS conductivity")
    st.caption("Formulation composition and measured ionic conductivity versus experiment number.")
    fig1, mode1 = plot_eis_with_composition(snapshot["campaign"], snapshot["cfg"])
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    if mode1 == "volume":
        st.info("Stock concentrations are not configured, so formulation is shown as "
                "stock-solution volume (µL).")
    st.divider()

    st.subheader("2. Measured EIS vs digital twin")
    st.caption("Measured conductivity compared with the pre-trial digital-twin prediction.")
    fig2 = plot_twin_vs_eis(snapshot["campaign"], snapshot["predictions"], snapshot["cfg"])
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption("The shaded band is the twin's reported raw conductivity uncertainty.")
    st.divider()

    st.subheader("3. LSV endpoints + composition")
    st.caption("HER and OER versus experiment number, with formulation composition aligned below.")
    fig3, mode3 = plot_lsv_with_composition(snapshot["campaign"], snapshot["cfg"])
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)
    if mode3 == "volume":
        st.info("Composition is currently shown in µL because stock molarities have not "
                "yet been configured.")


# ============================================================================
# GEMINI OBSERVER
# ============================================================================

def _get_observer():
    config_key = _config_value() or "__default__"
    if (st.session_state["observer"] is None
            or st.session_state["observer_config_key"] != config_key):
        st.session_state["observer"] = GeminiObserver(config_path=_config_value())
        st.session_state["observer_config_key"] = config_key
    return st.session_state["observer"]


# ============================================================================
# SCIENTIST CHAT HELPERS
# ============================================================================

def _ask_scientist(prompt: str) -> None:
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    try:
        observer = _get_observer()
        literature_requested = observer._explicit_literature_request(prompt)
        bond_enabled = bool(st.session_state["bond_enabled"])

        if literature_requested and not bond_enabled:
            answer = (
                "This question asks for literature, but Valency Bond is disabled or not configured. "
                "Enable the Bond switch in the sidebar and confirm BOND_BEARER_TOKEN is present in .env."
            )
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            return

        spinner_text = (
            "Searching Valency Bond and comparing literature…"
            if literature_requested
            else "Reading the latest campaign state…"
        )
        with st.spinner(spinner_text):
            response = observer.analyze(
                prompt,
                campaign_file=_path(st.session_state["campaign_path"]),
                prediction_log_file=_path(st.session_state["prediction_path"]),
                decision_log_file=_path(st.session_state["decision_path"]),
                use_bond=bond_enabled,
            )

        stored = {"role": "assistant", "content": response.answer}
        literature_status = getattr(observer, "last_literature_status", {}) or {}
        service_status = getattr(observer, "last_service_status", {}) or {}
        if literature_status.get("queried"):
            stored["literature_status"] = literature_status
        if service_status:
            stored["service_status"] = service_status

        if response.plot_request is not None:
            cfg = observer.cfg
            df = load_brine_csv(_path(st.session_state["campaign_path"]), cfg)
            decisions = load_decision_log(_path(st.session_state["decision_path"]))
            shadow_view = build_shadow_view(df, decisions)
            plot_path = Path("outputs") / "dashboard_chat_plot.png"
            create_plot(
                df, shadow_view, response.plot_request.plot_type, plot_path, cfg,
                input_name=response.plot_request.input_name,
            )
            stored["plot_path"] = str(plot_path)

        st.session_state["chat_messages"].append(stored)

    except Exception as exc:
        service_status = {}
        try:
            service_status = getattr(observer, "last_service_status", {}) or {}
        except Exception:
            pass

        details = []
        gemini_status = service_status.get("gemini", {}) or {}
        bond_status = service_status.get("bond", {}) or {}
        if bond_status.get("queried"):
            details.append("Bond succeeded" if bond_status.get("ok") else "Bond failed")
        if gemini_status.get("ok") is False:
            details.append("Gemini failed after retries")

        suffix = f" ({'; '.join(details)})" if details else ""
        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": f"Could not answer from the live system{suffix}: {exc}",
            "service_status": service_status,
        })


def _rerun_app():
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()


def _rerun_fragment():
    try:
        st.rerun(scope="fragment")
    except (TypeError, Exception):
        st.rerun()


def _open_compact_chat():
    st.session_state["compact_chat_open"] = True
    _rerun_app()


def _close_compact_chat():
    st.session_state["compact_chat_open"] = False
    _rerun_app()


def _maximize_chat_from(page: str):
    if page != "Scientist Chat":
        st.session_state["last_non_chat_page"] = page
    st.session_state["compact_chat_open"] = False
    st.session_state["pending_page"] = "Scientist Chat"
    _rerun_app()


def _render_chat_launcher(page: str):
    """Render a non-modal bottom-right assistant panel."""
    if page == "Scientist Chat":
        return

    if not st.session_state.get("compact_chat_open", False):
        with st.container(key="chat_launcher"):
            if st.button("🧠 Scientist Chat", key="open_scientist_chat", use_container_width=True):
                _open_compact_chat()
        return

    with st.container(key="chat_compact_panel"):
        h1, h2, h3 = st.columns([5.2, 1.0, 1.0], vertical_alignment="center")
        with h1:
            st.markdown("### 🧠 Scientist Chat")
            st.caption("Gemini · live BRINE/CHARIOT context")
        with h2:
            if st.button("—", key="compact_chat_minimize", help="Minimize", use_container_width=True):
                _close_compact_chat()
        with h3:
            if st.button("↗", key="compact_chat_maximize", help="Maximize", use_container_width=True):
                _maximize_chat_from(page)

        _render_chat_panel_contents(compact=True)


def _render_chat_panel_contents(compact: bool = False):
    st.markdown("#### Scientist chat")
    st.caption("Live BRINE + CHARIOT context. Bond is added only for explicit literature questions.")

    messages = st.session_state.get("chat_messages", [])
    shown = messages[-6:] if compact else messages

    if not shown:
        st.markdown(
            "<div class='chat-hero'><strong>Ask the live scientist interface</strong><br>"
            "<span class='mini-muted'>Examples: What is the current maximum? Why did CHARIOT review the latest trial? Compare the best result with literature.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in shown:
            who = "You" if msg.get("role") == "user" else "Gemini"
            with st.container(border=True):
                st.caption(who)
                st.markdown(str(msg.get("content", "")))
                lit = msg.get("literature_status") or {}
                if lit.get("queried"):
                    st.caption(
                        "Valency Bond · "
                        f"{lit.get('papers_retrieved', 0)} papers retrieved · "
                        f"{lit.get('papers_used', 0)} relevant papers used"
                    )

                svc = msg.get("service_status") or {}
                gem = svc.get("gemini", {}) or {}
                bond = svc.get("bond", {}) or {}
                status_bits = []
                if gem.get("ok") is True:
                    mode = gem.get("mode") or "ok"
                    status_bits.append(f"Gemini ✓ ({mode})")
                elif gem.get("ok") is False:
                    status_bits.append("Gemini unavailable — local fallback used")
                if bond.get("queried") and bond.get("ok") is True:
                    status_bits.append("Bond ✓")
                elif bond.get("queried") and bond.get("ok") is False:
                    status_bits.append("Bond failed")
                if status_bits:
                    st.caption(" · ".join(status_bits))

                plot_path = msg.get("plot_path")
                if plot_path and Path(plot_path).exists():
                    st.image(str(plot_path), use_container_width=True)

    form_key = "compact_scientist_chat_form" if compact else "full_scientist_chat_form"
    with st.form(form_key, clear_on_submit=True):
        prompt = st.text_input(
            "Question",
            placeholder="Ask about BRINE, CHARIOT, plots, or literature…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and prompt.strip():
        _ask_scientist(prompt)
        if compact:
            _rerun_fragment()
        else:
            _rerun_app()


def _render_full_chat_page():
    st.title("Scientist Chat")
    st.caption("Gemini reads the current campaign state on every question. Valency Bond is called only for explicit literature requests.")

    with st.container(key="full_chat_controls"):
        c1, c2, spacer = st.columns([1.05, 1.55, 5.4])
        with c1:
            if st.button("← Return", key="full_chat_return", use_container_width=True):
                st.session_state["compact_chat_open"] = False
                st.session_state["pending_page"] = st.session_state.get("last_non_chat_page", "Dashboard")
                _rerun_app()
        with c2:
            if st.button("— Minimize to corner", key="full_chat_minimize", use_container_width=True):
                st.session_state["compact_chat_open"] = True
                st.session_state["pending_page"] = st.session_state.get("last_non_chat_page", "Dashboard")
                _rerun_app()

    left, right = st.columns([3.1, 1.0])
    with left:
        _render_chat_panel_contents(compact=False)
    with right:
        st.markdown("### Live context")
        try:
            snap = _build_snapshot()
            st.metric("Trials", snap["status"]["campaign_rows"])
            st.metric("Best conductivity", _fmt(snap["best_conductivity_S_per_m"], 2, " S/m"))
            latest = snap.get("latest_decision") or {}
            st.markdown("**Latest CHARIOT**")
            st.markdown(_action_pill(latest.get("chariot_shadow_action")), unsafe_allow_html=True)
            st.caption(str(latest.get("chariot_reason") or "No frozen decision yet."))
        except Exception as exc:
            st.info(f"Live context unavailable: {exc}")

        st.markdown("### Literature")
        if os.getenv("BOND_BEARER_TOKEN") and st.session_state.get("bond_enabled"):
            st.success("Valency Bond ready")
            st.caption("Only queried when you explicitly ask for papers / literature.")
        else:
            st.warning("Valency Bond disabled or not configured")

# ============================================================================
# TOP NAVIGATION
# ============================================================================

NAV_PAGES = [
    ("Dashboard", "⌂ Dashboard"),
    ("Campaign", "▦ Campaign"),
    ("Shadow Decisions", "◇ Shadow Decisions"),
    ("Analysis", "⌁ Analysis"),
    ("Scientist Chat", "✦ Scientist Chat"),
    ("About", "ⓘ About"),
]


def _go_to_page(destination: str):
    current = st.session_state.get("active_page", "Dashboard")
    if destination == "Scientist Chat" and current != "Scientist Chat":
        st.session_state["last_non_chat_page"] = current
        st.session_state["compact_chat_open"] = False
    elif destination != "Scientist Chat":
        st.session_state["last_non_chat_page"] = destination
    st.session_state["active_page"] = destination
    _rerun_app()


def _render_top_navigation() -> str:
    active = st.session_state.get("active_page", "Dashboard")
    with st.container(key="top_nav_shell"):
        cols = st.columns([1.0, 1.0, 1.45, 0.9, 1.35, 0.75], gap="small")
        for col, (page_name, label) in zip(cols, NAV_PAGES):
            with col:
                if st.button(
                    label,
                    key=f"nav_{page_name.lower().replace(' ', '_')}",
                    type="primary" if page_name == active else "tertiary",
                    use_container_width=True,
                ):
                    _go_to_page(page_name)
    return st.session_state.get("active_page", "Dashboard")


page = _render_top_navigation()

# ============================================================================
# HEADER
# ============================================================================

if page == "Dashboard":
    st.title("BRINE × CHARIOT")
    st.markdown(
        "<div class='app-sub'><span class='live-dot'></span>"
        "Self-driving electrolyte laboratory · shadow characterization triage · scientist-facing AI.</div>"
        "<span class='chariot-badge'>SHADOW ONLY</span>"
        "<span class='chariot-badge'>READ-ONLY GEMINI</span>"
        "<span class='chariot-badge'>CONDUCTIVITY TWIN</span>",
        unsafe_allow_html=True,
    )


# ============================================================================
# LIVE DASHBOARD
# ============================================================================

def _render_live_dashboard():
    try:
        snapshot = _build_snapshot()
    except Exception as exc:
        st.error(f"Could not rebuild live state: {exc}")
        return

    status = snapshot["status"]
    summary = snapshot["summary"]
    campaign = snapshot["campaign"]
    shadow_view = snapshot["shadow_view"]

    integrity_errors = snapshot.get("prediction_integrity_errors", [])
    if integrity_errors:
        st.error("Twin prediction integrity errors: " + "; ".join(integrity_errors))

    # Count completed LSV measurements from the normalized BRINE columns.
    # load_brine_csv() maps the configured source columns (e.g. y2/y3) to
    # internal columns named "her" and "oer".  A completed LSV trial should
    # contain both endpoint potentials.
    if "her" in campaign.columns and "oer" in campaign.columns:
        her_values = pd.to_numeric(campaign["her"], errors="coerce")
        oer_values = pd.to_numeric(campaign["oer"], errors="coerce")
        n_lsv = int((her_values.notna() & oer_values.notna()).sum())
    else:
        n_lsv = 0

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total trials", status["campaign_rows"])
    metric_cols[1].metric("Best conductivity", _fmt(snapshot["best_conductivity_S_per_m"], 2, " S/m"))
    metric_cols[2].metric("Trials with LSV", n_lsv)
    metric_cols[3].metric("CHARIOT SKIP", snapshot["skip_lsv"])
    metric_cols[4].metric("CHARIOT REVIEW", snapshot["review_lsv"])
    metric_cols[5].metric("Status", "Live" if status["campaign_rows"] else "Waiting")

    chart_col, recent_col = st.columns([2.0, 1.0], gap="medium")
    with chart_col:
        with st.container(border=True):
            st.markdown("### Ionic conductivity · measured vs twin")
            _render_conductivity_chart(snapshot)

    with recent_col:
        with st.container(border=True):
            st.markdown("### Recent trials")
            recent = shadow_view.copy()
            desired = ["trial_index", "x1", "x2", "x3", "x4", "conductivity", "y1", "chariot_shadow_action"]
            cols = [c for c in desired if c in recent.columns]
            if "conductivity" in cols and "y1" in cols:
                cols.remove("y1")
            if cols:
                recent = recent[cols].sort_values("trial_index", ascending=False).head(7)
                recent = recent.rename(columns={
                    "trial_index": "Trial", "x1": "NaCl", "x2": "KCl", "x3": "ZnCl₂", "x4": "ZnSO₄",
                    "conductivity": "Cond. (S/m)", "y1": "Cond. (S/m)", "chariot_shadow_action": "Decision",
                })
                _render_theme_table(recent, max_rows=7)
            else:
                st.info("Recent campaign rows will appear here.")

    left, right = st.columns([1.1, 1.4], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("### Latest CHARIOT shadow decision")
            latest = snapshot.get("latest_decision") or {}
            if latest:
                st.markdown(
                    f"**Trial {_trial(latest.get('trial_index'))}** &nbsp; "
                    f"{_action_pill(latest.get('chariot_shadow_action'))}",
                    unsafe_allow_html=True,
                )
                measured = latest.get("measured_eis_conductivity_S_per_m")
                if measured is not None:
                    st.write(f"Measured conductivity: **{_fmt(measured, 2, ' S/m')}**")
                st.caption(str(latest.get("chariot_reason") or ""))
            else:
                st.info("No frozen CHARIOT decision yet.")

    with right:
        with st.container(border=True):
            st.markdown("### Key observations")
            target = snapshot["target_S_per_m"]
            pass_n = summary.get("n_conductivity_at_or_above_target", 0)
            total_n = status["campaign_rows"] or 0
            pass_pct = (100.0 * pass_n / total_n) if total_n else 0.0
            disagreement_n = snapshot["review_breakdown"].get("twin_eis_disagreement", 0) or 0
            model_ready_n = snapshot["model_ready_counts"].get("true", 0) or 0
            st.markdown(
                f"- Current best conductivity is **{_fmt(snapshot['best_conductivity_S_per_m'], 2, ' S/m')}**.\n"
                f"- **{pass_n}/{total_n} trials ({pass_pct:.1f}%)** are at or above the **{target:g} S/m** screening target.\n"
                f"- CHARIOT has frozen **{snapshot['run_lsv']} RUN**, **{snapshot['review_lsv']} REVIEW**, and **{snapshot['skip_lsv']} SKIP** recommendations.\n"
                f"- **{disagreement_n}** REVIEW decisions were triggered by twin–EIS disagreement.\n"
                f"- The uncertainty calibration was model-ready for **{model_ready_n}** frozen decisions."
            )


# ============================================================================
# ABOUT PAGE EXPLAINER FLOWS
# These are appended below the existing About page. They do not alter the
# existing architecture/content above.
# ============================================================================

def _flow_theme():
    if DARK_MODE:
        return {
            "bg": "#151515",
            "surface": "#1f1f1e",
            "surface2": "#252523",
            "border": "rgba(255,255,255,0.13)",
            "ink": "#f0efec",
            "muted": "#aaa89f",
            "blue": "#173b63",
            "blue_border": "#5aa0ef",
            "green": "#163f34",
            "green_border": "#4ecb96",
            "amber": "#503913",
            "amber_border": "#f1b84b",
            "rose": "#512735",
            "rose_border": "#e0789d",
            "violet": "#342d62",
            "violet_border": "#9b8df2",
            "slate": "#26343e",
            "slate_border": "#8ca0b3",
            "arrow": "#9a9993",
        }
    return {
        "bg": "#ffffff",
        "surface": "#ffffff",
        "surface2": "#f7f7f4",
        "border": "rgba(20,20,20,0.12)",
        "ink": "#171717",
        "muted": "#6f6f69",
        "blue": "#e8f3ff",
        "blue_border": "#6ba7e8",
        "green": "#e9f8ef",
        "green_border": "#5cba88",
        "amber": "#fff5df",
        "amber_border": "#e5ad45",
        "rose": "#ffedf1",
        "rose_border": "#df7f9c",
        "violet": "#f0edff",
        "violet_border": "#9587ea",
        "slate": "#eef3f6",
        "slate_border": "#8fa5b5",
        "arrow": "#76766f",
    }


def _chariot_explained_html() -> str:
    c = _flow_theme()
    return f"""
    <html><head><style>
      * {{ box-sizing:border-box; }}
      body {{ margin:0; background:transparent; font-family:Segoe UI, Arial, sans-serif; color:{c['ink']}; }}
      .wrap {{ width:100%; }}
      .flow-card {{
        background:{c['surface']}; border:1px solid {c['border']}; border-radius:16px;
        padding:16px 18px 18px; margin:0 0 14px 0;
      }}
      .flow-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }}
      .flow-title {{ font-size:17px; font-weight:750; color:{c['ink']}; margin:0; }}
      .flow-kicker {{ font-size:12px; color:{c['muted']}; margin-top:3px; }}
      .badge {{ font-size:11px; font-weight:700; color:{c['muted']}; border:1px solid {c['border']};
                padding:5px 9px; border-radius:999px; white-space:nowrap; background:{c['surface2']}; }}
      .flow {{ display:flex; align-items:center; gap:10px; width:100%; overflow:hidden; }}
      .node {{ min-height:92px; flex:1 1 0; border-radius:14px; padding:13px 14px; border:1.5px solid; display:flex;
               flex-direction:column; justify-content:center; text-align:center; }}
      .node b {{ font-size:14px; color:{c['ink']}; margin-bottom:5px; }}
      .node span {{ font-size:11.5px; line-height:1.35; color:{c['muted']}; }}
      .blue {{ background:{c['blue']}; border-color:{c['blue_border']}; }}
      .green {{ background:{c['green']}; border-color:{c['green_border']}; }}
      .amber {{ background:{c['amber']}; border-color:{c['amber_border']}; }}
      .rose {{ background:{c['rose']}; border-color:{c['rose_border']}; }}
      .violet {{ background:{c['violet']}; border-color:{c['violet_border']}; }}
      .slate {{ background:{c['slate']}; border-color:{c['slate_border']}; }}
      .arrow {{ flex:0 0 42px; position:relative; height:2px; background:{c['arrow']}; }}
      .arrow:after {{ content:''; position:absolute; right:-1px; top:-5px; width:0; height:0;
                      border-top:6px solid transparent; border-bottom:6px solid transparent;
                      border-left:9px solid {c['arrow']}; }}
      .gate {{ flex:1.15 1 0; }}
      .outcomes {{ flex:1.15 1 0; display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
      .mini {{ border-radius:10px; padding:10px 6px; text-align:center; border:1px solid {c['border']}; background:{c['surface2']}; }}
      .mini b {{ font-size:11.5px; color:{c['ink']}; display:block; }}
      .mini span {{ font-size:10px; color:{c['muted']}; display:block; margin-top:2px; }}
      .foot {{ margin-top:10px; color:{c['muted']}; font-size:11px; line-height:1.4; }}
      @media (max-width: 900px) {{ .flow {{ overflow-x:auto; padding-bottom:4px; }} .node {{ min-width:180px; }} .gate {{ min-width:220px; }} .outcomes {{ min-width:270px; }} }}
    </style></head><body><div class='wrap'>

      <div class='flow-card'>
        <div class='flow-head'><div><div class='flow-title'>Our electrolyte workflow</div><div class='flow-kicker'>Use fast conductivity evidence before spending a destructive LSV measurement.</div></div><div class='badge'>Current BRINE × CHARIOT use case</div></div>
        <div class='flow'>
          <div class='node blue'><b>Candidate electrolyte</b><span>NaCl · KCl · ZnCl₂ · ZnSO₄ formulation</span></div>
          <div class='arrow'></div>
          <div class='node green'><b>EIS conductivity</b><span>Fast, non-destructive screening measurement</span></div>
          <div class='arrow'></div>
          <div class='node amber gate'><b>CHARIOT gate</b><span>Does the conductivity evidence justify destructive LSV?</span></div>
          <div class='arrow'></div>
          <div class='outcomes'>
            <div class='mini'><b>RUN</b><span>perform LSV</span></div>
            <div class='mini'><b>REVIEW</b><span>evidence uncertain</span></div>
            <div class='mini'><b>SKIP</b><span>avoid LSV</span></div>
          </div>
          <div class='arrow'></div>
          <div class='node rose'><b>LSV</b><span>HER / OER stability endpoints when justified</span></div>
        </div>
      </div>

      <div class='flow-card'>
        <div class='flow-head'><div><div class='flow-title'>Example: automated corrosion screening</div><div class='flow-kicker'>A later, more aggressive polarization test can be gated by earlier electrochemical evidence.</div></div><div class='badge'>Generalized CHARIOT concept</div></div>
        <div class='flow'>
          <div class='node blue'><b>Candidate alloy / coating</b><span>Automated sample queue</span></div>
          <div class='arrow'></div>
          <div class='node green'><b>OCP + EIS + LPR</b><span>Earlier automated corrosion measurements</span></div>
          <div class='arrow'></div>
          <div class='node amber gate'><b>CHARIOT gate</b><span>Is the evidence strong enough to justify the next corrosion test?</span></div>
          <div class='arrow'></div>
          <div class='outcomes'>
            <div class='mini'><b>RUN</b><span>advance</span></div>
            <div class='mini'><b>REVIEW</b><span>ambiguous</span></div>
            <div class='mini'><b>STOP</b><span>do not advance</span></div>
          </div>
          <div class='arrow'></div>
          <div class='node rose'><b>CPP</b><span>Cyclic potentiodynamic polarization</span></div>
        </div>
      </div>

      <div class='flow-card'>
        <div class='flow-head'><div><div class='flow-title'>Example: automated materials characterization</div><div class='flow-kicker'>Use fast structural/property screening to decide whether a candidate deserves additional automated characterization.</div></div><div class='badge'>Generalized CHARIOT concept</div></div>
        <div class='flow'>
          <div class='node blue'><b>New material / film</b><span>Automated synthesis output</span></div>
          <div class='arrow'></div>
          <div class='node green'><b>XRD + conductivity</b><span>Fast structural and property screen</span></div>
          <div class='arrow'></div>
          <div class='node amber gate'><b>CHARIOT gate</b><span>Does the candidate satisfy the structural/property criteria?</span></div>
          <div class='arrow'></div>
          <div class='outcomes'>
            <div class='mini'><b>ADVANCE</b><span>characterize</span></div>
            <div class='mini'><b>REVIEW</b><span>uncertain</span></div>
            <div class='mini'><b>ITERATE</b><span>resynthesize</span></div>
          </div>
          <div class='arrow'></div>
          <div class='node violet'><b>Raman / SEM / UV–Vis</b><span>Additional automated characterization only when useful</span></div>
        </div>
      </div>

      <div class='foot'><b>Core idea:</b> CHARIOT does not choose or control the experiment. It sits between characterization stages and asks whether the available evidence justifies spending the next measurement.</div>
    </div></body></html>
    """


def _llm_bond_explained_html() -> str:
    c = _flow_theme()
    return f"""
    <html><head><style>
      * {{ box-sizing:border-box; }}
      body {{ margin:0; background:transparent; font-family:Segoe UI, Arial, sans-serif; color:{c['ink']}; }}
      .shell {{ background:{c['surface']}; border:1px solid {c['border']}; border-radius:16px; padding:18px; }}
      .flow {{ display:flex; align-items:center; gap:11px; overflow-x:auto; padding-bottom:4px; }}
      .node {{ min-height:104px; min-width:175px; flex:1 0 175px; border-radius:14px; padding:14px; border:1.5px solid;
               display:flex; flex-direction:column; justify-content:center; text-align:center; }}
      .node b {{ font-size:14px; color:{c['ink']}; margin-bottom:5px; }}
      .node span {{ font-size:11.5px; line-height:1.35; color:{c['muted']}; }}
      .blue {{ background:{c['blue']}; border-color:{c['blue_border']}; }}
      .green {{ background:{c['green']}; border-color:{c['green_border']}; }}
      .amber {{ background:{c['amber']}; border-color:{c['amber_border']}; }}
      .rose {{ background:{c['rose']}; border-color:{c['rose_border']}; }}
      .violet {{ background:{c['violet']}; border-color:{c['violet_border']}; }}
      .slate {{ background:{c['slate']}; border-color:{c['slate_border']}; }}
      .arrow {{ flex:0 0 38px; position:relative; height:2px; background:{c['arrow']}; }}
      .arrow:after {{ content:''; position:absolute; right:-1px; top:-5px; width:0; height:0;
                      border-top:6px solid transparent; border-bottom:6px solid transparent; border-left:9px solid {c['arrow']}; }}
      .router {{ min-width:210px; }}
      .fork {{ min-width:250px; display:grid; gap:8px; }}
      .route {{ border:1px solid {c['border']}; border-radius:11px; padding:10px; background:{c['surface2']}; text-align:left; }}
      .route b {{ font-size:12px; display:block; color:{c['ink']}; margin-bottom:3px; }}
      .route span {{ font-size:10.8px; color:{c['muted']}; line-height:1.3; }}
      .examples {{ margin-top:13px; display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
      .ex {{ border:1px solid {c['border']}; border-radius:12px; padding:11px 13px; background:{c['surface2']}; }}
      .ex b {{ color:{c['ink']}; font-size:12px; }}
      .ex div {{ margin-top:5px; color:{c['muted']}; font-size:11px; line-height:1.45; }}
      .foot {{ margin-top:12px; color:{c['muted']}; font-size:11px; line-height:1.4; }}
    </style></head><body>
      <div class='shell'>
        <div class='flow'>
          <div class='node slate'><b>Scientist asks a question</b><span>Live campaign, CHARIOT, plots, mechanisms, or literature</span></div>
          <div class='arrow'></div>
          <div class='node rose'><b>Gemini</b><span>Read-only scientist interface</span></div>
          <div class='arrow'></div>
          <div class='node amber router'><b>Question router</b><span>Does this require only live campaign evidence, or external literature?</span></div>
          <div class='arrow'></div>
          <div class='fork'>
            <div class='route'><b>LIVE CAMPAIGN QUESTION</b><span>Read BRINE + twin + CHARIOT files. No Bond call.</span></div>
            <div class='route'><b>EXPLICIT LITERATURE QUESTION</b><span>Query Valency Bond for papers, DOI links, and evidence.</span></div>
          </div>
          <div class='arrow'></div>
          <div class='node violet'><b>Evidence assembled</b><span>Current campaign state + validated external literature when requested</span></div>
          <div class='arrow'></div>
          <div class='node green'><b>Grounded scientist answer</b><span>Gemini explains the result and shows sources when literature was used</span></div>
        </div>
        <div class='examples'>
          <div class='ex'><b>Answered from the live campaign</b><div>“What is our current maximum?” · “Which formulation performed best?” · “Why did CHARIOT review Trial 40?” · “Where do twin and EIS disagree?”</div></div>
          <div class='ex'><b>Routes through Valency Bond</b><div>“Compare our maximum with literature.” · “Has this chemistry been reported?” · “Find papers supporting this mechanism.” · “What values have been published?”</div></div>
        </div>
        <div class='foot'><b>Routing rule:</b> ordinary campaign questions stay local. Valency Bond is called only when the scientist explicitly asks for literature, papers, published values, or an external comparison.</div>
      </div>
    </body></html>
    """


# ============================================================================
# ABOUT PAGE
# ============================================================================

def _render_about_page():
    st.title("About this system")
    st.caption("BRINE × CHARIOT: a self-driving electrolyte laboratory with shadow decision support, AI analysis, and literature integration.")

    overview_col, architecture_col, info_col = st.columns([0.9, 1.75, 0.95], gap="medium")

    with overview_col:
        with st.container(border=True):
            st.markdown("### Overview")
            st.write(
                "This system combines autonomous electrolyte experimentation (BRINE), a conductivity digital twin, "
                "CHARIOT shadow triage, Gemini scientific interpretation, and Valency Bond literature support."
            )
        with st.container(border=True):
            st.markdown("**🧪 BRINE**")
            st.caption("Physical campaign")
            st.write("Runs candidate electrolytes and writes measured EIS / LSV results.")
        with st.container(border=True):
            st.markdown("**⚙️ CHARIOT**")
            st.caption("Shadow decision layer")
            st.write("Evaluates conductivity evidence and recommends RUN / REVIEW / SKIP without controlling BRINE.")
        with st.container(border=True):
            st.markdown("**🧠 Gemini + Bond**")
            st.caption("Scientist interface + literature")
            st.write("Explains live results and, when asked, grounds literature statements in retrieved papers.")

    with architecture_col:
        with st.container(border=True):
            st.markdown("### System architecture")
            components.html(_architecture_component_html(), height=610, scrolling=False)

    with info_col:
        with st.container(border=True):
            st.markdown("### Key points")
            st.markdown(
                "- **Read-only Gemini:** does not control the laboratory.\n"
                "- **Shadow-only CHARIOT:** recommendations do not alter BRINE.\n"
                "- **Conductivity-only twin:** no HER, OER, or LSV prediction.\n"
                "- **Bond on demand:** used only for explicit literature requests.\n"
                "- **Fresh state:** answers are rebuilt from the current campaign files."
            )
        with st.container(border=True):
            st.markdown("### Variables")
            st.markdown(
                "- `x1` = NaCl stock amount (µL)\n"
                "- `x2` = KCl stock amount (µL)\n"
                "- `x3` = ZnCl₂ stock amount (µL)\n"
                "- `x4` = ZnSO₄ stock amount (µL)\n"
                "- `y1` = ionic conductivity (S/m)\n"
                "- HER / OER = LSV endpoint potentials (V)\n"
                "- Total formulation volume = 1200 µL"
            )
        with st.container(border=True):
            st.markdown("### Safety boundary")
            st.write("BRINE runs the experiment. CHARIOT is shadow-only. Gemini and Valency Bond are read-only.")

    # ------------------------------------------------------------------------
    # Appended explainer sections — existing About content above is unchanged.
    # ------------------------------------------------------------------------
    st.divider()
    st.markdown("## CHARIOT explained")
    st.caption("CHARIOT acts as a stage-gating layer between characterization steps: use earlier evidence to decide whether the next measurement is worth spending.")
    components.html(_chariot_explained_html(), height=650, scrolling=False)

    st.markdown("## Gemini + Valency Bond explained")
    st.caption("Gemini answers from the live campaign by default and only brings in Valency Bond when the scientist explicitly asks for external literature.")
    components.html(_llm_bond_explained_html(), height=365, scrolling=False)

# ============================================================================
# PAGE SWITCHING + AUTO REFRESH
# ============================================================================

def _render_campaign_page():
    st.title("Campaign")
    try:
        snapshot = _build_snapshot()
    except Exception as exc:
        st.error(f"Could not rebuild live state: {exc}")
        return
    _render_scientific_plots(snapshot)
    st.subheader("Campaign data")
    _render_theme_table(snapshot["shadow_view"].sort_values("trial_index", ascending=False), max_rows=100)


def _render_shadow_page():
    st.title("Shadow Decisions")
    st.caption("CHARIOT is a shadow-only stage gate: it uses measured EIS plus the frozen pre-trial twin evidence to decide whether LSV is justified, uncertain, or avoidable.")
    try:
        snapshot = _build_snapshot()
    except Exception as exc:
        st.error(f"Could not rebuild live state: {exc}")
        return

    _render_decision_summary(snapshot)

    c1, c2 = st.columns(2)
    with c1:
        _render_action_chart(snapshot)
    with c2:
        _render_disagreement_chart(snapshot)
        st.caption(
            "Disagreement review = the measured EIS and frozen twin prediction conflict more than the configured normalized threshold. "
            "CHARIOT does not trust either signal blindly; it escalates the trial to REVIEW."
        )

    st.subheader("Frozen decision log")
    st.caption("Each row is frozen once the pre-trial twin prediction and measured EIS are both available. Later LSV values do not rewrite the decision.")
    _render_decision_table(snapshot["decisions"], max_rows=100)


def _render_analysis_page():
    st.title("Analysis")
    try:
        snapshot = _build_snapshot()
    except Exception as exc:
        st.error(f"Could not rebuild live state: {exc}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best conductivity", _fmt(snapshot["best_conductivity_S_per_m"], 2, " S/m"))
    c2.metric("Mean conductivity", _fmt(snapshot["mean_conductivity_S_per_m"], 2, " S/m"))
    c3.metric("At/above target", snapshot["summary"].get("n_conductivity_at_or_above_target", 0))
    c4.metric("Frozen decisions", snapshot["status"]["frozen_chariot_decisions"])
    st.subheader("Measured EIS vs digital twin")
    _render_conductivity_chart(snapshot)
    st.subheader("Twin–EIS disagreement")
    _render_disagreement_chart(snapshot)


if page == "Dashboard":
    if st.session_state["auto_refresh"] and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{st.session_state['refresh_seconds']}s")
        def _live_fragment():
            _render_live_dashboard()
        _live_fragment()
    else:
        _render_live_dashboard()
elif page == "Campaign":
    _render_campaign_page()
elif page == "Shadow Decisions":
    _render_shadow_page()
elif page == "Analysis":
    _render_analysis_page()
elif page == "Scientist Chat":
    _render_full_chat_page()
elif page == "About":
    _render_about_page()

# Compact assistant launcher is available from every page except the full chat page.
_render_chat_launcher(page)
