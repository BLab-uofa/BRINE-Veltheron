# ax_v112_nimsos_synced.py — Ax 1.1.2 wrapper with your original specs
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.plot.pareto_utils import get_observed_pareto_frontiers, compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.adapter.registry import Generators

try:
    # Ax ≥1.1
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.generation_node import GenerationStep
except ImportError:
    # Older API
    from ax.generation_strategy import GenerationStrategy, GenerationStep
try:
    # Ax ≥1.1
    from ax.adapter.registry import Generators
except ImportError:
    try:
        # Mid-era
        from ax.modelbridge.registry import Generators
    except ImportError:
        # Very old
        from ax.modelbridge.registry import Models as Generators


class AxV112:
    """
    CSV schema (matches your original Ax code):
      x1..xN, y1..yM, trial_index

    Key features:
    - Eq_CONSTRAINT: compute missing last x via your equality rule
    - Parameter bounds: [20.0, 1040.0] for each effective x (as in your code)
    - Linear parameter constraint: 5*x1 + 4*x2 + 1.5*x3 <= 2140.0
    - Objectives: maximize y1, y2 with thresholds (qNEHVI semantics)
    - Pareto plotting toggle (PARETO_PLOT) with 200-point grid
    - Skip Pareto until ≥ SOBOL_WARMUP completed trials with both outputs
    - Noise-aware ingestion: pass values only (no SEM)
    """

    def __init__(self, input_file, output_file, output_folder, num_objectives, num_proposals, num_inputs, psycho, sobol_warmup=30,
             eq_constraint=False, pareto_plot=False, pareto_html=None):
        
        self.input_file = input_file
        self.output_file = output_file
        self.results_dir = output_folder
        
        self.num_objectives = int(num_objectives)
        self.num_proposals = int(num_proposals)
        self.num_inputs = int(num_inputs)

        if self.num_objectives == 1:
            self.reference_points = None
        elif self.num_objectives == 2:
            self.reference_points = (0.5, 2.0)   # hardcode reference
        elif self.num_objectives == 3:
            self.reference_points = (25.0, 0.1, 2.5)   # hardcode reference
            
        else:
            raise ValueError("Only 1 or 2 or 3 objectives for now.")

        os.makedirs(self.results_dir, exist_ok=True)

        self.psycho: int = int(psycho)
        self.sobol_warmup = int(sobol_warmup)
        self.Eq_CONSTRAINT = bool(eq_constraint)
        self.PARETO_PLOT = bool(pareto_plot)        
        
        if pareto_html is not None:
            self.pareto_html_path = pareto_html
        else:
            fname = f"pareto_{self.psycho:04d}.html"
            self.pareto_html_path = os.path.join(self.results_dir, fname)

        self.ax_json_path = os.path.join(self.results_dir, "ax_client.json")
        self._ax_client: AxClient | None = None


        self._ensure_input_file_exists()
        
    def _rebuild_from_csv(self, client: AxClient, df: pd.DataFrame, input_cols, output_cols) -> pd.DataFrame:
        updated = df.copy()
        
        def _scalar_tid(payload):
            """Extract a plain int trial index from anything (dict/list/Trial/np scalar)."""
            import numpy as np
            # direct ints
            if isinstance(payload, (int, np.integer)):
                return int(payload)
            # strings like "12"
            if isinstance(payload, str):
                try:
                    return int(payload)
                except Exception:
                    pass
            # Ax Trial-like objects
            if hasattr(payload, "index"):
                try:
                    return int(payload.index)
                except Exception:
                    pass
            # numpy/pandas scalar
            if hasattr(payload, "item"):
                try:
                    return int(payload.item())
                except Exception:
                    pass
            # dicts: try common keys, else DFS over values
            if isinstance(payload, dict):
                for k in ("trial_index", "index", "id"):
                    if k in payload:
                        try:
                            return _scalar_tid(payload[k])
                        except Exception:
                            pass
                if "trial" in payload:  # sometimes {"trial": Trial(...), ...}
                    try:
                        return _scalar_tid(payload["trial"])
                    except Exception:
                        pass
                # depth-first over all values
                for v in payload.values():
                    try:
                        return _scalar_tid(v)
                    except Exception:
                        continue
            # lists/tuples/sets: take first extractable
            if isinstance(payload, (list, tuple, set)):
                for v in payload:
                    try:
                        return _scalar_tid(v)
                    except Exception:
                        continue
            raise TypeError(f"Could not extract trial_index from: {repr(payload)}")
        
        
        for i, row in updated.iterrows():
            # skip rows without all x's
            if any(pd.isna(row.get(c)) for c in input_cols):
                continue
            params = {c: float(row[c]) for c in input_cols}
            # ensure we have a valid Ax trial for this row
            tid = row.get("trial_index")
            need_new = pd.isna(tid)
            if not need_new:
                try:
                    _ = client.experiment.trials[int(tid)]
                except Exception:
                    need_new = True
            if need_new:
                new_tid = client.attach_trial(parameters=params)
                print("DEBUG attach returned:", repr(new_tid))
                tid = _scalar_tid(new_tid)
                # write back as scalar to Int64 column
                updated.loc[i, "trial_index"] = tid
            # if outputs exist, complete the trial
            if all(not pd.isna(row.get(m)) for m in output_cols):
                raw = {m: float(row[m]) for m in output_cols}
                try:
                    client.complete_trial(trial_index=int(tid), raw_data=raw)
                except Exception:
                    pass
        # persist artifacts
        updated.to_csv(self.input_file, index=False)
        client.save_to_json_file(self.ax_json_path)
        return updated
    
    

    # ---------- NIMS-OS entry points ----------
    def run_Ax(self):
        df = pd.read_csv(self.input_file)
        df["trial_index"] = pd.to_numeric(df.get("trial_index"), errors="coerce").astype("Int64")
        input_cols, output_cols = self._read_columns(df)

        client = self._load_or_make_client(input_cols, output_cols)

        # Ingest completed rows (values only; unknown noise)
        if not df.empty:
            self._ingest_completed_rows(client, df, output_cols)

        # Propose next point (single suggestion)
        # df = pd.read_csv(self.input_file)  # refresh
        
        
        df = self._rebuild_from_csv(client, df, input_cols, output_cols)
        self._propose_one_and_append(client, df, input_cols, output_cols)

        # Pareto (optional, only if exactly 2 objectives)
        # if self.PARETO_PLOT and self.num_objectives == 2:
        #     try:
        #         self._plot_and_save_pareto(client, output_cols, self.pareto_html_path, grid_points=120)
        #     except Exception as e:
        #         print(f"Pareto plotting skipped due to error: {e}")
        # elif self.PARETO_PLOT and self.num_objectives != 2:
        #     print(f"Pareto plotting skipped: requires exactly 2 objectives (got {self.num_objectives}).")

    def write_results(self, inputs):
        """Optional helper: append a row with proposed inputs and empty ys."""
        df = pd.read_csv(self.input_file)
        input_cols, output_cols = self._read_columns(df)
        row = {**{c: v for c, v in zip(input_cols, inputs)}, **{m: float('nan') for m in output_cols}}
        df.loc[len(df)] = row
        df.to_csv(self.input_file, index=False)

    def select(self):
        print("Start of AxV112!")
        self.run_Ax()
        print("Selection of proposals ended!")
        return True

    # ---------- Internals ----------
    def _ensure_input_file_exists(self):
        if not os.path.exists(self.input_file):
            x_cols = [f"x{i+1}" for i in range(self.num_inputs)]
            y_cols = [f"y{i+1}" for i in range(self.num_objectives)]
            cols = x_cols + y_cols + ["trial_index"]
            pd.DataFrame(columns=cols).to_csv(self.input_file, index=False)
            print(f"Created input file '{self.input_file}' with headers: {cols}")

    def _read_columns(self, df: pd.DataFrame):
        x_cols = [f"x{i+1}" for i in range(self.num_inputs)]
        y_cols = [f"y{i+1}" for i in range(self.num_objectives)]
        missing = [c for c in x_cols + y_cols + ["trial_index"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing expected columns: {missing}")
        return x_cols, y_cols

    def _effective_input_cols(self, input_cols_all):
        if self.Eq_CONSTRAINT:
            if len(input_cols_all) < 2:
                raise ValueError("Eq_CONSTRAINT=True requires at least 2 input columns.")
            return input_cols_all[:-1]  # last x is computed by equality rule
        return input_cols_all

    def _compute_missing_input(self, known_params: dict) -> float:
        # equality filler:
        # 5*x1 + 4*x2 + 1.5*x3 + 3*x_missing = 2200  ->  x_missing = ...
        x1 = known_params.get("x1")
        x2 = known_params.get("x2")
        x3 = known_params.get("x3")
        if any(v is None for v in [x1, x2, x3]):
            raise KeyError("compute_missing_input expects keys 'x1','x2','x3'.")
        x_m = (2200.0 - 4.0 * x1 - 4.0 * x2 - 1.5 * x3) / 3.0
        return float(x_m)

    def _load_or_make_client(self, input_cols_all, output_cols) -> AxClient:
        if self._ax_client is not None:
            return self._ax_client
        
        if os.path.isfile(self.ax_json_path):
            self._ax_client = AxClient.load_from_json_file(self.ax_json_path)
            return self._ax_client
        
        gs = GenerationStrategy(
            steps=[
                GenerationStep(generator=Generators.SOBOL, num_trials=self.sobol_warmup, model_gen_kwargs={"max_rs_draws": 10000000}, should_deduplicate=False),
                GenerationStep(generator=Generators.BOTORCH_MODULAR, num_trials=-1),
            ])


        axc = AxClient(random_seed=42, torch_device="cpu", generation_strategy=gs)

        
        if self.num_objectives == 1:
            objectives = {output_cols[0]: ObjectiveProperties(minimize=False)}
            
        else:
            objectives = {col: ObjectiveProperties(minimize=False, threshold=threshold)
                          for col, threshold in zip(output_cols, self.reference_points)}


        eff_inputs = self._effective_input_cols(input_cols_all)
        # params = [{"name": n, "type": "range", "bounds": [20.0, 1040.0]} for n in eff_inputs]
        params = [
            {"name": "x1", "type": "range", "bounds": [0.0, 1100.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 1100.0]},
            {"name": "x3", "type": "range", "bounds": [0.0, 1100.0]},
            {"name": "x4", "type": "range", "bounds": [0.0, 1100.0]},
        ]

        axc.create_experiment(
            name="BRINE_Vel_MO_Maximise",
            parameters=params,
            objectives=objectives,
            parameter_constraints=["x1 + x2 + x3 + x4 <= 1101.0"], # hardcoded linear constraint (applies if x1..x3 in eff_inputs)
            overwrite_existing_experiment=True,)


        axc.save_to_json_file(self.ax_json_path)
        self._ax_client = axc
        return self._ax_client

    def _ingest_completed_rows(self, client: AxClient, df: pd.DataFrame, output_cols):
        if "trial_index" not in df.columns:
            return
        mask = df["trial_index"].notna()
        for m in output_cols:
            mask &= df[m].notna()
        for _, row in df[mask].iterrows():
            tid = int(row["trial_index"])
            raw = {m: float(row[m]) for m in output_cols}  # values only (noise-aware)
            try:
                client.complete_trial(trial_index=tid, raw_data=raw)
            except Exception:
                pass
        client.save_to_json_file(self.ax_json_path)

    def _propose_one_and_append(self, client: AxClient, df: pd.DataFrame, input_cols_all, output_cols):
        params, tid = client.get_next_trial()
        # params["x1"] = max(params["x1"] - 0.1 , 20.0) # Zinc MUST BE PRESENT, but Bullshit
        if self.Eq_CONSTRAINT:
            computed_name = input_cols_all[-1]
            computed_value = self._compute_missing_input(params)
            full_inputs = {**params, computed_name: float(computed_value)}
            ordered_inputs = {name: full_inputs.get(name) for name in input_cols_all}
            row = {**ordered_inputs, **{m: None for m in output_cols}, "trial_index": tid}
        else:
            row = {**params, **{m: None for m in output_cols}, "trial_index": tid}

        df.loc[len(df)] = row
        df.to_csv(self.input_file, index=False)
        client.save_to_json_file(self.ax_json_path)
        print(f"Proposed params (trial {tid}): {params}")
        print(f"-> Added to {self.input_file}. Fill {output_cols} for trial_index={tid}.")


    # Pareto (2 objectives only)
    def _plot_and_save_pareto(self, client: AxClient, output_cols, html_path: str, grid_points: int = 120):
        exp = client.experiment
        data = exp.fetch_data()
        df_long = data.df

        # Require ≥ self.sobol_warmup completed trials with both outputs
        have_both = (
            df_long[df_long["metric_name"].isin([output_cols[0], output_cols[1]])]
            .groupby("trial_index")["metric_name"].nunique()
        )
        completed_both = int((have_both == 2).sum())
        if completed_both < self.sobol_warmup:
            print(f"Pareto: skipping (only {completed_both} completed trials; need ≥ {self.sobol_warmup}).")
            return

        fronts = get_observed_pareto_frontiers(exp)
        fronts = [f for f in fronts if hasattr(f, "means") and f.means is not None]
        if not fronts:
            print("Pareto: no observed frontier yet.")
            return
        observed = fronts[-1]

        primary_metric = exp.metrics[output_cols[0]]
        secondary_metric = exp.metrics[output_cols[1]]
        modeled = compute_posterior_pareto_frontier(
            experiment=exp,
            data=data,
            primary_objective=primary_metric,
            secondary_objective=secondary_metric,
            absolute_metrics=[primary_metric.name, secondary_metric.name],
            num_points=grid_points,
        )

        # --- Matplotlib (all trials in gray; observed in red; modeled in blue) ---
        wide = (
            df_long[df_long["metric_name"].isin([output_cols[0], output_cols[1]])]
            .pivot_table(index=["trial_index", "arm_name"], columns="metric_name", values="mean")
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=160)
        if set([output_cols[0], output_cols[1]]).issubset(wide.columns):
            ax.scatter(
                wide[output_cols[1]],  # x: y2
                wide[output_cols[0]],  # y: y1
                color="gray", alpha=0.5, s=16, label="All trials"
            )

        ax.scatter(
            observed.means[output_cols[1]],
            observed.means[output_cols[0]],
            color="tab:red", s=28, label="Observed Pareto"
        )

        ax.plot(
            modeled.means[secondary_metric.name],
            modeled.means[primary_metric.name],
            color="tab:blue", lw=1.5, label="Modelled Pareto"
        )

        ax.set_xlabel(secondary_metric.name)
        ax.set_ylabel(primary_metric.name)
        ax.set_title("Pareto Frontier")
        ax.grid(alpha=0.35)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5.0)  # keep open for 5s
        plt.close(fig)

        # --- Save interactive Plotly HTML (no inline Plotly shown) ---
        mod_cfg = plot_pareto_frontier(frontier=modeled, CI_level=0.90)  # Ax 1.1.2 signature
        fig_html = go.Figure(mod_cfg.data["data"], mod_cfg.data["layout"])
        pio.write_html(fig_html, html_path, include_plotlyjs="cdn")
        print(f"✅ Saved interactive Pareto plot → {html_path}")
