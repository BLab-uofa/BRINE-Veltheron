# ax_v112_nimsos_csv_authoritative.py
# Ax 1.1.2 / NIMS-OS wrapper
#
# The input CSV, NOT a previous Ax client or JSON snapshot, is the sole source
# of experimental history. Every run recreates the Ax experiment from scratch.

import hashlib
import os
import tempfile
from numbers import Integral

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.plot.pareto_utils import get_observed_pareto_frontiers, compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

try:
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.generation_node import GenerationStep
except ImportError:
    from ax.generation_strategy import GenerationStrategy, GenerationStep

try:
    from ax.adapter.registry import Generators
except ImportError:
    try:
        from ax.modelbridge.registry import Generators
    except ImportError:
        from ax.modelbridge.registry import Models as Generators


class AxV112:
    """
    CSV schema:
        x1, x2, ..., xN, y1, y2, ..., yM, [trial_index]

    CSV is authoritative:
        * Every select()/run_Ax() call constructs a NEW AxClient.
        * Existing x/y rows are reattached directly from the CURRENT CSV.
        * Deleted/edited CSV experiments disappear/change immediately.
        * Old trial_index values are ignored and reassigned on reconstruction.
        * ax_client.json is only an optional output snapshot; it is NEVER loaded.
        * Any incomplete objective row is attached as a pending trial.

    Behaviour retained from the supplied file:
        * Single next proposal per run_Ax() call (num_proposals kept for API compatibility).
        * Maximisation of all objectives, with original reference thresholds.
        * Ax search bounds: 0 to 1100 for optimised inputs.
        * Sum of optimised inputs <= 1101 (same for the normal 4-input setup).
        * Optional equality-computed fourth input (Eq_CONSTRAINT).
        * Optional two-objective Pareto plotting.
    """

    def __init__(self, input_file, output_file, output_folder, num_objectives, num_proposals, num_inputs, psycho, sobol_warmup=30,
                 eq_constraint=False, pareto_plot=False, pareto_html=None):
        self.input_file = input_file
        self.output_file = output_file
        self.results_dir = output_folder
        self.num_objectives = int(num_objectives)
        self.num_proposals = int(num_proposals)
        self.num_inputs = int(num_inputs)
        self.psycho = int(psycho)
        self.sobol_warmup = int(sobol_warmup)
        self.Eq_CONSTRAINT = bool(eq_constraint)
        self.PARETO_PLOT = bool(pareto_plot)

        if self.num_objectives == 1:
            self.reference_points = None
        elif self.num_objectives == 2:
            self.reference_points = (0.5, 2.0)
        elif self.num_objectives == 3:
            self.reference_points = (25.0, 0.1, 2.5)
        else:
            raise ValueError("Only 1, 2, or 3 objectives are supported.")

        if self.num_inputs < 1:
            raise ValueError("num_inputs must be at least 1.")
        if self.sobol_warmup < 1:
            raise ValueError("sobol_warmup must be at least 1.")
        if self.Eq_CONSTRAINT and self.num_inputs != 4:
            raise ValueError("Eq_CONSTRAINT=True currently requires x1, x2, x3, x4.")

        os.makedirs(self.results_dir, exist_ok=True)
        self.pareto_html_path = pareto_html or os.path.join(self.results_dir, f"pareto_{self.psycho:04d}.html")
        self.ax_json_path = os.path.join(self.results_dir, "ax_client.json")

        # For inspection only. This is RESET at the start of every run_Ax().
        # It is never used to reconstruct experiment history.
        self._ax_client = None
        self._ensure_input_file_exists()

    # ------------------ NIMS-OS public entry points ------------------

    def select(self):
        print("Start of AxV112 (CSV-authoritative mode)!")
        self.run_Ax()
        print("Selection of proposals ended!")
        return True

    def run_Ax(self):
        """Reconstruct using only the current CSV, then append one Ax proposal."""
        self._ax_client = None  # no cross-run cache, including in the same Python process
        df, input_cols, output_cols = self._load_csv()
        completed_count = int(df[output_cols].notna().all(axis=1).sum())

        # Reattempt client/model failures once. Both attempts use the SAME CSV
        # and class configuration, never a failed/previous AxClient or its JSON.
        for attempt in range(2):
            try:
                client = self._make_fresh_client(input_cols, output_cols, df, completed_count, attempt)
                updated = self._rebuild_from_csv(client, df, input_cols, output_cols)
                updated, params, tid = self._propose_one_and_append(client, updated, input_cols, output_cols)
                break
            except Exception as exc:
                self._ax_client = None
                if attempt == 1:
                    raise
                print(f"Ax run failed ({type(exc).__name__}: {exc}). Retrying with another fresh CSV-only AxClient.")

        # Commit only after a full successful reconstruction AND proposal.
        # A generation/model error never leaves half-rewritten CSV trial IDs.
        self._atomic_write_csv(updated)
        self._ax_client = client
        self._save_json_snapshot(client)

        print(f"Rebuilt Ax from {len(df)} CSV rows ({completed_count} completed).")
        print(f"Proposed params (trial {tid}): {params}")
        print(f"-> Added to {self.input_file}. Fill {output_cols} for trial_index={tid}.")

        if self.PARETO_PLOT and self.num_objectives == 2:
            try:
                self._plot_and_save_pareto(client, output_cols, self.pareto_html_path, grid_points=120)
            except Exception as exc:
                print(f"Pareto plotting skipped due to error: {exc}")
        elif self.PARETO_PLOT:
            print(f"Pareto plotting skipped: requires exactly 2 objectives (got {self.num_objectives}).")

    def write_results(self, inputs):
        """Original helper: append supplied x values as a pending CSV row."""
        df, input_cols, output_cols = self._load_csv()
        if len(inputs) != len(input_cols):
            raise ValueError(f"Expected {len(input_cols)} inputs, received {len(inputs)}.")
        row = {**dict(zip(input_cols, map(float, inputs))), **{name: float("nan") for name in output_cols}, "trial_index": pd.NA}
        df.loc[len(df)] = row
        self._atomic_write_csv(df)

    # ------------------ CSV loading / validation ------------------

    def _ensure_input_file_exists(self):
        if not os.path.exists(self.input_file):
            input_cols = [f"x{i + 1}" for i in range(self.num_inputs)]
            output_cols = [f"y{i + 1}" for i in range(self.num_objectives)]
            columns = input_cols + output_cols + ["trial_index"]
            os.makedirs(os.path.dirname(os.path.abspath(self.input_file)), exist_ok=True)
            pd.DataFrame(columns=columns).to_csv(self.input_file, index=False)
            print(f"Created input file '{self.input_file}' with headers: {columns}")

    def _read_columns(self, df):
        input_cols = [f"x{i + 1}" for i in range(self.num_inputs)]
        output_cols = [f"y{i + 1}" for i in range(self.num_objectives)]
        missing = [name for name in input_cols + output_cols if name not in df.columns]
        if missing:
            raise ValueError(f"CSV missing expected x/y columns: {missing}")
        return input_cols, output_cols

    def _load_csv(self):
        df = pd.read_csv(self.input_file)
        input_cols, output_cols = self._read_columns(df)
        df = df.copy()

        # Optional column: values, if supplied, are NOT authoritative IDs.
        if "trial_index" not in df.columns:
            df["trial_index"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        else:
            df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce").astype("Int64")

        for name in input_cols + output_cols:
            try:
                df[name] = pd.to_numeric(df[name], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV column '{name}' contains a non-numeric value.") from exc
            if df[name].notna().any() and not df[name].dropna().map(lambda v: float("-inf") < float(v) < float("inf")).all():
                raise ValueError(f"CSV column '{name}' contains an infinite value.")

        for row_index, row in df.iterrows():
            missing_x = [name for name in input_cols if pd.isna(row[name])]
            if not missing_x:
                continue
            if len(missing_x) == len(input_cols) and all(pd.isna(row[name]) for name in output_cols):
                continue  # Fully blank CSV row: preserve it, but do not attach it.
            raise ValueError(f"CSV line {row_index + 2} has missing inputs: {missing_x}")

        return df, input_cols, output_cols

    def _atomic_write_csv(self, df):
        path = self.input_file
        directory = os.path.dirname(os.path.abspath(path))
        fd, temp_path = tempfile.mkstemp(prefix=".ax_csv_", suffix=".csv", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                df.to_csv(handle, index=False)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ------------------ Experiment configuration ------------------

    def _effective_input_cols(self, input_cols_all):
        return input_cols_all[:-1] if self.Eq_CONSTRAINT else input_cols_all

    def _compute_missing_input(self, known_params):
        # Preserve the original IMPLEMENTED equality formula (4*x1, not 5*x1):
        # 4*x1 + 4*x2 + 1.5*x3 + 3*x4 = 2200.
        return float((2200.0 - 4.0 * known_params["x1"] - 4.0 * known_params["x2"] - 1.5 * known_params["x3"]) / 3.0)

    def _csv_seed(self, df, input_cols, output_cols, attempt):
        """A repeatable seed derived solely from current CSV x/y values."""
        if df.empty:
            return 42 + attempt
        data = df[input_cols + output_cols].to_csv(index=False, na_rep="", float_format="%.17g")
        digest = hashlib.blake2s(data.encode("utf-8"), digest_size=4).digest()
        return (42 + int.from_bytes(digest, "big") + attempt) % (2**31 - 1)

    def _make_fresh_client(self, input_cols, output_cols, df, completed_count, attempt=0):
        # ATTACHED historical trials are not counted as generated Sobol trials.
        # A fixed SOBOL -> BOTORCH strategy would therefore restart at Sobol
        # on every CSV-only rebuild, even with hundreds of finished rows.
        # Select the phase from the CURRENT number of completed CSV experiments.
        if completed_count < self.sobol_warmup:
            steps = [GenerationStep(generator=Generators.SOBOL, num_trials=-1,
                                    model_gen_kwargs={"max_rs_draws": 10000000}, should_deduplicate=True)]
            phase = "SOBOL"
        else:
            steps = [GenerationStep(generator=Generators.BOTORCH_MODULAR, num_trials=-1)]
            phase = "BOTORCH_MODULAR"

        gs = GenerationStrategy(steps=steps)
        seed = self._csv_seed(df, input_cols, output_cols, attempt)
        client = AxClient(random_seed=seed, torch_device="cpu", generation_strategy=gs)

        if self.num_objectives == 1:
            objectives = {output_cols[0]: ObjectiveProperties(minimize=False)}
        else:
            objectives = {
                name: ObjectiveProperties(minimize=False, threshold=threshold)
                for name, threshold in zip(output_cols, self.reference_points)
            }

        effective_inputs = self._effective_input_cols(input_cols)
        parameters = [{"name": name, "type": "range", "bounds": [0.0, 1100.0]} for name in effective_inputs]

        constraints = []
        if len(effective_inputs) >= 2:
            constraints.append(" + ".join(effective_inputs) + " <= 1101.0")

        # If x4 is computed, restrict optimised x1..x3 so its implied value
        # cannot be negative. Its maximum is always <= 2200/3 < 1100.
        if self.Eq_CONSTRAINT:
            constraints.append("4*x1 + 4*x2 + 1.5*x3 <= 2200.0")

        client.create_experiment(
            name="BRINE_Vel_MO_Maximise",
            parameters=parameters,
            objectives=objectives,
            parameter_constraints=constraints or None,
            overwrite_existing_experiment=True,
        )
        print(f"Building fresh Ax client: {phase}, {completed_count} completed CSV trials, seed={seed}")
        return client

    # ------------------ CSV -> Ax replay and new proposal ------------------

    @staticmethod
    def _extract_trial_index(result):
        # AxClient.attach_trial returns (parameters, trial_index) in Ax 1.1.2.
        # Some other versions have returned a scalar integer instead.
        if isinstance(result, Integral):
            return int(result)
        if isinstance(result, (list, tuple)) and len(result) == 2 and isinstance(result[1], Integral):
            return int(result[1])
        if hasattr(result, "index") and isinstance(result.index, Integral):
            return int(result.index)
        raise TypeError(f"Unexpected return value from AxClient.attach_trial(): {result!r}")

    def _rebuild_from_csv(self, client, df, input_cols, output_cols):
        updated = df.copy()
        updated["trial_index"] = pd.Series(pd.NA, index=updated.index, dtype="Int64")
        effective_inputs = self._effective_input_cols(input_cols)

        for row_index, row in df.iterrows():
            if all(pd.isna(row[name]) for name in input_cols):
                continue

            parameters = {name: float(row[name]) for name in effective_inputs}
            try:
                tid = self._extract_trial_index(client.attach_trial(parameters=parameters))
            except Exception as exc:
                raise ValueError(f"Ax cannot attach CSV line {row_index + 2}, parameters={parameters}: {exc}") from exc
            updated.at[row_index, "trial_index"] = tid

            if all(pd.notna(row[name]) for name in output_cols):
                measurements = {name: float(row[name]) for name in output_cols}
                try:
                    client.complete_trial(trial_index=tid, raw_data=measurements)
                except Exception as exc:
                    raise RuntimeError(f"Ax could not complete CSV line {row_index + 2}, measurements={measurements}: {exc}") from exc
            # Missing y1...yM -> remain as pending, but never make up a y value.

        return updated

    def _propose_one_and_append(self, client, df, input_cols, output_cols):
        # Prevent regenerating a point still present in CSV. Abandoned duplicate
        # candidate trials live only in this in-memory client, never in CSV history.
        for _ in range(20):
            parameters, tid = client.get_next_trial()
            parameters = {name: float(value) for name, value in parameters.items()}
            if self.Eq_CONSTRAINT:
                full_inputs = {**parameters, input_cols[-1]: self._compute_missing_input(parameters)}
            else:
                full_inputs = parameters

            duplicated = any(
                all(abs(float(row[name]) - full_inputs[name]) < 1e-8 for name in input_cols)
                for _, row in df.iterrows() if all(pd.notna(row[name]) for name in input_cols)
            )
            if duplicated:
                client.abandon_trial(trial_index=int(tid), reason="Duplicate of a current CSV row")
                continue

            record = {name: full_inputs[name] for name in input_cols}
            record.update({name: float("nan") for name in output_cols})
            record["trial_index"] = int(tid)
            updated = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            updated["trial_index"] = pd.to_numeric(updated["trial_index"], errors="coerce").astype("Int64")
            return updated, full_inputs, int(tid)

        raise RuntimeError("Ax generated 20 consecutive points already present in the current CSV.")

    def _save_json_snapshot(self, client):
        """Save the new client for diagnostics ONLY; never load it next run."""
        directory = os.path.dirname(os.path.abspath(self.ax_json_path))
        fd, temp_path = tempfile.mkstemp(prefix=".ax_snapshot_", suffix=".json", dir=directory)
        os.close(fd)
        try:
            client.save_to_json_file(temp_path)
            os.replace(temp_path, self.ax_json_path)
        except Exception as exc:
            print(f"Warning: optional Ax JSON snapshot was not saved: {exc}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ------------------ Optional Pareto plot (2 objectives) ------------------

    def _plot_and_save_pareto(self, client, output_cols, html_path, grid_points=120):
        exp = client.experiment
        data = exp.fetch_data()
        df_long = data.df

        if df_long.empty or "metric_name" not in df_long.columns:
            print("Pareto: no observations available.")
            return

        have_both = (
            df_long[df_long["metric_name"].isin([output_cols[0], output_cols[1]])]
            .groupby("trial_index")["metric_name"].nunique()
        )
        completed_both = int((have_both == 2).sum())
        if completed_both < self.sobol_warmup:
            print(f"Pareto: skipping (only {completed_both} completed trials; need >= {self.sobol_warmup}).")
            return

        fronts = get_observed_pareto_frontiers(exp)
        fronts = [front for front in fronts if hasattr(front, "means") and front.means is not None]
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

        wide = (
            df_long[df_long["metric_name"].isin([output_cols[0], output_cols[1]])]
            .pivot_table(index=["trial_index", "arm_name"], columns="metric_name", values="mean")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=160)
        if set(output_cols).issubset(wide.columns):
            ax.scatter(wide[output_cols[1]], wide[output_cols[0]], color="gray", alpha=0.5, s=16, label="All trials")
        ax.scatter(observed.means[output_cols[1]], observed.means[output_cols[0]], color="tab:red", s=28, label="Observed Pareto")
        ax.plot(modeled.means[secondary_metric.name], modeled.means[primary_metric.name], color="tab:blue", lw=1.5, label="Modelled Pareto")
        ax.set_xlabel(secondary_metric.name)
        ax.set_ylabel(primary_metric.name)
        ax.set_title("Pareto Frontier")
        ax.grid(alpha=0.35)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(5.0)
        plt.close(fig)

        mod_cfg = plot_pareto_frontier(frontier=modeled, CI_level=0.90)
        fig_html = go.Figure(mod_cfg.data["data"], mod_cfg.data["layout"])
        pio.write_html(fig_html, html_path, include_plotlyjs="cdn")
        print(f"Saved interactive Pareto plot -> {html_path}")
