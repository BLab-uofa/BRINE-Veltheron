import os
import json
import time

from .config import DTConfig
from .candidate_table import CandidateTable
from .composition_builder import CompositionBuilder
from .phreeqc_features import PhreeqcFeatureBuilder
from .viscosity_laliberte import LaliberteViscosityBuilder
from .mobility_features import MobilityFeatureBuilder
from .nernst_baseline import NernstBaselineBuilder
from .conductivity_theory import ConductivityTheoryBuilder
from .feature_schema import FeatureSchema
from .residual_targets import ResidualTargetBuilder
from .training_table import TrainingTableBuilder
from .residual_model_manager import ResidualModelManager


class DigitalTwinRunner:
    """Generate physics descriptors, fit selected residual models and predict pending rows."""

    def __init__(self, config: DTConfig):
        config.validate()
        self.config = config
        self.table = CandidateTable(config.input_file, config.num_inputs, config.objective_columns)
        self.composition_builder = CompositionBuilder(config)
        self.phreeqc_feature_builder = PhreeqcFeatureBuilder(config)
        self.viscosity_builder = LaliberteViscosityBuilder(config)
        self.mobility_builder = MobilityFeatureBuilder(config)
        self.nernst_builder = NernstBaselineBuilder(config)
        self.conductivity_theory_builder = ConductivityTheoryBuilder(config) if "sigma" in config.objective_targets else None
        self.feature_schema = FeatureSchema(config)
        self.residual_target_builder = ResidualTargetBuilder(config)
        self.training_table_builder = TrainingTableBuilder(config, self.feature_schema)
        self.residual_model_manager = ResidualModelManager(config, self.training_table_builder)

    def run(self):
        self.config.validate()
        print(f"[digital_twin] Start: {self.config.objective_map}")
        df = self.table.load()
        # All dt_* columns are derived outputs. Rebuild them to avoid stale data
        # when composition, physics settings or the selected objectives change.
        df = df.drop(columns=[col for col in df.columns if col.startswith("dt_")])
        input_cols = self.table.input_columns(df)
        objective_cols = self.table.objective_columns(df)
        metadata_cols = self.table.metadata_columns(df)
        pending_mask = self.table.pending_mask(df)
        completed_mask = self.table.completed_mask(df)

        df["dt_cycle_last_seen"] = int(self.config.Psycho)
        df["dt_is_pending"] = pending_mask
        df["dt_is_completed"] = completed_mask
        df["dt_status"] = "completed_history"
        df.loc[pending_mask, "dt_status"] = "pending_measurement"

        df = self.composition_builder.build(df, input_cols)
        df = self.feature_schema.ensure_static_features(df)
        # Density must be known before PHREEQC converts mol/L input to mol/kgw.
        df = self.viscosity_builder.build(df, input_cols)
        df = self.phreeqc_feature_builder.build(df, input_cols)
        df = self.mobility_builder.build(df)
        if {"her", "oer"} & set(self.config.objective_targets):
            df = self.nernst_builder.build(df)
        if self.conductivity_theory_builder is not None:
            df = self.conductivity_theory_builder.build(df)

        df = self.residual_target_builder.build(df)
        self.feature_schema.validate(df)
        df = self.residual_model_manager.run(df)
        mode_columns = [self.residual_model_manager.TARGETS[target]["mode"] for target in self.config.objective_targets]
        df["dt_prediction_mode"] = df[mode_columns].agg(",".join, axis=1)
        self.table.save(df)
        self._write_cycle_report(df, input_cols, objective_cols, metadata_cols, pending_mask, completed_mask)
        print(f"[digital_twin] Finished. Completed: {int(completed_mask.sum())}, pending: {int(pending_mask.sum())}.")
        return True

    def _write_cycle_report(self, df, input_cols, objective_cols, metadata_cols, pending_mask, completed_mask):
        out_dir = os.path.join(self.config.output_folder, self.config.dt_subfolder)
        os.makedirs(out_dir, exist_ok=True)
        stage_columns = ["dt_volume_valid", "dt_viscosity_ok", "dt_phreeqc_ok", "dt_mobility_ok",
                         "dt_nernst_ok", "dt_conductivity_theory_ok"]
        stages = {col: {"ok": int(df[col].sum()), "failed": int((~df[col].astype(bool)).sum())}
                  for col in stage_columns if col in df}
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cycle": int(self.config.Psycho),
            "input_file": self.config.input_file,
            "num_inputs": self.config.num_inputs,
            "num_objectives": self.config.num_objectives,
            "num_dt_objectives": len(self.config.objective_map),
            "objective_map": self.config.objective_map,
            "conductivity_scale_to_S_m": self.config.conductivity_scale_to_S_m,
            "conductivity_viscosity_source": self.config.conductivity_viscosity_source,
            "num_proposals": self.config.num_proposals,
            "input_columns": input_cols,
            "objective_columns": objective_cols,
            "metadata_columns": metadata_cols,
            "n_rows_total": len(df),
            "n_rows_pending": int(pending_mask.sum()),
            "n_rows_completed": int(completed_mask.sum()),
            "mode": "physics_plus_selected_residual_models",
            "stages": stages,
            "models": self.residual_model_manager.summary,
        }
        prefix = f"dt_cycle_{int(self.config.Psycho):04d}"
        with open(os.path.join(out_dir, prefix + "_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        df.loc[pending_mask].to_csv(os.path.join(out_dir, prefix + "_pending_rows.csv"), index=False)
