from dataclasses import dataclass
from typing import Sequence, Any
import math


@dataclass
class DTConfig:
    input_file: str
    output_folder: str

    num_inputs: int
    num_objectives: int
    num_proposals: int
    
    # Only these measured columns belong to the DT. Other BO outputs are ignored.
    objective_map: dict[str, str]
    
    Psycho: int

    cell_volume_uL: float
    stock_molarities: Sequence[float]
    salt_formulas: Sequence[str]
    
    phreeqc_exe_path: str
    phreeqc_database_path: str
    mobility_file: str = "mobility.csv"
    
    # Campaign temperature in degrees Celsius.
    temp_C: float = 25.0
    
    # Viscosity
    mobility_scale_factor: float = 1e8
    laliberte_enabled: bool = True
    
    # XGBoost residual ensemble
    xgb_ensemble_size: int = 20
    xgb_min_training_rows: int = 15
    xgb_seed: int = 42
    xgb_n_estimators: int = 250
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.90
    xgb_min_child_weight: float = 2.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_n_jobs: int = 1

    # PHREEQC
    phreeqc_run_dir: str = "phreeqc_species_run"
    phreeqc_units: str = "mol/l"
    phreeqc_ph_guess: float = 7.0
    phreeqc_charge_balance_pH: bool = True
    # Historical option names retained; these parser thresholds are mol/kgw.
    phreeqc_parse_min_species_M: float = 1e-12
    phreeqc_species_column_threshold_M: float = 0.05
    
    # Nernst baseline
    water_activity_default: float = 1.0
    nernst_p_H2: float = 1.0
    nernst_p_O2: float = 1.0
    
    # Conductivity theory baseline
    conductivity_enabled: bool = True
    conductivity_min_species_M: float = 1e-12
    conductivity_epsilon_r: float = 78.43
    conductivity_default_viscosity_Pa_s: float = 0.890e-3  # Legacy option; no fallback uses it.
    conductivity_default_sigma_m: float = 4.0e-10  # Legacy option; replaced by ion-specific diameters.
    # One campaign-wide input choice. The published model uses solvent viscosity.
    conductivity_viscosity_source: str = "solvent"
    conductivity_ion_diameters_m: dict[str, float] | None = None

    # Raw measurement -> S/m. Use 1.0 for S/m; 0.1 for mS/cm.
    conductivity_scale_to_S_m: float = 1.0
    # Historical BO stores -E_HER so both BO objectives can be maximised.
    her_bo_negated: bool = True
    

    # Passed for bookkeeping / future use
    Salt_Tank: Any = None
    Water_Tank: Any = None
    EC_Tests: Any = None
    Real_Inputs: int | None = None

    dt_subfolder: str = "DigitalTwin"

    @property
    def objective_columns(self):
        return list(self.objective_map)

    @property
    def objective_targets(self):
        aliases = {"conductivity": "sigma", "sigma": "sigma", "her": "her", "oer": "oer"}
        return tuple(aliases[str(name).strip().lower()] for name in self.objective_map.values())

    def objective_column(self, target):
        return self.objective_columns[self.objective_targets.index(target)]

    def validate(self):
        self.num_inputs = int(self.num_inputs)
        self.num_objectives = int(self.num_objectives)
        self.num_proposals = int(self.num_proposals)
        self.Psycho = int(self.Psycho)
        self.cell_volume_uL = float(self.cell_volume_uL)
        
        self.mobility_scale_factor = float(self.mobility_scale_factor)
        self.temp_C = float(self.temp_C)
        
        self.water_activity_default = float(self.water_activity_default)
        self.nernst_p_H2 = float(self.nernst_p_H2)
        self.nernst_p_O2 = float(self.nernst_p_O2)

        self.conductivity_min_species_M = float(self.conductivity_min_species_M)
        self.conductivity_epsilon_r = float(self.conductivity_epsilon_r)
        self.conductivity_default_viscosity_Pa_s = float(self.conductivity_default_viscosity_Pa_s)
        self.conductivity_default_sigma_m = float(self.conductivity_default_sigma_m)
        
        self.xgb_ensemble_size = int(self.xgb_ensemble_size)
        self.xgb_min_training_rows = int(self.xgb_min_training_rows)

        if not self.objective_map:
            raise ValueError('Specify objective_map, for example {"y1": "conductivity"}.')
        allowed_targets = {"conductivity", "sigma", "her", "oer"}
        invalid = [name for name in self.objective_map.values() if str(name).strip().lower() not in allowed_targets]
        if invalid:
            raise ValueError(f"Unknown DT objectives: {invalid}")
        if len(set(self.objective_targets)) != len(self.objective_targets):
            raise ValueError("Each electrochemical objective must be mapped once.")
        # num_objectives remains the BO's total output count. DT may select a subset.
        if len(self.objective_map) > self.num_objectives:
            raise ValueError("DT cannot select more objectives than num_objectives.")
        self.conductivity_scale_to_S_m = float(self.conductivity_scale_to_S_m)
        if not math.isfinite(self.conductivity_scale_to_S_m) or self.conductivity_scale_to_S_m <= 0:
            raise ValueError("conductivity_scale_to_S_m must be positive and finite.")
        if "sigma" in self.objective_targets and not self.conductivity_enabled:
            raise ValueError("A conductivity objective requires conductivity_enabled=True.")
        if "sigma" in self.objective_targets and not self.laliberte_enabled:
            raise ValueError("The corrected conductivity model requires laliberte_enabled=True.")
        if self.conductivity_viscosity_source not in {"solvent", "mixture"}:
            raise ValueError("conductivity_viscosity_source must be 'solvent' or 'mixture'.")
        
        if self.mobility_scale_factor <= 0:
            raise ValueError("mobility_scale_factor must be positive.")

        if self.num_inputs <= 0:
            raise ValueError("num_inputs must be positive.")

        if self.num_objectives <= 0:
            raise ValueError("num_objectives must be positive.")

        if self.cell_volume_uL <= 0:
            raise ValueError("cell_volume_uL must be positive.")

        if len(self.stock_molarities) != self.num_inputs:
            raise ValueError(
                f"stock_molarities length must match num_inputs. "
                f"Got {len(self.stock_molarities)} values for {self.num_inputs} inputs."
            )

        if len(self.salt_formulas) != self.num_inputs:
            raise ValueError(
                f"salt_formulas length must match num_inputs. "
                f"Got {len(self.salt_formulas)} formulas for {self.num_inputs} inputs."
            )

        self.stock_molarities = [float(x) for x in self.stock_molarities]
        self.salt_formulas = [str(x).strip() for x in self.salt_formulas]

        if any(x < 0 for x in self.stock_molarities):
            raise ValueError("stock_molarities cannot contain negative values.")

        if any(x == "" for x in self.salt_formulas):
            raise ValueError("salt_formulas cannot contain empty strings.")

        if self.phreeqc_species_column_threshold_M < 0:
            raise ValueError("phreeqc_species_column_threshold_M cannot be negative.")
        
        
        if self.water_activity_default <= 0:
            raise ValueError("water_activity_default must be positive.")

        if self.nernst_p_H2 <= 0:
            raise ValueError("nernst_p_H2 must be positive.")

        if self.nernst_p_O2 <= 0:
            raise ValueError("nernst_p_O2 must be positive.")

        if self.conductivity_epsilon_r <= 0:
            raise ValueError("conductivity_epsilon_r must be positive.")

        if self.conductivity_default_viscosity_Pa_s <= 0:
            raise ValueError("conductivity_default_viscosity_Pa_s must be positive.")

        if self.conductivity_default_sigma_m <= 0:
            raise ValueError("conductivity_default_sigma_m must be positive.")
        
        
        if self.xgb_ensemble_size < 2:
            raise ValueError("xgb_ensemble_size must be at least 2.")

        if self.xgb_min_training_rows < 2:
            raise ValueError("xgb_min_training_rows must be at least 2.")

        if not 0 < self.xgb_subsample <= 1:
            raise ValueError("xgb_subsample must be in (0, 1].")

        if not 0 < self.xgb_colsample_bytree <= 1:
            raise ValueError("xgb_colsample_bytree must be in (0, 1].")
