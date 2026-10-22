import json
import math
from dataclasses import dataclass

import pandas as pd
from .mobility_features import MobilityFeatureBuilder
from .transport_parameters import extend_transport_parameters
from thermo.electrochem import Laliberte_viscosity_w


@dataclass(frozen=True)
class PhysicalConstants:
    e: float = 1.602176634e-19
    k_B: float = 1.380649e-23
    N_A: float = 6.02214076e23
    F: float = 96485.33212
    R: float = 8.314462618
    eps0: float = 8.8541878128e-12


@dataclass(frozen=True)
class ConductiveSpecies:
    name: str
    charge: int
    lambda0_S_m2_mol: float
    sigma_m: float


class ConductivityTheoryBuilder:
    """
    Computes a corrected theoretical ionic conductivity baseline in S/m.

    The model uses charged PHREEQC species with available:
        charge z
        limiting ionic mobility, converted to lambda0
        ion-size descriptor sigma_m

    Output:
        dt_sigma_theory_S_m

    Notes:
        - Neutral species do not directly contribute to conductivity.
        - Unknown charged complexes are skipped unless added to the species database later.
        - This is a physics baseline, not the final DT prediction.
    """

    SIMPLE_SPECIES_CHARGE = MobilityFeatureBuilder.CHARGE
    SPECIES_ALIASES = MobilityFeatureBuilder.SPECIES_ALIASES

    def __init__(self, config):
        self.config = config
        self.constants = PhysicalConstants()
        self.mobility = self._load_mobility_file(config.mobility_file)
        self.mobility, self.diameters, self.mobility_sources = extend_transport_parameters(
            self.mobility, config.conductivity_ion_diameters_m)

    def build(self, df):
        df = df.copy()

        numeric_outputs = ["dt_sigma_theory_S_m", "dt_sigma_theory_ideal_S_m", "dt_conductivity_species_count",
                           "dt_conductivity_debye_length_m", "dt_conductivity_ionic_strength_M",
                           "dt_conductivity_charge_weight_coverage", "dt_conductivity_estimated_charge_weight_fraction",
                           "dt_conductivity_viscosity_Pa_s", "dt_sigma_theory_raw_S_m",
                           "dt_conductivity_min_ep_factor", "dt_conductivity_min_rel_factor"]
        for col in numeric_outputs:
            df[col] = math.nan
        df["dt_conductivity_theory_ok"] = False
        df["dt_conductivity_theory_error"] = ""
        df["dt_conductivity_unmapped_species_json"] = "{}"
        df["dt_conductivity_estimated_species_json"] = "[]"
        df["dt_conductivity_viscosity_source"] = self.config.conductivity_viscosity_source
        df["dt_conductivity_corrections_physical"] = False

        if not bool(self.config.conductivity_enabled):
            df["dt_conductivity_theory_ok"] = False
            df["dt_conductivity_theory_error"] = "Conductivity theory disabled."
            return df

        temperature_K = float(self.config.temp_C) + 273.15

        for idx, row in df.iterrows():
            try:
                if not bool(row.get("dt_phreeqc_ok", False)):
                    raise ValueError("PHREEQC speciation unavailable for this row.")
                species = self._load_species_json(row)
                factor = float(row.get("dt_phreeqc_molality_to_molarity_factor", math.nan))
                if not math.isfinite(factor) or factor <= 0:
                    raise ValueError("PHREEQC molality-to-molarity conversion factor unavailable.")
                species_df = self._build_conductive_species_table(species, temperature_K, factor)

                mixture_viscosity = float(row.get("dt_viscosity_Pa_s", math.nan))
                if not bool(row.get("dt_viscosity_ok", False)) or not math.isfinite(mixture_viscosity) or mixture_viscosity <= 0:
                    raise ValueError("Valid mixture viscosity required for the four-salt DT descriptors.")
                # Eq 22 defines eta as solvent viscosity. Laliberte mixture viscosity
                # remains an ML descriptor; using it here is an explicit adaptation.
                viscosity = (Laliberte_viscosity_w(temperature_K)
                             if self.config.conductivity_viscosity_source == "solvent" else mixture_viscosity)
                df.at[idx, "dt_conductivity_viscosity_Pa_s"] = viscosity

                all_charge_weight = 0.0
                unmapped = {}
                for name, amount in species.items():
                    canonical = MobilityFeatureBuilder.canonical_species(name)
                    z = MobilityFeatureBuilder.species_charge(canonical)
                    all_charge_weight += float(amount) * factor * z ** 2
                    if z and canonical not in set(species_df["species"]):
                        unmapped[name] = float(amount)
                covered = float((species_df["c_mol_L"] * species_df["z"] ** 2).sum())
                df.at[idx, "dt_conductivity_charge_weight_coverage"] = covered / all_charge_weight
                df.at[idx, "dt_conductivity_unmapped_species_json"] = json.dumps(unmapped, sort_keys=True)
                estimated = species_df["mobility_source"] == "paper_eq35_estimate"
                estimated_charge = float((species_df.loc[estimated, "c_mol_L"] * species_df.loc[estimated, "z"] ** 2).sum())
                df.at[idx, "dt_conductivity_estimated_charge_weight_fraction"] = estimated_charge / all_charge_weight
                df.at[idx, "dt_conductivity_estimated_species_json"] = json.dumps(species_df.loc[estimated, "species"].tolist())

                sigma_S_m, aux = self._calculate_corrected_conductivity(
                    species_df=species_df,
                    temperature_K=temperature_K,
                    epsilon_r=float(self.config.conductivity_epsilon_r),
                    viscosity_Pa_s=float(viscosity),
                )

                # Preserve diagnostics even if the high-concentration correction
                # is rejected. Only the usable baseline enters residual training.
                df.at[idx, "dt_sigma_theory_raw_S_m"] = sigma_S_m
                df.at[idx, "dt_sigma_theory_ideal_S_m"] = aux["sigma_ideal_S_m"]
                df.at[idx, "dt_conductivity_species_count"] = len(species_df)
                df.at[idx, "dt_conductivity_debye_length_m"] = aux["debye_length_m"]
                df.at[idx, "dt_conductivity_ionic_strength_M"] = aux["ionic_strength_M"]
                df.at[idx, "dt_conductivity_min_ep_factor"] = aux["min_ep_factor"]
                df.at[idx, "dt_conductivity_min_rel_factor"] = aux["min_rel_factor"]
                df.at[idx, "dt_conductivity_corrections_physical"] = aux["corrections_physical"]
                if not aux["corrections_physical"]:
                    raise ValueError(f"Non-physical DHO correction: raw sigma={sigma_S_m:.6g} S/m, "
                                     f"min(1+ep)={aux['min_ep_factor']:.6g}, "
                                     f"min(1+rel)={aux['min_rel_factor']:.6g}.")
                df.at[idx, "dt_sigma_theory_S_m"] = sigma_S_m
                df.at[idx, "dt_conductivity_theory_ok"] = True
                df.at[idx, "dt_conductivity_theory_error"] = ""

            except Exception as e:
                df.at[idx, "dt_conductivity_theory_ok"] = False
                df.at[idx, "dt_conductivity_theory_error"] = str(e)

                df.at[idx, "dt_sigma_theory_S_m"] = math.nan

        return df

    def _load_mobility_file(self, mobility_file):
        df = pd.read_csv(mobility_file)

        if df.empty:
            raise ValueError("mobility_file is empty.")

        row = df.iloc[0].to_dict()
        out = {}

        for key, value in row.items():
            if pd.isna(value):
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Mobility must be a positive SI magnitude for {key}.")
            out[str(key).strip()] = value

        return out

    @staticmethod
    def _load_species_json(row):
        raw = row.get("dt_species_json", "{}")

        if raw is None or pd.isna(raw) or str(raw).strip() == "":
            return {}

        return json.loads(raw)

    def _build_conductive_species_table(self, species, temperature_K, molality_to_molarity_factor):
        rows = []

        for name, amount in species.items():
            canonical = MobilityFeatureBuilder.canonical_species(name)
            conc_mol_kgw = float(amount)
            conc_M = conc_mol_kgw * molality_to_molarity_factor

            if conc_M <= float(self.config.conductivity_min_species_M):
                continue

            z = MobilityFeatureBuilder.species_charge(canonical)
            mobility = MobilityFeatureBuilder.mobility_for(self, canonical)
            if not z or mobility is None or canonical not in self.diameters:
                continue
            mobility_m2_V_s = float(mobility)

            # The D0/omega equations below use equivalent, not molar, conductivity:
            # lambda_eq = F*u, D0 = R*T*u/(F*|z|), omega = u/(e*|z|).
            lambda0 = self.constants.F * mobility_m2_V_s
            sigma_m = float(self.diameters[canonical])

            c_mol_m3 = 1000.0 * conc_M
            rho_m3 = self.constants.N_A * c_mol_m3
            D0 = self._lambda0_to_D0(lambda0, z, temperature_K)
            omega = self._lambda0_to_omega(lambda0, z)

            rows.append(
                {
                    "species": canonical,
                    "z": z,
                    "sigma_m": sigma_m,
                    "lambda0_S_m2_mol": lambda0,
                    "c_mol_kgw": conc_mol_kgw,
                    "c_mol_L": conc_M,
                    "c_mol_m3": c_mol_m3,
                    "rho_m3": rho_m3,
                    "D0_m2_s": D0,
                    "mobility_source": self.mobility_sources[canonical],
                    "omega": omega,
                }
            )

        species_df = pd.DataFrame(rows)

        if species_df.empty:
            raise ValueError("No charged species with mobility data were available for conductivity theory.")

        return species_df.sort_values("omega").reset_index(drop=True)

    @staticmethod
    def _find_species_concentration(species, aliases):
        for alias in aliases:
            if alias in species:
                return species[alias]
        return None

    def _lambda0_to_D0(self, lambda0, z, temperature_K):
        if z == 0:
            raise ValueError("Neutral species cannot be converted to D0.")

        return (
            self.constants.R
            * temperature_K
            * lambda0
            / (self.constants.F ** 2 * abs(z))
        )

    def _lambda0_to_omega(self, lambda0, z):
        if z == 0:
            raise ValueError("Neutral species cannot be converted to omega.")

        return (
            self.constants.N_A
            * lambda0
            / (self.constants.F ** 2 * abs(z))
        )

    @staticmethod
    def _ionic_strength(species_df):
        return 0.5 * float((species_df["c_mol_L"] * species_df["z"] ** 2).sum())

    def _inverse_debye_length(self, species_df, temperature_K, epsilon_r):
        sum_rho_z2 = float((species_df["rho_m3"] * species_df["z"] ** 2).sum())

        return math.sqrt(
            self.constants.e ** 2
            / (
                self.constants.eps0
                * epsilon_r
                * self.constants.k_B
                * temperature_K
            )
            * sum_rho_z2
        )

    def _transport_weights(self, species_df):
        df = species_df.copy()

        total = float((df["rho_m3"] * df["z"] ** 2).sum())

        if total <= 0:
            raise ValueError("Charge-density sum is not positive.")

        df["theta_i"] = df["rho_m3"] * df["z"] ** 2 / total

        omega_bar = float((df["theta_i"] * df["omega"]).sum())

        if omega_bar <= 0:
            raise ValueError("omega_bar is not positive.")

        df["t_i"] = df["theta_i"] * df["omega"] / omega_bar

        return df, omega_bar

    @staticmethod
    def _bisect(func, left, right, max_iter=300, tol=1e-14):
        f_left = func(left)
        f_right = func(right)

        if not math.isfinite(f_left) or not math.isfinite(f_right):
            raise ValueError("Bisection endpoints are non-finite.")

        if f_left == 0:
            return left

        if f_right == 0:
            return right

        if f_left * f_right > 0:
            raise ValueError("Bisection root is not bracketed.")

        a = left
        b = right
        fa = f_left

        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            fm = func(mid)

            if abs(fm) < tol or abs(b - a) < tol:
                return mid

            if fa * fm <= 0:
                b = mid
            else:
                a = mid
                fa = fm

        return 0.5 * (a + b)

    def _relaxation_roots(self, species_df):
        omega = species_df["omega"].to_numpy(dtype=float)
        t = species_df["t_i"].to_numpy(dtype=float)

        n_species = len(omega)

        if n_species == 1:
            return [0.0]

        # Solve in dimensionless variables: unscaled function values are ~1e-24
        # and made the old absolute-tolerance bisection stop at its first midpoint.
        scale = float(max(omega))
        omega = omega / scale
        alphas = [0.0]

        def root_function(alpha):
            total = 0.0
            for tj, wj in zip(t, omega):
                total += tj / (wj ** 2 - alpha ** 2)
            return total

        tiny = 1e-10

        for p in range(1, n_species):
            left = omega[p - 1] * (1.0 + tiny)
            right = omega[p] * (1.0 - tiny)

            if right <= left:
                alphas.append(0.5 * (omega[p - 1] + omega[p]))
                continue

            alphas.append(self._bisect(root_function, left, right))

        return [alpha * scale for alpha in alphas]

    def _relaxation_modes(self, species_df, omega_bar):
        df = species_df.copy()

        omega = df["omega"].to_numpy(dtype=float)
        t = df["t_i"].to_numpy(dtype=float)
        alpha = self._relaxation_roots(df)

        q_values = []
        n_values = []
        zeta_matrix = []

        for p, alpha_p in enumerate(alpha):
            denom = [wj ** 2 - alpha_p ** 2 for wj in omega]

            inv_n2 = 0.0
            for tj, wj, denom_j in zip(t, omega, denom):
                inv_n2 += tj * wj ** 2 / denom_j ** 2

            if inv_n2 <= 0:
                raise ValueError("Relaxation normalization is not positive.")

            n_p = 1.0 / math.sqrt(inv_n2)

            zeta_p = []
            for wj, denom_j in zip(omega, denom):
                zeta_p.append(n_p * wj / denom_j)

            q_p = 0.0
            for tj, wj in zip(t, omega):
                q_p += omega_bar * tj / (wj + alpha_p)

            if q_p <= 0:
                raise ValueError("Relaxation q value is not positive.")

            q_values.append(q_p)
            n_values.append(n_p)
            zeta_matrix.append(zeta_p)

        return {
            "alpha": alpha,
            "q": q_values,
            "N": n_values,
            "zeta": zeta_matrix,
        }

    def _electrophoretic_correction(self, species_df, kappa_1_m, temperature_K, epsilon_r, viscosity_Pa_s):
        corrections = []

        for _, row_i in species_df.iterrows():
            z_i = float(row_i["z"])
            sigma_i = float(row_i["sigma_m"])
            lambda0_i = float(row_i["lambda0_S_m2_mol"])

            pair_sum = 0.0

            for _, row_j in species_df.iterrows():
                z_j = float(row_j["z"])
                rho_j = float(row_j["rho_m3"])
                sigma_j = float(row_j["sigma_m"])
                sigma_ij = 0.5 * (sigma_i + sigma_j)

                bracket = (
                    math.exp(kappa_1_m * (sigma_i - sigma_ij))
                    / (kappa_1_m * (1.0 + kappa_1_m * sigma_i))
                    +
                    math.exp(kappa_1_m * (sigma_j - sigma_ij))
                    / (kappa_1_m * (1.0 + kappa_1_m * sigma_j))
                )

                pair_sum += rho_j * self.constants.e ** 2 * z_j ** 2 * bracket

            prefactor = (
                -self.constants.F ** 2
                * abs(z_i)
                / (
                    12.0
                    * math.pi
                    * self.constants.eps0
                    * epsilon_r
                    * viscosity_Pa_s
                    * self.constants.k_B
                    * temperature_K
                    * self.constants.N_A
                    * lambda0_i
                )
            )

            corrections.append(prefactor * pair_sum)

        return corrections

    @staticmethod
    def _charge_mobility_ratio(q_i, omega_i, q_j, omega_j):
        return (q_i * omega_i - q_j * omega_j) / (omega_i + omega_j)

    def _relaxation_correction(self, species_df, kappa_1_m, temperature_K, epsilon_r):
        df, omega_bar = self._transport_weights(species_df)
        modes = self._relaxation_modes(df, omega_bar)

        q_values = modes["q"]
        zeta_matrix = modes["zeta"]

        n_species = len(df)
        corrections = []

        for m in range(n_species):
            row_m = df.iloc[m]
            z_m = float(row_m["z"])
            q_m_charge = self.constants.e * z_m

            outer_sum = 0.0

            for p in range(n_species):
                q_p = float(q_values[p])
                sqrt_q_p = math.sqrt(q_p)
                zeta_m_p = float(zeta_matrix[p][m])

                inner_sum = 0.0

                for j in range(n_species):
                    row_j = df.iloc[j]
                    t_j = float(row_j["t_i"])
                    zeta_j_p = float(zeta_matrix[p][j])
                    sigma_j = float(row_j["sigma_m"])
                    z_j = float(row_j["z"])
                    q_j_charge = self.constants.e * z_j
                    omega_j = float(row_j["omega"])

                    for i in range(n_species):
                        row_i = df.iloc[i]
                        theta_i = float(row_i["theta_i"])
                        sigma_i = float(row_i["sigma_m"])
                        z_i = float(row_i["z"])
                        q_i_charge = self.constants.e * z_i
                        omega_i = float(row_i["omega"])

                        sigma_ji = 0.5 * (sigma_j + sigma_i)

                        ratio = self._charge_mobility_ratio(
                            q_i=q_i_charge,
                            omega_i=omega_i,
                            q_j=q_j_charge,
                            omega_j=omega_j,
                        )

                        x = kappa_1_m * sqrt_q_p * sigma_ji

                        hydrodynamic_factor = math.sinh(x) / sigma_ji

                        bracket = (
                            math.exp(kappa_1_m * sigma_j) / (1.0 + kappa_1_m * sigma_j)
                            +
                            math.exp(kappa_1_m * sigma_i) / (1.0 + kappa_1_m * sigma_i)
                        )

                        decay = math.exp(
                            -kappa_1_m * (1.0 + sqrt_q_p) * sigma_ji
                        )

                        inner_sum += (
                            t_j
                            * zeta_j_p
                            * theta_i
                            * ratio
                            * hydrodynamic_factor
                            * bracket
                            * decay
                        )

                outer_sum += zeta_m_p / (sqrt_q_p + q_p) * inner_sum

            prefactor = (
                q_m_charge
                / (
                    24.0
                    * math.pi
                    * self.constants.eps0
                    * epsilon_r
                    * self.constants.k_B
                    * temperature_K
                )
            )

            corrections.append(prefactor * outer_sum)

        return corrections

    def _calculate_corrected_conductivity(
        self,
        species_df,
        temperature_K,
        epsilon_r,
        viscosity_Pa_s,
    ):
        df = species_df.copy()

        kappa_1_m = self._inverse_debye_length(
            species_df=df,
            temperature_K=temperature_K,
            epsilon_r=epsilon_r,
        )

        df, _omega_bar = self._transport_weights(df)

        electrophoretic = self._electrophoretic_correction(
            species_df=df,
            kappa_1_m=kappa_1_m,
            temperature_K=temperature_K,
            epsilon_r=epsilon_r,
            viscosity_Pa_s=viscosity_Pa_s,
        )

        relaxation = self._relaxation_correction(
            species_df=df,
            kappa_1_m=kappa_1_m,
            temperature_K=temperature_K,
            epsilon_r=epsilon_r,
        )

        df["electrophoretic_factor"] = electrophoretic
        df["relaxation_factor"] = relaxation

        df["correction_factor"] = (
            (1.0 + df["electrophoretic_factor"])
            * (1.0 + df["relaxation_factor"])
        )

        df["ideal_contribution_S_m"] = (
            self.constants.e ** 2
            / (self.constants.k_B * temperature_K)
            * df["rho_m3"]
            * df["D0_m2_s"]
            * df["z"] ** 2
        )

        df["corrected_contribution_S_m"] = (
            df["ideal_contribution_S_m"] * df["correction_factor"]
        )

        sigma_ideal = float(df["ideal_contribution_S_m"].sum())
        sigma_corrected = float(df["corrected_contribution_S_m"].sum())
        min_ep_factor = float((1.0 + df["electrophoretic_factor"]).min())
        min_rel_factor = float((1.0 + df["relaxation_factor"]).min())
        physical = (math.isfinite(sigma_corrected) and sigma_corrected >= 0
                    and min_ep_factor >= 0 and min_rel_factor >= 0)

        aux = {
            "sigma_ideal_S_m": sigma_ideal,
            "debye_length_m": 1.0 / kappa_1_m,
            "ionic_strength_M": self._ionic_strength(df),
            "min_ep_factor": min_ep_factor,
            "min_rel_factor": min_rel_factor,
            "corrections_physical": physical,
        }

        return sigma_corrected, aux
