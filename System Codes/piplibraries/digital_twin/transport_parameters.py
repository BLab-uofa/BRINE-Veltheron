"""Transport inputs for the four-salt campaign and explicit paper-based estimates.

Bare diameters: Naseri Boroujeni et al. (2024), Table 1; Zn2+: (2023), Table 2.
Complex diameters and limiting mobilities: the 2024 paper, eqs 34 and 35.
Complex estimates extend the paper's recipe to the specified Zn/chloride and
alkali/sulfate pairs. They are modelling assumptions, not measured transport data.
"""
from .phreeqc_features import PhreeqcFeatureBuilder


ION_DIAMETERS_M = {"H+": 0.74e-10, "Na+": 2.04e-10, "K+": 2.76e-10,
                   "Zn+2": 1.50e-10, "Cl-": 3.62e-10, "SO4-2": 4.60e-10}

# PHREEQC names and stoichiometries actually present in this campaign.
COMPLEX_IONS = {
    "ZnCl+": {"Zn+2": 1, "Cl-": 1},
    "ZnCl3-": {"Zn+2": 1, "Cl-": 3},
    "ZnCl4-2": {"Zn+2": 1, "Cl-": 4},
    "KSO4-": {"K+": 1, "SO4-2": 1},
    "NaSO4-": {"Na+": 1, "SO4-2": 1},
    "HSO4-": {"H+": 1, "SO4-2": 1},
}


def estimate_complex_transport(parts, charge, mobilities, diameters):
    """Inverse-cubic mobility and volume-additive diameter, in SI units.

u_complex = |z_complex| / (sum n_i*(|z_i|/u_i)^3)^(1/3).
This is the eq-35 form that reproduces the paper's tabulated conductivities.
"""
    denominator = sum(count * (abs(PhreeqcFeatureBuilder._infer_charge(ion)) / mobilities[ion]) ** 3
                      for ion, count in parts.items())
    mobility = abs(charge) / denominator ** (1.0 / 3.0)
    diameter = sum(count * diameters[ion] ** 3 for ion, count in parts.items()) ** (1.0 / 3.0)
    return mobility, diameter


def extend_transport_parameters(mobilities, diameter_overrides=None):
    mobilities = dict(mobilities)
    diameters = dict(ION_DIAMETERS_M)
    diameters.update(diameter_overrides or {})
    sources = {ion: "mobility_file" for ion in mobilities}
    for complex_name, parts in COMPLEX_IONS.items():
        if not all(ion in mobilities and ion in diameters for ion in parts):
            continue
        charge = PhreeqcFeatureBuilder._infer_charge(complex_name)
        mobility, diameter = estimate_complex_transport(parts, charge, mobilities, diameters)
        if complex_name not in mobilities:
            mobilities[complex_name] = mobility
            sources[complex_name] = "paper_eq35_estimate"
        diameters.setdefault(complex_name, diameter)
    return mobilities, diameters, sources
