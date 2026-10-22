from __future__ import annotations

import re

TARGET_CHEMISTRY = {"NaCl", "KCl", "ZnCl2", "ZnSO4"}
CHEMICAL_ALIASES = {
    "NaCl": ["nacl", "sodium chloride"],
    "KCl": ["kcl", "potassium chloride"],
    "ZnCl2": ["zncl2", "zinc chloride"],
    "ZnSO4": ["znso4", "zinc sulfate", "zinc sulphate"],
}


def convert_conductivity_to_S_per_m(value: float, unit: str) -> float:
    u = unit.lower().replace(" ", "").replace("·", "")
    if u in {"s/m", "sm-1", "sm^-1"}:
        return value
    if u in {"s/cm", "scm-1", "scm^-1"}:
        return value * 100.0
    if u in {"ms/cm", "mscm-1", "mscm^-1"}:
        return value * 0.1
    raise ValueError(f"Unsupported conductivity unit: {unit}")


def chemicals_found(text: str) -> set[str]:
    low = (text or "").lower()
    found = set()
    for chemical, aliases in CHEMICAL_ALIASES.items():
        if any(alias.lower() in low for alias in aliases):
            found.add(chemical)
    return found


def _primary_study_text(title: str, abstract: str) -> str:
    return f"{title or ''} {(abstract or '')[:700]}"


def classify_similarity(title: str, abstract: str) -> dict:
    primary = chemicals_found(_primary_study_text(title, abstract))
    all_components = chemicals_found(f"{title or ''} {abstract or ''}")
    full = f"{title or ''} {abstract or ''}".lower()
    overlap = TARGET_CHEMISTRY.intersection(primary)

    if TARGET_CHEMISTRY.issubset(primary):
        level = "direct_match"
    elif "ZnCl2" in primary and "ZnSO4" in primary:
        level = "very_close_analogue"
    elif "ZnCl2" in primary or "ZnSO4" in primary:
        level = "close_analogue"
    elif "zinc" in full or "zn2+" in full or "zn²+" in full:
        level = "broad_zinc_analogue"
    else:
        level = "low_similarity"

    return {
        "similarity_level": level,
        "matched_components": sorted(overlap),
        "primary_components_found": sorted(primary),
        "all_components_mentioned": sorted(all_components),
    }


def extract_conductivity_values(text: str) -> list[dict]:
    # Accept common slash and inverse-centimeter renderings from abstracts.
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*mS\s*(?:/\s*cm|cm\s*(?:\^?\s*-?1|[⁻−]1))", "mS/cm"),
        (r"(\d+(?:\.\d+)?)\s*S\s*(?:/\s*cm|cm\s*(?:\^?\s*-?1|[⁻−]1))", "S/cm"),
        (r"(\d+(?:\.\d+)?)\s*S\s*(?:/\s*m|m\s*(?:\^?\s*-?1|[⁻−]1))", "S/m"),
    ]
    results = []
    for pattern, unit in patterns:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            value = float(match.group(1))
            results.append({
                "reported_value": value,
                "reported_unit": unit,
                "normalized_S_per_m": convert_conductivity_to_S_per_m(value, unit),
                "source_text": match.group(0),
            })
    return results


def build_evidence_packet(bond_result: dict) -> list[dict]:
    evidence = []
    for index, paper in enumerate(bond_result.get("papers", []) or [], start=1):
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        similarity = classify_similarity(title, abstract)
        evidence.append({
            "paper_id": f"P{index}",
            "title": title,
            "doi": paper.get("doi"),
            "url": paper.get("url"),
            "source": paper.get("source"),
            "semantic_score": paper.get("semantic_score"),
            "citation_count": paper.get("citation_count"),
            **similarity,
            "conductivity_values": extract_conductivity_values(abstract),
            "abstract": abstract,
        })
    return evidence


def compare_conductivity_values(campaign_max_S_per_m: float, evidence_packet: list[dict]) -> list[dict]:
    comparisons = []
    for paper in evidence_packet:
        for value in paper.get("conductivity_values", []):
            lit = float(value["normalized_S_per_m"])
            diff = lit - campaign_max_S_per_m
            pct = (diff / campaign_max_S_per_m * 100.0) if campaign_max_S_per_m else None
            comparisons.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "doi": paper.get("doi"),
                "similarity_level": paper["similarity_level"],
                "campaign_max_S_per_m": campaign_max_S_per_m,
                "literature_value_original": value["reported_value"],
                "literature_unit_original": value["reported_unit"],
                "literature_value_S_per_m": lit,
                "difference_S_per_m": diff,
                "percent_difference_vs_campaign": pct,
            })
    return comparisons


def summarize_evidence(evidence_packet: list[dict]) -> dict:
    counts = {k: 0 for k in ["direct_match", "very_close_analogue", "close_analogue", "broad_zinc_analogue", "low_similarity"]}
    numeric = 0
    for paper in evidence_packet:
        level = paper.get("similarity_level", "low_similarity")
        if level in counts:
            counts[level] += 1
        if paper.get("conductivity_values"):
            numeric += 1
    return {
        "number_of_papers": len(evidence_packet),
        "similarity_counts": counts,
        "papers_with_explicit_numeric_conductivity": numeric,
        "direct_numeric_benchmark_available": any(
            p.get("similarity_level") == "direct_match" and bool(p.get("conductivity_values"))
            for p in evidence_packet
        ),
    }
