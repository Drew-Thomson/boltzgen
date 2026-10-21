"""Topology-specific material ranking presets for the filtering task."""

from __future__ import annotations

from typing import Any


def material_filtering_overrides(topology: str) -> dict[str, Any] | None:
    """Return metrics_override values for material topologies.

    Filter maximizes every configured metric. Negative-valued analysis columns
    are used for metrics where smaller raw values are better.
    """
    if topology in {"cage_tetrahedral", "cage_octahedral"}:
        return {
            "design_to_target_iptm": None,
            "neg_min_design_to_target_pae": None,
            "neg_cage_sphericity_rmsd": 1,
            "cage_void_radius": 1,
            "neg_lattice_rmsd_refolded": 1,
        }
    if topology in {"linear_tape", "double_tape", "bilayer_sheet"}:
        overrides = {
            "neg_min_interaction_pae": 1,
            "complex_plddt": 1,
            "design_to_target_iptm": None,
            "neg_lattice_rmsd_refolded": 1,
            "neg_global_rmsd": None,
        }
        if topology == "double_tape":
            overrides.update({
                "double_tape_neatness_score": 1,
            })
        if topology == "bilayer_sheet":
            overrides.update({
                "sheet_planarity": 1,
            })
        return overrides
    if topology == "hexagonal_mesh":
        return {
            "sheet_planarity": 1,
            "neg_pore_aspect_ratio": 1,
        }
    if topology == "cylindrical":
        return {
            "complex_plddt": 1,
            "neg_min_interaction_pae": 1,
            "neg_cylinder_radius_rmsd": 2,
            "cylinder_pore_radius": 1,
        }
    return None


def heteromer_filtering_overrides() -> dict[str, Any]:
    """Return ranking weights for opt-in heteromer counter-screen metrics."""
    return {
        "delta_design_iptm_vs_homomer_max": 1,
        "delta_complex_plddt_vs_homomer_max": 1,
        "delta_protein_iptm_vs_homomer_max": 2,
    }


def format_metrics_override(topology: str) -> str | None:
    """Format material ranking overrides as a Hydra-compatible dictionary."""
    overrides = material_filtering_overrides(topology)
    if overrides is None:
        return None
    values = ", ".join(
        f"{key}: {'null' if value is None else value}"
        for key, value in overrides.items()
    )
    return f"metrics_override={{{values}}}"


def format_additional_filters(
    topology: str,
    enable_consecutive_hydrophobics: bool = False,
    is_heteromer: bool = False,
    enable_composition_filters: bool = False,
) -> str | None:
    """Format optional additional hard filters for material topologies.

    Consecutive hydrophobic filtering (max_consecutive_ilv_sheet <= 2) is
    inactivated by default but can be enabled via enable_consecutive_hydrophobics.
    Heteromer assemblies enforce two_chain_alternation_ratio >= 0.8 to eliminate
    homomer collapse.
    Composition filtering enforces limits on outward aromatics, overall hydrophobicity,
    and ensures minimum Threonine presence for double_tape assemblies.
    """
    valid_topologies = {"linear_tape", "double_tape", "bilayer_sheet"}
    if topology not in valid_topologies:
        return None

    filters: list[str] = []
    if enable_consecutive_hydrophobics:
        filters.append(
            "{feature: max_consecutive_ilv_sheet, lower_is_better: True, threshold: 2}"
        )
    if is_heteromer:
        filters.append(
            "{feature: two_chain_alternation_ratio, "
            "lower_is_better: False, threshold: 0.8}"
        )
        if topology == "double_tape":
            filters.extend([
                "{feature: strand_polarity_target_match, "
                "lower_is_better: False, threshold: 0.9}",
                "{feature: intra_sheet_alternation_ratio, "
                "lower_is_better: False, threshold: 0.85}",
                "{feature: bilayer_dist_std, lower_is_better: True, threshold: 1.5}",
            ])
    if enable_composition_filters and topology == "double_tape":
        filters.extend([
            "{feature: outward_aromatic_fraction, "
            "lower_is_better: True, threshold: 0.25}",
            "{feature: design_hydrophobicity, lower_is_better: True, threshold: 28.0}",
            "{feature: THR_fraction, lower_is_better: False, threshold: 0.11}",
        ])

    if not filters:
        return None
    return f"additional_filters=[{', '.join(filters)}]"
