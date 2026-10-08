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
            "ligand_burial_fraction": 1,
            "neg_min_design_to_target_pae": 1,
            "neg_outward_ilv_fraction": 1,
            "neg_design_largest_hydrophobic_patch_refolded": 2,
            "neg_max_consecutive_ilv_sheet": 2,
            "complex_plddt": 1,
            "design_to_target_iptm": None,
            "neg_lattice_rmsd_refolded": None,
        }
        if topology == "bilayer_sheet":
            overrides.update({
                "sheet_planarity": 1,
                "h_bonds_per_interface_refolded": 1,
            })
        return overrides
    if topology == "hexagonal_mesh":
        return {
            "sheet_planarity": 1,
            "neg_pore_aspect_ratio": 1,
            "h_bonds_per_interface_refolded": 1,
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

def format_additional_filters(topology: str) -> str | None:
    if topology in {"linear_tape", "double_tape", "bilayer_sheet"}:
        return "additional_filters=[{feature: max_consecutive_ilv_sheet, lower_is_better: True, threshold: 2}]"
    return None
