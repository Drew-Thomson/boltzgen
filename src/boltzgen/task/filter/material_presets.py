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


def ligand_counter_screen_filtering_overrides() -> dict[str, Any]:
    """Return ranking weights for opt-in ligand (apo) counter-screen metrics.

    Weights in Filter correspond to inverse importance (1 = primary priority,
    higher values de-emphasize the metric).
    """
    return {
        "delta_rmsd_vs_apo": 1,
        "holo_apo_rmsd": 1,
        "delta_protein_iptm_vs_apo": 1,
        "neg_num_ligand_clashes": 1,
        "apo_rmsd": 2,
        "delta_complex_plddt_vs_apo": 2,
        "min_ligand_distance": 2,
        "ligand_iptm": 2,
        "mean_ligand_coordination": 2,
        "mean_ligand_plddt": 2,
        "neg_lattice_rmsd_refolded": 3,
        "complex_plddt": 3,
        "neg_local_rmsd": None,
        "neg_global_rmsd": None,
        "design_ptm": None,
        "neg_min_interaction_pae": None,
        "design_to_target_iptm": None,
        "neg_min_design_to_target_pae": None,
        "h_bonds_per_interface_refolded": None,
        "packing_density_refolded": None,
    }


CYCLIC_LIGAND_SYMMETRY_COLUMNS: tuple[str, ...] = (
    "neg_sym_chain_rmsd",
    "neg_sym_ligand_rmsd",
    "neg_ligand_coord_geom_dev",
    "neg_ligand_coord_bond_dev",
    "ligand_min_coord_number",
)
"""Analysis columns whose presence triggers the cyclic ligand-symmetry preset."""

CYCLIC_LIGAND_SYMMETRY_RMSD_THRESHOLD: float = 15.0
"""Relaxed template-RMSD hard-filter threshold (Å); symmetry/coordination dominate."""


def cyclic_ligand_symmetry_filtering_overrides() -> dict[str, Any]:
    """Return ranking weights for cyclic assemblies with metal ligands.

    Weights are inverse importances (1 = primary). Fitted-axis symmetry and the
    metal coordination sphere dominate; template RMSD metrics are disabled.
    """
    return {
        "neg_sym_chain_rmsd": 1,
        "neg_sym_ligand_rmsd": 1,
        "neg_ligand_coord_geom_dev": 1,
        "neg_ligand_coord_bond_dev": 1,
        "ligand_min_coord_number": 1,
        "neg_ligand_coord_geom_dev_std": 2,
        "neg_ligand_coord_cn_std": 2,
        "neg_num_ligand_clashes": 2,
        "mean_ligand_plddt": 2,
        "complex_plddt": 2,
        "design_ptm": 3,
        "neg_min_interaction_pae": 3,
        "neg_local_rmsd": None,
        "neg_global_rmsd": None,
        "neg_lattice_rmsd_refolded": None,
        "neg_min_design_to_target_pae": None,
        "design_to_target_iptm": None,
    }


def cyclic_ligand_symmetry_filters() -> list[dict[str, Any]]:
    """Hard filters (metal coordination and ring symmetry) for the cyclic ligand preset."""
    return [
        {"feature": "ligand_min_coord_number", "threshold": 3.0, "lower_is_better": False},
        {"feature": "ligand_coord_bond_dev", "threshold": 0.5, "lower_is_better": True},
        {"feature": "sym_ligand_rmsd", "threshold": 1.0, "lower_is_better": True},
        {"feature": "sym_chain_rmsd", "threshold": 2.0, "lower_is_better": True},
        {"feature": "sym_ligand_orbit_ok", "threshold": 1.0, "lower_is_better": False},
    ]


def format_ligand_counter_screen_metrics_override() -> str:
    """Format ligand counter-screen ranking overrides as a Hydra-compatible dictionary."""
    overrides = ligand_counter_screen_filtering_overrides()
    values = ", ".join(
        f"{key}: {'null' if value is None else value}"
        for key, value in overrides.items()
    )
    return f"metrics_override={{{values}}}"


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
