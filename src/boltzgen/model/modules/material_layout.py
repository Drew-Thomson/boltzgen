"""Expand material-builder templates into BoltzGen entities and layout metadata."""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any


SIGMA_ORIENTATION = 0.2  # radians, approx 11 degrees tolerance for side-chain orientation

def classify_sidechains(coms, rots, residues):
    """
    Classify each residue side-chain as "inward" or "outward" relative to the bilayer normal of a double-tape lattice.

    Parameters
    ----------
    coms : list of centre-of-mass coordinates for each chain (from ideal lattice generation)
    rots : list of rotation matrices for each chain (same order as ``coms``)
    residues : list of residue records containing ``chain_index`` and atomic coordinates
               (CA, CB or N for Gly). The order must match the sequence used in the
               decoder output.

    Returns
    -------
    list[str]
        A list with the same length as ``residues`` containing either "inward"
        or "outward".  The classification is based on the angle between the
        side-chain vector (CB-CA for non-Gly, N-CA for Gly) and the local bilayer
        normal.  An angle smaller than ``SIGMA_ORIENTATION`` (in radians) is
        considered "inward"; otherwise "outward".
    """
    import numpy as np
    classifications = []
    # compute bilayer normal for each chain from its rotation matrix (Z-axis)
    chain_normals = [rot[:, 2] for rot in rots]
    for res in residues:
        chain_idx = res["chain_index"]
        # side-chain vector
        if res.get("is_glycine"):
            vec = np.array(res["N"]) - np.array(res["CA"])
        else:
            vec = np.array(res["CB"]) - np.array(res["CA"])
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            angle = np.pi
        else:
            vec = vec / vec_norm
            normal = chain_normals[chain_idx]
            dot = np.clip(np.dot(vec, normal), -1.0, 1.0)
            angle = np.arccos(dot)
        if angle < SIGMA_ORIENTATION:
            classifications.append("inward")
        else:
            classifications.append("outward")
    return classifications




_BUILDER_KEYS = {
    "name",
    "attach_to",
    "ligands_per_peptide",
    "ligands_per_asym_unit",
    "anchor_residue",
    "anchor_distance",
    "anchor_direction",
    "placement",
    "constraint",
    "axis",
    "count",
    "axial_z",
    "distribution",
    "z_range",
    "min_distance",
    "min_ligand_distance",
    "min_protein_distance",
}


def _entity_type(template: dict[str, Any]) -> str:
    kind = str(template.get("type", "protein")).lower()
    if kind not in {"protein", "ligand"}:
        raise ValueError(f"Unsupported material entity type: {kind}")
    return kind


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_ratio(value: Any, field: str) -> Fraction:
    """Parse an integer or positive rational ligand:peptide ratio."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer or rational number")
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{field} must be a positive integer or rational number") from exc
    if ratio <= 0:
        raise ValueError(f"{field} must be a positive integer or rational number")
    return ratio


def _heteromer_partner_slot(topology: str, chain_order: int) -> tuple[int, int | None]:
    """Return the source partner slot and tape side for one lattice chain."""
    if topology != "double_tape":
        return chain_order % 2, None
    side = chain_order % 2
    axial_position = chain_order // 2
    return axial_position % 2, side


def expand_material_spec(
    copies: int,
    asym_unit: list[dict[str, Any]],
    *,
    heteromer_screening: bool = False,
    topology: str | None = None,
    topology_params: dict[str, Any] | None = None,
    charge_bias_strength: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand material templates and return BoltzGen entities plus a sidecar layout.

    The layout's asym indices follow the emitted entity order. The current
    expansion writes all protein templates first in each placement followed by
    ligand instances in config order, making topology indexing independent of
    ligand multiplicity.
    """
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("copies must be a positive integer")
    if not isinstance(asym_unit, list) or not asym_unit:
        raise ValueError("asym_unit must be a non-empty list")

    proteins: list[tuple[int, dict[str, Any]]] = []
    ligands: list[tuple[int, dict[str, Any]]] = []
    names: dict[str, int] = {}
    for template_idx, template in enumerate(asym_unit):
        if not isinstance(template, dict):
            raise ValueError(f"asym_unit entry {template_idx} must be a mapping")
        kind = _entity_type(template)
        name = template.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"asym_unit entry {template_idx} name must be non-empty")
            if name in names:
                raise ValueError(f"Duplicate asym_unit template name: {name}")
        if kind == "protein":
            protein_idx = len(proteins)
            proteins.append((template_idx, template))
            if name is not None:
                names[name] = protein_idx
        else:
            ligands.append((template_idx, template))

    supported_heteromer_topologies = {
        "cyclic",
        "cylindrical",
        "linear_tape",
        "double_tape",
        "open_arc",
        "helical",
    }
    is_heteromer_assembly = heteromer_screening or (
        len(proteins) == 2 and topology in supported_heteromer_topologies
    )
    if heteromer_screening:
        if len(proteins) != 2:
            raise ValueError(
                "heteromer counter-screening requires exactly two protein templates"
            )
        if ligands:
            raise ValueError(
                "heteromer-vs-homomer counter-screening is unsuitable for material "
                "asymmetric units containing ligands; ligand-mediated assembly is "
                "a ternary interaction and is not screened by the current protein-only controls"
            )
        if topology not in supported_heteromer_topologies:
            raise ValueError(
                "heteromer counter-screening supports only cyclic, linear_tape, "
                "double_tape, open_arc, and helical topologies"
            )
        if topology == "double_tape":
            if copies % 4 != 0:
                raise ValueError(
                    f"heteromer counter-screening with double_tape requires copies to be a multiple of 4; got {copies}"
                )
        elif copies % 2 != 0:
            raise ValueError(
                f"heteromer counter-screening requires an even number of copies (multiple of 2); got {copies}"
            )
    elif is_heteromer_assembly:
        if topology == "double_tape":
            if copies % 4 != 0:
                raise ValueError(
                    f"double_tape with alternating partner templates requires copies to be a multiple of 4; got {copies}"
                )
        elif copies % 2 != 0:
            raise ValueError(
                f"{topology} with alternating partner templates requires copies to be an even number (multiple of 2); got {copies}"
            )
    else:
        if len(proteins) > 1 and copies % len(proteins) != 0:
            raise ValueError(
                f"copies ({copies}) must be a multiple of the asymmetric unit protein count ({len(proteins)})"
            )
        if topology == "double_tape" and copies % 2 != 0:
            raise ValueError(
                f"double_tape requires copies to be an even number (multiple of 2); got {copies}"
            )
        if topology == "cage_tetrahedral" and copies != 12:
            raise ValueError(
                f"cage_tetrahedral requires copies to be 12; got {copies}"
            )
        if topology == "cage_octahedral" and copies != 24:
            raise ValueError(
                f"cage_octahedral requires copies to be 24; got {copies}"
            )
        if topology_params is not None:
            if (
                topology == "bilayer_sheet"
                and "grid_dim_x" in topology_params
                and "grid_dim_y" in topology_params
            ):
                expected = (
                    2
                    * int(topology_params["grid_dim_x"])
                    * int(topology_params["grid_dim_y"])
                )
                if copies != expected:
                    raise ValueError(
                        f"bilayer_sheet requires copies to be {expected}; got {copies}"
                    )
            elif (
                topology == "hexagonal_mesh"
                and "grid_dim_x" in topology_params
                and "grid_dim_y" in topology_params
            ):
                expected = (
                    3
                    * int(topology_params["grid_dim_x"])
                    * int(topology_params["grid_dim_y"])
                )
                if copies != expected:
                    raise ValueError(
                        f"hexagonal_mesh requires copies to be {expected}; got {copies}"
                    )
            elif (
                topology == "nanotube"
                and "ring_size" in topology_params
                and "num_tiers" in topology_params
            ):
                expected = int(topology_params["ring_size"]) * int(
                    topology_params["num_tiers"]
                )
                if copies != expected:
                    raise ValueError(
                        f"nanotube requires copies to be {expected}; got {copies}"
                    )
            elif topology == "multi_helical" and "num_starts" in topology_params:
                starts = int(topology_params["num_starts"])
                if copies % starts != 0:
                    raise ValueError(
                        f"multi_helical requires copies to be a multiple of num_starts "
                        f"({starts}); got {copies}"
                    )

    n_placements = copies if is_heteromer_assembly else copies // len(proteins)

    normalized_ligands: list[tuple[int, dict[str, Any], str, int | None, int | Fraction]] = []
    axial_ligands: list[tuple[int, dict[str, Any], list[float], str, list[float], float]] = []
    for template_idx, ligand in ligands:
        placement_mode = str(ligand.get("placement", "")).strip().lower()
        constraint_mode = str(ligand.get("constraint", "")).strip().lower()
        axis_mode = str(ligand.get("axis", "")).strip().lower()
        is_axial = (
            placement_mode in ("z_axis", "axial", "center", "central")
            or constraint_mode in ("z_axis", "axial")
            or axis_mode in ("z", "z_axis")
            or "axial_z" in ligand
        )
        if is_axial:
            if ligand.get("attach_to") is not None:
                raise ValueError(
                    f"attach_to is not supported for axial ligand template {template_idx}"
                )
            if ligand.get("ligands_per_peptide") is not None:
                raise ValueError(
                    f"ligands_per_peptide cannot be combined with axial ligand template {template_idx}"
                )
            if ligand.get("ligands_per_asym_unit") is not None:
                raise ValueError(
                    f"ligands_per_asym_unit cannot be combined with axial ligand template {template_idx}"
                )

            dist_mode = str(ligand.get("distribution", "fixed")).strip().lower()
            if dist_mode in ("random", "even", "stratified", "random_even"):
                dist_mode = "random_even"
            else:
                dist_mode = "fixed"

            raw_z_range = ligand.get("z_range")
            if raw_z_range is None:
                z_range = [-5.0, 5.0]
            elif isinstance(raw_z_range, (int, float)):
                z_range = [-abs(float(raw_z_range)), abs(float(raw_z_range))]
            elif isinstance(raw_z_range, (list, tuple)) and len(raw_z_range) == 2:
                z_range = [float(raw_z_range[0]), float(raw_z_range[1])]
            else:
                raise ValueError(f"z_range must be a float or [min, max] list, got {raw_z_range}")

            min_dist = float(
                ligand.get(
                    "min_ligand_distance", ligand.get("min_distance", 3.5)
                )
            )
            min_prot_dist = float(ligand.get("min_protein_distance", 2.2))

            count_val = ligand.get("count")
            raw_axial_z = ligand.get("axial_z")
            if raw_axial_z is None:
                count = 1 if count_val is None else int(count_val)
                if count == 1:
                    z_offsets = [0.0]
                elif count == 2:
                    half_z = max(min_dist / 2.0, abs(z_range[1]) / 2.0)
                    z_offsets = [-half_z, half_z]
                else:
                    w = (z_range[1] - z_range[0]) / count
                    z_offsets = [z_range[0] + (i + 0.5) * w for i in range(count)]
            elif isinstance(raw_axial_z, (list, tuple)):
                z_offsets = [float(z) for z in raw_axial_z]
                if count_val is not None and int(count_val) != len(z_offsets):
                    raise ValueError(
                        f"count={count_val} does not match len(axial_z)={len(z_offsets)}"
                    )
                count = len(z_offsets)
            else:
                val = float(raw_axial_z)
                if count_val is None:
                    count = 1 if placement_mode in ("center", "central") and val == 0.0 else 1
                else:
                    count = int(count_val)
                if count == 1:
                    z_offsets = [val]
                elif count == 2:
                    z_offsets = [-abs(val), abs(val)] if val != 0.0 else [-min_dist / 2.0, min_dist / 2.0]
                else:
                    raise ValueError("axial_z must be a list of floats when count > 2")

            axial_ligands.append(
                (template_idx, ligand, z_offsets, dist_mode, z_range, min_dist, min_prot_dist)
            )
            continue

        per_peptide = ligand.get("ligands_per_peptide")
        per_unit = ligand.get("ligands_per_asym_unit")
        if per_peptide is not None and per_unit is not None:
            raise ValueError(
                f"ligand template {template_idx} cannot set both ligands_per_peptide "
                "and ligands_per_asym_unit"
            )
        attach_to = ligand.get("attach_to")
        if per_peptide is not None:
            multiplicity = _positive_ratio(per_peptide, "ligands_per_peptide")
            if not proteins:
                raise ValueError("ligands_per_peptide requires a protein template")
            if attach_to is None:
                if len(proteins) != 1:
                    raise ValueError(
                        "attach_to is required for ligands_per_peptide when the "
                        "asym_unit contains multiple protein templates"
                    )
                parent_idx = 0
            elif isinstance(attach_to, bool):
                raise ValueError("attach_to must be a protein template name or zero-based index")
            elif isinstance(attach_to, int):
                parent_idx = attach_to
                if parent_idx < 0 or parent_idx >= len(proteins):
                    raise ValueError(f"attach_to protein index {parent_idx} is out of range")
            elif isinstance(attach_to, str):
                if attach_to not in names:
                    raise ValueError(f"attach_to={attach_to!r} does not name a protein template")
                parent_idx = names[attach_to]
            else:
                raise ValueError("attach_to must be a protein template name or zero-based index")
            normalized_ligands.append((template_idx, ligand, "per_peptide", parent_idx, multiplicity))
        else:
            if attach_to is not None:
                raise ValueError("attach_to is only valid with ligands_per_peptide")
            multiplicity = (
                _positive_int(per_unit, "ligands_per_asym_unit") if per_unit is not None else 1
            )
            normalized_ligands.append((template_idx, ligand, "per_unit", None, multiplicity))

    entities: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    unit_records: list[dict[str, Any]] = []
    for placement_idx in range(n_placements):
        if is_heteromer_assembly:
            partner_slot, side = _heteromer_partner_slot(topology, placement_idx)
            emitted_proteins = [(partner_slot, proteins[partner_slot], side)]
        else:
            emitted_proteins = [
                (slot, protein_template, None)
                for slot, protein_template in enumerate(proteins)
            ]
        unit_record = {
            "placement_index": placement_idx,
            "protein_asym_indices": [],
            "ligand_asym_indices": [],
        }
        protein_asym_by_slot: dict[int, int] = {}

        for protein_slot, (template_idx, protein), side in emitted_proteins:
            asym_index = len(entities)
            chain_id = _chain_id(asym_index)
            entity = _make_entity("protein", protein, chain_id, protein_slot + 1)
            
            if is_heteromer_assembly and charge_bias_strength is not None and float(charge_bias_strength) != 0:
                partner_label = "A" if protein_slot == 0 else "B"
                bias = float(charge_bias_strength)
                if partner_label == "A":
                    weights = {"R": bias * 1.15, "K": bias * 1.15, "D": -bias * 0.33, "E": -bias * 0.33}
                else:
                    weights = {"D": bias, "E": bias, "R": -bias * 0.33, "K": -bias * 0.33}
                
                weights = {k: (0.99 if round(v, 4) == 1.0 else v) for k, v in weights.items()}
                
                chain_length = int(entity["protein"].get("length", len(entity["protein"].get("sequence", ""))))
                if "length" not in entity["protein"] and "sequence" in entity["protein"]:
                    if str(entity["protein"]["sequence"]).isdigit():
                        chain_length = int(entity["protein"]["sequence"])
                
                if topology == "double_tape":
                    position_spec = ",".join(str(p) for p in range(1, chain_length + 1, 2))
                else:
                    position_spec = f"1..{chain_length}"
                
                new_constraint = {
                    "position": position_spec,
                    "weights": weights
                }
                
                existing = entity["protein"].get("residue_constraints", [])
                entity["protein"]["residue_constraints"] = list(existing) + [new_constraint]

            entities.append(entity)
            protein_asym_by_slot[protein_slot] = asym_index
            unit_record["protein_asym_indices"].append(asym_index)
            records.append(
                {
                    "asym_index": asym_index,
                    "chain_id": chain_id,
                    "role": "protein",
                    "placement_index": placement_idx,
                    "protein_slot_index": protein_slot,
                    "protein_template_index": template_idx,
                    "template_name": protein.get("name"),
                    "parent_asym_index": None,
                    **(
                        {
                            "partner_label": "A" if protein_slot == 0 else "B",
                            "topology_order_index": placement_idx,
                            "topology_pattern": "alternating_along_each_side"
                            if topology == "double_tape"
                            else "alternating_in_topology_order",
                            **(
                                {
                                    "double_tape_side": side
                                }
                                if topology == "double_tape"
                                else {}
                            ),
                        }
                        if is_heteromer_assembly
                        else {}
                    ),
                }
            )

        for template_idx, ligand, scope, parent_slot, multiplicity in normalized_ligands:
            if scope == "per_peptide":
                parent_count = n_placements
                total = parent_count * multiplicity
                if total.denominator != 1:
                    raise ValueError(
                        f"ligands_per_peptide={multiplicity} for template {template_idx} does not "
                        f"produce an integer ligand count across {parent_count} peptides"
                    )
                ligand_count = (
                    (placement_idx + 1) * multiplicity.numerator // multiplicity.denominator
                    - placement_idx * multiplicity.numerator // multiplicity.denominator
                )
            else:
                ligand_count = int(multiplicity)
            for instance_idx in range(ligand_count):
                asym_index = len(entities)
                chain_id = _chain_id(asym_index)
                entities.append(_make_entity("ligand", ligand, chain_id, None))
                parent_asym_index = (
                    protein_asym_by_slot[parent_slot] if scope == "per_peptide" else None
                )
                unit_record["ligand_asym_indices"].append(asym_index)
                anchor_res = ligand.get("anchor_residue")
                if anchor_res is None:
                    anchor_res = os.environ.get("MAT_LIGAND_ANCHOR_RESIDUE", "random")
                anchor_dist = ligand.get("anchor_distance")
                if anchor_dist is None and "MAT_LIGAND_ANCHOR_DISTANCE" in os.environ:
                    anchor_dist = float(os.environ["MAT_LIGAND_ANCHOR_DISTANCE"])
                elif anchor_dist is not None:
                    anchor_dist = float(anchor_dist)
                else:
                    anchor_dist = 4.0
                anchor_dir = ligand.get("anchor_direction")
                if anchor_dir is None and "MAT_LIGAND_ANCHOR_DIRECTION" in os.environ:
                    anchor_dir = os.environ["MAT_LIGAND_ANCHOR_DIRECTION"]
                elif anchor_dir is None:
                    anchor_dir = "outward"

                records.append(
                    {
                        "asym_index": asym_index,
                        "chain_id": chain_id,
                        "role": "ligand",
                        "placement_index": placement_idx,
                        "ligand_template_index": template_idx,
                        "ligand_instance_index": instance_idx,
                        "attachment_scope": scope,
                        "parent_asym_index": parent_asym_index,
                        "parent_protein_slot_index": parent_slot,
                        "anchor_residue": anchor_res,
                        "anchor_distance": anchor_dist,
                        "anchor_direction": anchor_dir,
                    }
                )
        unit_records.append(unit_record)

    for (
        template_idx,
        ligand,
        z_offsets,
        dist_mode,
        z_range,
        min_dist,
        min_prot_dist,
    ) in axial_ligands:
        for instance_idx, z_val in enumerate(z_offsets):
            asym_index = len(entities)
            chain_id = _chain_id(asym_index)
            entities.append(_make_entity("ligand", ligand, chain_id, None))
            records.append(
                {
                    "asym_index": asym_index,
                    "chain_id": chain_id,
                    "role": "ligand",
                    "placement_index": None,
                    "ligand_template_index": template_idx,
                    "ligand_instance_index": instance_idx,
                    "attachment_scope": "axial",
                    "constraint": "z_axis",
                    "parent_asym_index": None,
                    "parent_protein_slot_index": None,
                    "axial_z": z_val,
                    "distribution": dist_mode,
                    "z_range": z_range,
                    "min_distance": min_dist,
                    "min_ligand_distance": min_dist,
                    "min_protein_distance": min_prot_dist,
                }
            )

    layout = {
        "version": 1,
        "copies": copies,
        "guided_proteins_per_unit": 1 if is_heteromer_assembly else len(proteins),
        "chains": records,
        "placements": unit_records,
    }
    if topology_params is not None:
        layout["topology_params"] = dict(topology_params)
    if heteromer_screening:
        layout["heteromer_screening"] = {
            "enabled": True,
            "partner_labels": ["A", "B"],
            "topology": topology,
            "copies": copies,
            "source_templates": [protein for _, protein in proteins],
            "placement_pattern": "alternating_along_each_side"
            if topology == "double_tape"
            else "alternating_in_topology_order",
            "protein_template_names": [
                protein.get("name")
                or f"protein_{slot}"
                for slot, (_, protein) in enumerate(proteins)
            ],
            "counter_screen_supported": True,
            "topology_params": dict(topology_params or {}),
        }
    return entities, layout


def _make_entity(
    kind: str, template: dict[str, Any], chain_id: str, default_symmetric_group: int | None
) -> dict[str, Any]:
    entity: dict[str, Any] = {"id": chain_id}
    if kind == "protein":
        entity["sequence"] = str(
            template.get("sequence")
            if template.get("sequence") is not None
            else template.get("length", 15)
        )
        entity["symmetric_group"] = template.get(
            "symmetric_group", default_symmetric_group
        )
        secondary_structure = template.get("secondary_structure")
        if secondary_structure:
            length = template.get("length", 15)
            entity["secondary_structure"] = (
                secondary_structure * length if len(secondary_structure) == 1 else secondary_structure
            )
    else:
        if "ccd" in template:
            entity["ccd"] = template["ccd"]
        if "smiles" in template:
            entity["smiles"] = template["smiles"]
        if "ccd" not in entity and "smiles" not in entity:
            raise ValueError("ligand template must define either ccd or smiles")

    for key, value in template.items():
        if key not in _BUILDER_KEYS | {
            "type",
            "length",
            "sequence",
            "secondary_structure",
            "symmetric_group",
            "ccd",
            "smiles",
            "__material_multiplicity",
        }:
            entity[key] = value
    return {kind: entity}


def _chain_id(idx: int) -> str:
    result = ""
    while idx >= 0:
        result = chr(65 + idx % 26) + result
        idx = idx // 26 - 1
    return result


def write_layout(layout: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.write_text(json.dumps(layout, indent=2) + "\n")
    return str(path)


def build_homomer_counter_screen_entities(
    sequence: str,
    copies: int,
    *,
    partner_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a standard entity list and parent metadata for one homomer screen."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("counter-screen sequence must be a non-empty string")
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("counter-screen copies must be a positive integer")
    if partner_label not in {"A", "B"}:
        raise ValueError("partner_label must be 'A' or 'B'")
    entities = [
        {"protein": {"id": _chain_id(index), "sequence": sequence}}
        for index in range(copies)
    ]
    mapping = {
        "version": 1,
        "screen_type": "homomer",
        "partner_label": partner_label,
        "copies": copies,
        "sequence": sequence,
    }
    return entities, mapping


def build_heteromer_counter_screen_specs(
    partner_sequences: dict[str, str],
    copies: int,
    topology: str,
    topology_params: dict[str, Any],
    parent_design_id: str,
) -> dict[str, dict[str, Any]]:
    """Describe paired homomer jobs derived from one heteromer design.

    The returned records contain ordinary BoltzGen entity specifications and
    explicit parent/child metadata. Structure and prediction files are created
    by the caller because they depend on the source design's coordinates.
    """
    if set(partner_sequences) != {"A", "B"}:
        raise ValueError("partner_sequences must contain exactly keys 'A' and 'B'")
    if any(not isinstance(seq, str) or not seq for seq in partner_sequences.values()):
        raise ValueError("both partner sequences must be non-empty strings")
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 2 or copies % 2:
        raise ValueError("heteromer counter-screen copies must be a positive even integer")
    if topology not in {"cyclic", "cylindrical", "linear_tape", "double_tape", "open_arc", "helical"}:
        raise ValueError(f"Unsupported counter-screen topology: {topology}")
    if not isinstance(parent_design_id, str) or not parent_design_id:
        raise ValueError("parent_design_id must be a non-empty string")

    jobs: dict[str, dict[str, Any]] = {}
    for partner in ("A", "B"):
        screen_id = f"{parent_design_id}__homomer_{partner.lower()}"
        entities, layout = build_homomer_counter_screen_entities(
            partner_sequences[partner], copies, partner_label=partner
        )
        jobs[partner] = {
            "id": screen_id,
            "parent_design_id": parent_design_id,
            "partner_label": partner,
            "topology": topology,
            "topology_params": dict(topology_params),
            "entities": entities,
            "layout": layout,
        }
    return jobs
