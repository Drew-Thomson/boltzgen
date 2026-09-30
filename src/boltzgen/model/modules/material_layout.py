"""Expand material-builder templates into BoltzGen entities and layout metadata."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


_BUILDER_KEYS = {
    "name",
    "attach_to",
    "ligands_per_peptide",
    "ligands_per_asym_unit",
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
        "linear_tape",
        "double_tape",
        "open_arc",
        "helical",
    }
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
        if copies % 2:
            raise ValueError(
                "heteromer counter-screening requires an even number of placements"
            )
        if topology not in supported_heteromer_topologies:
            raise ValueError(
                "heteromer counter-screening supports only cyclic, linear_tape, "
                "double_tape, open_arc, and helical topologies"
            )

    normalized_ligands: list[tuple[int, dict[str, Any], str, int | None, int | Fraction]] = []
    for template_idx, ligand in ligands:
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
    for placement_idx in range(copies):
        if heteromer_screening:
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
                            "topology_pattern": "double_tape_side_alternating"
                            if topology == "double_tape"
                            else "alternating",
                            **(
                                {
                            "double_tape_side": side
                                }
                                if topology == "double_tape"
                                else {}
                            ),
                        }
                        if heteromer_screening
                        else {}
                    ),
                }
            )

        for template_idx, ligand, scope, parent_slot, multiplicity in normalized_ligands:
            if scope == "per_peptide":
                total = copies * multiplicity
                if total.denominator != 1:
                    raise ValueError(
                        f"ligands_per_peptide={multiplicity} for template {template_idx} does not "
                        f"produce an integer ligand count across {copies} peptides"
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
                    }
                )
        unit_records.append(unit_record)

    layout = {
        "version": 1,
        "copies": copies,
        "guided_proteins_per_unit": 1 if heteromer_screening else len(proteins),
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
        for record in records:
            if record["role"] == "protein":
                record.update(
                    {
                        "partner_label": (
                            "A"
                            if record["protein_template_index"] == proteins[0][0]
                            else "B"
                        ),
                        "topology_order_index": int(record["placement_index"]),
                        "topology_pattern": layout["heteromer_screening"]["placement_pattern"],
                        **(
                            {"double_tape_side": _heteromer_partner_slot(
                                topology, int(record["placement_index"])
                            )[1]}
                            if topology == "double_tape"
                            else {}
                        ),
                    }
                )
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
    if topology not in {"cyclic", "linear_tape", "double_tape", "open_arc", "helical"}:
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
