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


def expand_material_spec(
    copies: int, asym_unit: list[dict[str, Any]]
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
        unit_record = {
            "placement_index": placement_idx,
            "protein_asym_indices": [],
            "ligand_asym_indices": [],
        }
        protein_asym_by_slot: dict[int, int] = {}

        for protein_slot, (template_idx, protein) in enumerate(proteins):
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
        "guided_proteins_per_unit": len(proteins),
        "chains": records,
        "placements": unit_records,
    }
    return entities, layout


def _make_entity(
    kind: str, template: dict[str, Any], chain_id: str, default_symmetric_group: int | None
) -> dict[str, Any]:
    entity: dict[str, Any] = {"id": chain_id}
    if kind == "protein":
        entity["sequence"] = str(template.get("length", 15))
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
