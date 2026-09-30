"""Prepare topology-matched homomer inputs from inverse-folded heteromers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boltzgen.data.data import Structure
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.modules.material_layout import _chain_id, build_heteromer_counter_screen_specs
from boltzgen.model.modules.materials import generate_ideal_lattice
from boltzgen.task.task import Task


def _place_chain(seed: Structure, center: torch.Tensor, rotation: torch.Tensor) -> Structure:
    """Clone a one-chain structure and place it using row-vector transforms."""
    coords = seed.coords.copy()
    atoms = seed.atoms.copy()
    source = torch.as_tensor(coords["coords"], dtype=center.dtype)
    source_center = source.mean(dim=0)
    placed = (source - source_center) @ rotation + center
    coords["coords"] = placed.cpu().numpy()
    atom_coords = torch.as_tensor(atoms["coords"], dtype=center.dtype)
    atoms["coords"] = (
        (atom_coords - source_center) @ rotation + center
    ).cpu().numpy()
    chains = seed.chains.copy()
    chains["name"] = np.asarray([_chain_id(0)])
    chains["asym_id"] = 0
    chains["entity_id"] = 0
    chains["sym_id"] = 1
    return Structure(
        atoms=atoms,
        bonds=seed.bonds.copy(),
        residues=seed.residues.copy(),
        chains=chains,
        interfaces=seed.interfaces.copy(),
        mask=seed.mask.copy(),
        coords=coords,
        ensemble=seed.ensemble.copy(),
    )


def _extract_partner(parsed, chain_name: str) -> tuple[Structure, str]:
    chain_indices = np.flatnonzero(parsed.data.chains["name"] == chain_name)
    if len(chain_indices) != 1:
        raise ValueError(f"Expected one source chain named {chain_name!r}")
    chain = parsed.data.chains[int(chain_indices[0])]
    residue_mask = np.zeros(len(parsed.data.residues), dtype=bool)
    residue_mask[chain["res_idx"] : chain["res_idx"] + chain["res_num"]] = True
    structure = Structure.extract_residues(parsed.data, residue_mask, res_reindex=True)
    sequence = parsed.sequences.get(chain_name)
    if not sequence:
        raise ValueError(f"No parsed polymer sequence for chain {chain_name!r}")
    return structure, sequence


def _build_homomer(
    seed: Structure,
    sequence: str,
    topology: str,
    copies: int,
    params: dict[str, Any],
) -> Structure:
    centers, rotations = generate_ideal_lattice(
        topology,
        copies,
        1,
        params,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    result = None
    for index in range(copies):
        placed = _place_chain(seed, centers[index], rotations[index])
        if result is None:
            result = placed
        else:
            result = Structure.concatenate(result, placed)
    if result is None:
        raise ValueError("counter-screen copy count must be positive")
    if len(sequence) != int(result.chains[0]["res_num"]):
        raise ValueError("parsed partner sequence length does not match source structure")
    return result


class PrepareHeteromerCounterScreens(Task):
    """Expand each inverse-folded A/B candidate into A-only and B-only fold inputs."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        layout_path: str,
        topology: str,
        copies: int,
        topology_params: dict[str, Any] | None = None,
        moldir: str | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.layout_path = Path(layout_path)
        self.topology = topology
        self.copies = copies
        self.topology_params = topology_params or {}
        self.moldir = moldir
        self.num_chains = copies

    def run(self, config=None) -> None:  # noqa: ANN001, ARG002
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Counter-screen input directory not found: {self.input_dir}")
        layout = json.loads(self.layout_path.read_text())
        if not layout.get("heteromer_screening", {}).get("enabled"):
            raise ValueError("material layout does not enable heteromer counter-screening")
        if any(record.get("role") == "ligand" for record in layout.get("chains", [])):
            raise ValueError(
                "heteromer-vs-homomer counter-screening is unsuitable for ligand-bearing "
                "complexes: ligand-mediated assembly is a ternary interaction not tested "
                "by the current protein-only homomer controls"
            )
        partner_templates = layout.get("heteromer_screening", {}).get(
            "protein_template_names", []
        )
        if len(partner_templates) != 2:
            raise ValueError("material layout must define exactly two partner templates")
        template_definitions = layout.get("heteromer_screening", {}).get(
            "source_templates", []
        )
        if len(template_definitions) != 2:
            raise ValueError("material layout does not persist both partner template definitions")
        self.num_chains = int(layout["copies"])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, dict[str, str]] = {}
        for source_path in sorted(self.input_dir.glob("*.cif")):
            if source_path.name.endswith("_native.cif"):
                continue
            parsed = parse_mmcif(source_path, moldir=self.moldir, use_original_res_idx=False)
            source_ids: dict[str, tuple[str, int]] = {}
            source_chain_names = [str(chain["name"]) for chain in parsed.data.chains]
            expected_chain_count = (
                self.num_chains
                if layout["guided_proteins_per_unit"] == 1
                else self.num_chains * 2
            )
            if len(source_chain_names) != expected_chain_count:
                raise ValueError(
                    f"Expected {expected_chain_count} protein chains in {source_path.name}, "
                    f"found {len(source_chain_names)}"
                )
            layout_chain_ids = {
                str(record["chain_id"])
                for record in layout["chains"]
                if record.get("role") == "protein"
            }
            if set(source_chain_names) != layout_chain_ids:
                raise ValueError(
                    f"Material layout chains do not match {source_path.name}: "
                    f"expected {sorted(layout_chain_ids)}, found {sorted(source_chain_names)}"
                )
            for chain in parsed.data.chains:
                chain_name = str(chain["name"])
                record = next(
                    (
                        item
                        for item in layout["chains"]
                        if item.get("role") == "protein"
                        and item.get("chain_id") == chain_name
                    ),
                    None,
                )
                if record is None:
                    raise ValueError(
                        f"Material layout has no record for chain {chain_name!r} in {source_path.name}"
                    )
                partner = record.get("partner_label")
                if partner in {"A", "B"}:
                    source_ids.setdefault(
                        str(partner), (chain_name, int(record["placement_index"]))
                    )
            if layout["guided_proteins_per_unit"] == 1 and len(source_ids) != 2:
                raise ValueError(
                    "Heteromer layout does not contain both alternating partner chains "
                    f"in {source_path.name}"
                )
            if set(source_ids) != {"A", "B"}:
                raise ValueError(f"Could not map both heteromer partners in {source_path.name}")

            parent_id = source_path.stem
            mapping[parent_id] = {}
            partner_sequences = {
                partner: _extract_partner(parsed, source_ids[partner])[1]
                for partner in ("A", "B")
            }
            job_specs = build_heteromer_counter_screen_specs(
                partner_sequences,
                self.num_chains,
                self.topology,
                self.topology_params,
                parent_id,
            )
            for partner in ("A", "B"):
                chain_name, _placement_idx = source_ids[partner]
                seed, sequence = _extract_partner(parsed, chain_name)
                template = template_definitions[0 if partner == "A" else 1]
                template_sequence = template.get("sequence")
                if template_sequence is not None and len(sequence) != len(str(template_sequence)):
                    raise ValueError(
                        f"Partner {partner} sequence length from source does not match layout"
                    )
                homomer = _build_homomer(
                    seed,
                    sequence,
                    self.topology,
                    self.num_chains,
                    self.topology_params,
                )
                screen_id = job_specs[partner]["id"]
                # Mark all homomer residues as designed so no chain is treated as
                # an unmarked target/template during the counter-screen fold.
                color_features = np.full(len(homomer.residues), 1.0, dtype=np.float32)
                (self.output_dir / f"{screen_id}.cif").write_text(
                    to_mmcif(homomer, design_coloring=True, color_features=color_features)
                )
                np.savez_compressed(
                    self.output_dir / f"{screen_id}.npz",
                    design_mask=np.ones(len(homomer.residues), dtype=bool),
                )
                mapping[parent_id][partner] = screen_id

        (self.output_dir / "counter_screen_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
