"""Prepare ligand-free (apo) counter-screen inputs from ligand-bearing assemblies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from boltzgen.data import const
from boltzgen.data.data import Structure
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.analyze.filter_target_candidates import (
    candidate_matches,
    load_candidate_ids,
)
from boltzgen.task.task import Task


def extract_protein_only_structure(parsed) -> Structure:
    """Extract only protein chains from parsed structure, removing all ligands.

    Parameters
    ----------
    parsed : ParseResult
        Parsed MMCIF result containing data and sequences.

    Returns
    -------
    Structure
        Extracted structure containing only protein chains.
    """
    protein_chain_indices: list[int] = []
    for idx, chain in enumerate(parsed.data.chains):
        is_protein = (
            int(chain["mol_type"]) == const.chain_type_ids["PROTEIN"]
            or str(chain["name"]) in parsed.sequences
        )
        if is_protein:
            protein_chain_indices.append(idx)

    if not protein_chain_indices:
        raise ValueError("Structure contains no protein chains")

    residue_mask = np.zeros(len(parsed.data.residues), dtype=bool)
    for idx in protein_chain_indices:
        chain = parsed.data.chains[idx]
        start = int(chain["res_idx"])
        count = int(chain["res_num"])
        residue_mask[start : start + count] = True

    return Structure.extract_residues(parsed.data, residue_mask, res_reindex=True)


class PrepareLigandCounterScreens(Task):
    """Strip metal/ligand entities from candidate designs to prepare apo fold inputs."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        moldir: str | None = None,
        candidates_path: str | None = None,
        layout_path: str | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.moldir = moldir
        self.candidates_path = Path(candidates_path) if candidates_path else None
        self.layout_path = Path(layout_path) if layout_path else None

    def run(self, config=None) -> None:  # noqa: ANN001, ARG002
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Counter-screen input directory not found: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = (
            load_candidate_ids(self.candidates_path)
            if self.candidates_path is not None
            else None
        )

        mapping: dict[str, dict[str, str]] = {}
        processed_count = 0

        for source_path in sorted(self.input_dir.glob("*.cif")):
            if source_path.name.endswith("_native.cif"):
                continue
            if candidate_ids is not None and not candidate_matches(
                source_path.stem, candidate_ids
            ):
                continue

            parsed = parse_mmcif(source_path, moldir=self.moldir, use_original_res_idx=False)
            apo_structure = extract_protein_only_structure(parsed)

            parent_id = source_path.stem
            screen_id = f"{parent_id}_apo"

            color_features = np.full(len(apo_structure.residues), 1.0, dtype=np.float32)
            cif_text = to_mmcif(apo_structure, design_coloring=True, color_features=color_features)
            (self.output_dir / f"{screen_id}.cif").write_text(cif_text)
            np.savez_compressed(
                self.output_dir / f"{screen_id}.npz",
                design_mask=np.ones(len(apo_structure.residues), dtype=bool),
            )

            mapping[parent_id] = {"apo": screen_id}
            processed_count += 1

        mapping_file = self.output_dir / "counter_screen_mapping.json"
        mapping_file.write_text(json.dumps(mapping, indent=2) + "\n")
        print(f"Prepared {processed_count} apo counter-screen inputs in {self.output_dir}")
