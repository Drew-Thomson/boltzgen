"""Generate tri-state deformation counter-screen inputs for piezoelectric evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from boltzgen.data.data import Structure
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.analyze.filter_target_candidates import (
    candidate_matches,
    load_candidate_ids,
)
from boltzgen.task.task import Task


def apply_axial_strain(structure: Structure, strain: float) -> Structure:
    r"""Apply uniform axial strain along the principal z-axis to a multi-chain structure.

    .. math::
        z' = z_{\text{centroid}} + (z - z_{\text{centroid}}) \cdot (1 + \epsilon)

    Parameters
    ----------
    structure : Structure
        Input multi-chain peptide structure.
    strain : float
        Engineering strain \epsilon (e.g. +0.10 for tension, -0.10 for compression).

    Returns
    -------
    Structure
        New structure with axially strained atom and model coordinates.
    """
    coords = structure.coords.copy()
    atoms = structure.atoms.copy()

    # Apply strain to atom coordinates
    atom_arr = np.array(atoms["coords"], copy=True)
    z_centroid = float(np.mean(atom_arr[:, 2]))
    atom_arr[:, 2] = z_centroid + (atom_arr[:, 2] - z_centroid) * (1.0 + strain)
    atoms["coords"] = atom_arr

    # Apply strain to model coords
    coords_arr = np.array(coords["coords"], copy=True)
    coords_arr[:, 2] = z_centroid + (coords_arr[:, 2] - z_centroid) * (1.0 + strain)
    coords["coords"] = coords_arr

    return Structure(
        atoms=atoms,
        bonds=structure.bonds.copy(),
        residues=structure.residues.copy(),
        chains=structure.chains.copy(),
        interfaces=structure.interfaces.copy(),
        mask=structure.mask.copy(),
        coords=coords,
        ensemble=structure.ensemble.copy(),
    )


class PreparePiezoCounterScreens(Task):
    """Generate tensile (+eps) and compressive (-eps) variants of each candidate."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        strain_amplitude: float = 0.10,
        candidates_path: str | None = None,
        moldir: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.strain = float(strain_amplitude)
        self.candidates_path = Path(candidates_path) if candidates_path else None
        self.moldir = moldir

    def run(self, config: Any = None) -> None:  # noqa: ANN401, ARG002
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Piezo counter-screen input directory not found: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = (
            load_candidate_ids(self.candidates_path)
            if self.candidates_path is not None
            else None
        )

        mapping: dict[str, dict[str, Any]] = {}
        for source_path in sorted(self.input_dir.glob("*.cif")):
            if source_path.name.endswith("_native.cif"):
                continue
            if candidate_ids is not None and not candidate_matches(source_path.stem, candidate_ids):
                continue

            parsed = parse_mmcif(source_path, moldir=self.moldir, use_original_res_idx=False)
            parent_id = source_path.stem

            tension_id = f"{parent_id}_tension"
            compression_id = f"{parent_id}_compression"

            strained_tension = apply_axial_strain(parsed.data, strain=self.strain)
            strained_comp = apply_axial_strain(parsed.data, strain=-self.strain)

            color_features = np.full(len(parsed.data.residues), 1.0, dtype=np.float32)
            design_mask = np.ones(len(parsed.data.residues), dtype=bool)

            # Write tension variant
            (self.output_dir / f"{tension_id}.cif").write_text(
                to_mmcif(strained_tension, design_coloring=True, color_features=color_features)
            )
            np.savez_compressed(
                self.output_dir / f"{tension_id}.npz",
                design_mask=design_mask,
            )

            # Write compression variant
            (self.output_dir / f"{compression_id}.cif").write_text(
                to_mmcif(strained_comp, design_coloring=True, color_features=color_features)
            )
            np.savez_compressed(
                self.output_dir / f"{compression_id}.npz",
                design_mask=design_mask,
            )

            mapping[parent_id] = {
                "tension": tension_id,
                "compression": compression_id,
                "strain_tension": self.strain,
                "strain_compression": self.strain,
            }

        (self.output_dir / "piezo_counter_screen_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
