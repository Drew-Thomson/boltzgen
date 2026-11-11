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


def apply_axial_strain(
    structure: Structure,
    strain: float,
    mode: str = "rigid_centroid",
    poisson_ratio: float = 0.35,
) -> Structure:
    r"""Apply axial strain along principal z-axis to a multi-chain structure.

    When ``mode == "rigid_centroid"`` and multiple chains exist, each chain is
    translated as a rigid body according to the displacement of its center of mass
    under axial strain :math:`\epsilon_z` and Poisson transverse strain
    :math:`\epsilon_{xy} = -\nu \epsilon_z`:

    .. math::
        \mathbf{c}_k = \frac{1}{N_k} \sum_{i \in \text{chain } k} \mathbf{r}_i, \quad
        \mathbf{c}_{\text{assembly}} = \frac{1}{K} \sum_{k=0}^{K-1} \mathbf{c}_k

    .. math::
        \Delta c_{k, z} = (c_{k, z} - c_{\text{assembly}, z}) \cdot \epsilon_z

    .. math::
        \Delta c_{k, x} = -(c_{k, x} - c_{\text{assembly}, x}) \nu \epsilon_z

    .. math::
        \Delta c_{k, y} = -(c_{k, y} - c_{\text{assembly}, y}) \nu \epsilon_z

    .. math::
        \mathbf{r}_{i \in \text{chain } k}' = \mathbf{r}_i + \Delta \mathbf{c}_k

    This preserves 100% of intra-chain covalent bond lengths, angles, and secondary
    structures while physically displacing subunit centroids along the assembly axis and
    accounting for lateral Poisson contraction/expansion.

    When ``mode == "affine"`` or for single-chain systems, uniform 1D affine coordinate
    scaling is applied:

    .. math::
        z' = z_{\text{centroid}} + (z - z_{\text{centroid}}) \cdot (1 + \epsilon)

    Parameters
    ----------
    structure : Structure
        Input multi-chain peptide structure.
    strain : float
        Engineering strain :math:`\epsilon` (+0.15 tension, -0.15 compression).
    mode : str, default="rigid_centroid"
        Deformation mode: ``"rigid_centroid"`` for rigid-body chain centroid
        displacement with Poisson coupling, or ``"affine"`` for 1D affine scaling.
    poisson_ratio : float, default=0.35
        Effective Poisson's ratio :math:`\nu` governing transverse strain
        :math:`\epsilon_{xy} = -\nu \epsilon_z`.

    Returns
    -------
    Structure
        New structure with axially strained atom and model coordinates.
    """
    coords = structure.coords.copy()
    atoms = structure.atoms.copy()
    atom_arr = np.array(atoms["coords"], copy=True)
    coords_arr = np.array(coords["coords"], copy=True)

    has_multiple_chains = (
        hasattr(structure, "chains")
        and structure.chains is not None
        and len(structure.chains) > 1
    )

    if mode == "rigid_centroid" and has_multiple_chains:
        valid_centroids: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, ch in enumerate(structure.chains):
            start = int(ch["atom_idx"])
            num = int(ch["atom_num"])
            if num > 0:
                c_k = np.mean(atom_arr[start : start + num], axis=0)
                valid_centroids.append(c_k)
                valid_indices.append(idx)

        if valid_centroids:
            centroids_arr = np.array(valid_centroids, dtype=np.float64)  # [K, 3]
            assembly_center = np.mean(centroids_arr, axis=0)  # [3]

            for c_k, idx in zip(centroids_arr, valid_indices):
                ch = structure.chains[idx]
                start = int(ch["atom_idx"])
                num = int(ch["atom_num"])
                delta = np.zeros(3, dtype=np.float64)
                delta[0] = -(c_k[0] - assembly_center[0]) * poisson_ratio * strain
                delta[1] = -(c_k[1] - assembly_center[1]) * poisson_ratio * strain
                delta[2] = (c_k[2] - assembly_center[2]) * strain

                atom_arr[start : start + num] += delta
                coords_arr[start : start + num] += delta
        else:
            z_centroid = float(np.mean(atom_arr[:, 2]))
            atom_arr[:, 2] = z_centroid + (atom_arr[:, 2] - z_centroid) * (1.0 + strain)
            coords_arr[:, 2] = (
                z_centroid + (coords_arr[:, 2] - z_centroid) * (1.0 + strain)
            )
    else:
        # Fallback 1D affine coordinate scaling along z
        z_centroid = float(np.mean(atom_arr[:, 2]))
        atom_arr[:, 2] = z_centroid + (atom_arr[:, 2] - z_centroid) * (1.0 + strain)
        coords_arr[:, 2] = (
            z_centroid + (coords_arr[:, 2] - z_centroid) * (1.0 + strain)
        )

    atoms["coords"] = atom_arr
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
        mode: str = "rigid_centroid",
        poisson_ratio: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.strain = float(strain_amplitude)
        self.candidates_path = Path(candidates_path) if candidates_path else None
        self.moldir = moldir
        self.mode = mode
        self.poisson_ratio = float(poisson_ratio)

    def run(self, config: Any = None) -> None:  # noqa: ANN401, ARG002
        """Execute generation of tension and compression counter-screen structures."""
        if not self.input_dir.is_dir():
            msg = f"Piezo counter-screen input directory not found: {self.input_dir}"
            raise FileNotFoundError(msg)

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
            if candidate_ids is not None and not candidate_matches(
                source_path.stem, candidate_ids
            ):
                continue

            parsed = parse_mmcif(
                source_path, moldir=self.moldir, use_original_res_idx=False
            )
            parent_id = source_path.stem

            tension_id = f"{parent_id}_tension"
            compression_id = f"{parent_id}_compression"

            strained_tension = apply_axial_strain(
                parsed.data,
                strain=self.strain,
                mode=self.mode,
                poisson_ratio=self.poisson_ratio,
            )
            strained_comp = apply_axial_strain(
                parsed.data,
                strain=-self.strain,
                mode=self.mode,
                poisson_ratio=self.poisson_ratio,
            )

            color_features = np.full(len(parsed.data.residues), 1.0, dtype=np.float32)
            design_mask = np.ones(len(parsed.data.residues), dtype=bool)

            # Write tension variant
            (self.output_dir / f"{tension_id}.cif").write_text(
                to_mmcif(
                    strained_tension,
                    design_coloring=True,
                    color_features=color_features,
                )
            )
            np.savez_compressed(
                self.output_dir / f"{tension_id}.npz",
                design_mask=design_mask,
            )

            # Write compression variant
            (self.output_dir / f"{compression_id}.cif").write_text(
                to_mmcif(
                    strained_comp,
                    design_coloring=True,
                    color_features=color_features,
                )
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
                "mode": self.mode,
                "poisson_ratio": self.poisson_ratio,
            }

        (self.output_dir / "piezo_counter_screen_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
