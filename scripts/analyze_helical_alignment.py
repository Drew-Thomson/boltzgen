"""Analyze helical alignment, chain tilts, and polar metrics for helical assemblies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import gemmi
import numpy as np


def analyze_structure_helical_alignment(
    cif_path: str | Path,
) -> Dict[str, float | List[float]]:
    """Compute per-chain helical tilt angles from the assembly Z axis.

    Parameters
    ----------
    cif_path : str or Path
        Path to structure CIF file.

    Returns
    -------
    dict
        Dictionary containing mean_tilt_deg, std_tilt_deg, per_chain_tilts,
        and assembly length along Z.
    """
    st = gemmi.read_structure(str(cif_path))
    model = st[0]

    chain_tilts = []
    chain_lengths = []
    chain_names = []
    chain_coms = []

    # Compute assembly long axis via SVD of centered CA coordinates of all chains
    all_cas = np.array(
        [a.pos.tolist() for chain in model for res in chain for a in res if a.name == "CA"],
        dtype=np.float64,
    )
    all_centered = all_cas - all_cas.mean(axis=0)
    u_all, _, _ = np.linalg.svd(all_centered.T @ all_centered)
    assembly_long_axis = u_all[:, 0]

    for chain in model:
        cas = np.array(
            [a.pos.tolist() for res in chain for a in res if a.name == "CA"],
            dtype=np.float64,
        )
        if len(cas) < 4:
            continue
        com = cas.mean(axis=0)
        centered = cas - com
        _, _, vh = np.linalg.svd(centered)
        helix_axis = vh[0]  # Primary eigenvector

        # Compute acute angle with assembly long axis
        cos_theta = abs(np.dot(helix_axis, assembly_long_axis))
        tilt_deg = float(np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0))))

        chain_tilts.append(tilt_deg)
        chain_lengths.append(len(cas))
        chain_names.append(chain.name)
        chain_coms.append(com)

    coms_arr = np.array(chain_coms)
    z_span = float(coms_arr[:, 2].max() - coms_arr[:, 2].min()) if len(coms_arr) > 0 else 0.0

    return {
        "num_chains": len(chain_tilts),
        "mean_tilt_deg": float(np.mean(chain_tilts)),
        "std_tilt_deg": float(np.std(chain_tilts)),
        "min_tilt_deg": float(np.min(chain_tilts)),
        "max_tilt_deg": float(np.max(chain_tilts)),
        "per_chain_tilts": chain_tilts,
        "chain_names": chain_names,
        "z_span": z_span,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze helical alignment from Z axis")
    parser.add_argument("cif_files", nargs="+", help="Paths to CIF files to analyze")
    args = parser.parse_args()

    print(f"{'Structure':<45} | {'Chains':<6} | {'Mean Tilt (deg)':<15} | {'Std':<6} | {'Min-Max (deg)':<15} | {'Z-span (A)':<10}")
    print("-" * 115)
    for f in args.cif_files:
        res = analyze_structure_helical_alignment(f)
        min_max = f"{res['min_tilt_deg']:.1f} - {res['max_tilt_deg']:.1f}"
        print(
            f"{Path(f).name:<45} | {res['num_chains']:<6} | {res['mean_tilt_deg']:<15.2f} | {res['std_tilt_deg']:<6.2f} | {min_max:<15} | {res['z_span']:<10.1f}"
        )


if __name__ == "__main__":
    main()
