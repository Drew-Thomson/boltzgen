# Copyright (c) 2024 BoltzGen Contributors
# Distributed under the MIT license.
# ruff: noqa: E501, PLR2004, NPY002, ANN201, INP001

"""Unit tests for physically meaningful piezo deformation and screening."""

from __future__ import annotations

import numpy as np
from scripts.plot_diffusion_points_figure import prepare_helical_trajectory

from boltzgen.data.data import Chain, Structure
from boltzgen.task.predict.piezo_counter_screen import apply_axial_strain


def _build_synthetic_multichain_structure(
    num_chains: int = 4,
    atoms_per_chain: int = 20,
    radius: float = 12.0,
    pitch_dz: float = 5.0,
) -> Structure:
    """Build a synthetic multi-chain helical assembly Structure."""
    np.random.seed(42)
    total_atoms = num_chains * atoms_per_chain

    atom_dtype = [
        ("coords", np.dtype("3f4")),
        ("name", np.dtype("<U4")),
        ("element", np.dtype("<U2")),
        ("is_present", np.dtype("?")),
    ]
    atoms = np.zeros(total_atoms, dtype=atom_dtype)
    atoms["is_present"] = True
    atoms["name"] = "CA"
    atoms["element"] = "C"

    chains = np.zeros(num_chains, dtype=Chain)

    for k in range(num_chains):
        angle = k * (2.0 * np.pi / num_chains)
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        center_z = (k - (num_chains - 1) / 2.0) * pitch_dz

        # Local chain points (small helix along local z)
        local_t = np.linspace(0, 2.0 * np.pi, atoms_per_chain)
        local_coords = np.stack(
            [
                2.0 * np.cos(local_t),
                2.0 * np.sin(local_t),
                np.linspace(-3.0, 3.0, atoms_per_chain),
            ],
            axis=-1,
        )

        start = k * atoms_per_chain
        end = start + atoms_per_chain
        atoms["coords"][start:end] = local_coords + np.array(
            [center_x, center_y, center_z]
        )

        chains[k]["name"] = f"A{k}"
        chains[k]["atom_idx"] = start
        chains[k]["atom_num"] = atoms_per_chain
        chains[k]["res_idx"] = start
        chains[k]["res_num"] = atoms_per_chain

    coords_dtype = [("coords", np.dtype("3f4"))]
    coords = np.zeros(total_atoms, dtype=coords_dtype)
    coords["coords"] = atoms["coords"].copy()

    return Structure(
        atoms=atoms,
        bonds=np.empty(0),
        residues=np.empty(0),
        chains=chains,
        interfaces=np.empty(0),
        mask=np.ones(total_atoms, dtype=bool),
        coords=coords,
        ensemble=np.empty(0),
    )


def test_rigid_centroid_intra_chain_invariance() -> None:
    """Verify rigid_centroid deformation strictly preserves intra-chain distances."""
    np.random.seed(42)
    st = _build_synthetic_multichain_structure(num_chains=4, atoms_per_chain=15)
    strain = 0.15
    poisson_ratio = 0.35

    strained = apply_axial_strain(
        st, strain=strain, mode="rigid_centroid", poisson_ratio=poisson_ratio
    )

    orig_coords = np.array(st.atoms["coords"])
    strained_coords = np.array(strained.atoms["coords"])

    for ch in st.chains:
        start = int(ch["atom_idx"])
        num = int(ch["atom_num"])
        c_orig = orig_coords[start : start + num]
        c_strain = strained_coords[start : start + num]

        dist_orig = np.linalg.norm(c_orig[:, None, :] - c_orig[None, :, :], axis=-1)
        dist_strain = np.linalg.norm(
            c_strain[:, None, :] - c_strain[None, :, :], axis=-1
        )

        max_dist_diff = np.max(np.abs(dist_orig - dist_strain))
        assert max_dist_diff < 1e-5, f"Distances changed by {max_dist_diff:.2e} A"


def test_affine_vs_rigid_centroid_distinction() -> None:
    """Verify affine mode distorts bonds while rigid_centroid preserves them."""
    st = _build_synthetic_multichain_structure(num_chains=4, atoms_per_chain=15)
    strain = 0.15

    affine_strained = apply_axial_strain(st, strain=strain, mode="affine")
    rigid_strained = apply_axial_strain(st, strain=strain, mode="rigid_centroid")

    orig_coords = np.array(st.atoms["coords"])
    aff_coords = np.array(affine_strained.atoms["coords"])
    rig_coords = np.array(rigid_strained.atoms["coords"])

    ch0 = st.chains[0]
    start = int(ch0["atom_idx"])
    num = int(ch0["atom_num"])

    # Affine must change internal z-distances by (1 + strain)
    orig_z_diff = orig_coords[start + num - 1, 2] - orig_coords[start, 2]
    aff_z_diff = aff_coords[start + num - 1, 2] - aff_coords[start, 2]
    rig_z_diff = rig_coords[start + num - 1, 2] - rig_coords[start, 2]

    assert abs(aff_z_diff - orig_z_diff * (1.0 + strain)) < 1e-4
    assert abs(rig_z_diff - orig_z_diff) < 1e-4


def test_poisson_lateral_scaling() -> None:
    """Verify transverse subunit radius expands under compression and contracts under tension."""
    st = _build_synthetic_multichain_structure(
        num_chains=4, atoms_per_chain=10, radius=15.0
    )
    strain_tension = 0.15
    strain_comp = -0.15
    nu = 0.35

    s_ten = apply_axial_strain(
        st, strain=strain_tension, mode="rigid_centroid", poisson_ratio=nu
    )
    s_comp = apply_axial_strain(
        st, strain=strain_comp, mode="rigid_centroid", poisson_ratio=nu
    )

    def _mean_radius(s: Structure) -> float:
        centroids = []
        for ch in s.chains:
            start = int(ch["atom_idx"])
            num = int(ch["atom_num"])
            centroids.append(np.mean(s.atoms["coords"][start : start + num], axis=0))
        c_arr = np.array(centroids)
        center = np.mean(c_arr, axis=0)
        radii = np.linalg.norm(c_arr[:, :2] - center[:2], axis=-1)
        return float(np.mean(radii))

    r_orig = _mean_radius(st)
    r_ten = _mean_radius(s_ten)
    r_comp = _mean_radius(s_comp)

    # Tension along z => transverse Poisson contraction (r_ten < r_orig)
    assert r_ten < r_orig
    assert abs((r_ten - r_orig) / r_orig - (-nu * strain_tension)) < 1e-3

    # Compression along z => transverse Poisson expansion (r_comp > r_orig)
    assert r_comp > r_orig
    assert abs((r_comp - r_orig) / r_orig - (-nu * strain_comp)) < 1e-3


def test_prepare_helical_trajectory_rigid_centroid() -> None:
    """Verify trajectory point extraction with rigid_centroid deformation and Poisson coupling."""
    num_chains = 4
    atoms_per_chain = 20
    total_atoms = num_chains * atoms_per_chain

    np.random.seed(42)
    coords = np.zeros((3, total_atoms, 3), dtype=np.float32)
    asym_id = np.repeat(np.arange(num_chains), atoms_per_chain)
    atom_pad_mask = np.ones(total_atoms, dtype=np.float32)

    for step in range(3):
        for k in range(num_chains):
            angle = k * (2.0 * np.pi / num_chains)
            cx, cy, cz = 10.0 * np.cos(angle), 10.0 * np.sin(angle), (k - 1.5) * 5.0
            coords[step, k * atoms_per_chain : (k + 1) * atoms_per_chain] = (
                np.random.randn(atoms_per_chain, 3) * 0.5 + np.array([cx, cy, cz])
            )

    traj_data = {
        "coords": coords,
        "asym_id": asym_id,
        "atom_pad_mask": atom_pad_mask,
        "step_indices": np.array([0, 1, 2]),
    }

    frames_unstrained, _, _, _ = prepare_helical_trajectory(
        traj_data,
        num_timepoints=2,
        frame_indices=[2],
        align_to_axis=False,
        axial_strain=0.0,
    )
    frames_strained, _, _, _ = prepare_helical_trajectory(
        traj_data,
        num_timepoints=2,
        frame_indices=[2],
        align_to_axis=False,
        axial_strain=0.15,
        poisson_ratio=0.35,
        deformation_mode="rigid_centroid",
    )

    p_orig = frames_unstrained[-1]
    p_strain = frames_strained[-1]

    # Verify intra-chain distance invariance
    for k in range(num_chains):
        c_orig = p_orig[asym_id == k]
        c_strain = p_strain[asym_id == k]
        dist_orig = np.linalg.norm(c_orig[:, None, :] - c_orig[None, :, :], axis=-1)
        dist_strain = np.linalg.norm(
            c_strain[:, None, :] - c_strain[None, :, :], axis=-1
        )
        assert np.max(np.abs(dist_orig - dist_strain)) < 1e-4
