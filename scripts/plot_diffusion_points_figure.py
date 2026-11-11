#!/usr/bin/env python3
# Copyright (c) 2024 BoltzGen Contributors
# Distributed under the MIT license.

"""Publication-quality figure generation comparing diffusion point clouds.

Generates a two-row multi-panel 3D point-cloud figure:
- Row 1: Traditional Binder Design (fixed target context + coalescing binder)
- Row 2: Symmetry-Driven Helical Design (multi-chain assembly along screw axis)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

EPSILON = 1e-8
ATOM14_NUM_ATOMS = 14
BACKBONE_ATOMS_PER_RES = 4
CA_ATOM_OFFSET = 1
TAB10_MAX_COLORS = 10


def kabsch_rigid_align(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    coords_to_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Rigidly align 3D coordinates using the Kabsch algorithm.

    Computes the optimal rotation matrix $R \in SO(3)$ and translation
    vector $\mathbf{t} \in \mathbb{R}^3$ minimizing root-mean-square deviation:
    .. math::
        \min_{R, \mathbf{t}} \sum_{i=1}^N \| R (\mathbf{p}_i - \bar{\mathbf{p}})
        + \bar{\mathbf{q}} - \mathbf{q}_i \|^2

    Parameters
    ----------
    source_coords : np.ndarray
        Source reference points to align from, shape `(N, 3)`.
    target_coords : np.ndarray
        Target reference points to align to, shape `(N, 3)`.
    coords_to_transform : np.ndarray, optional
        Additional points to apply the derived rotation and translation to,
        shape `(M, 3)`. If None, transforms `source_coords`.

    Returns
    -------
    np.ndarray
        Transformed coordinates, shape `(M, 3)` or `(N, 3)`.
    """
    if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        if coords_to_transform is not None:
            return coords_to_transform
        return source_coords

    pts_to_apply = (
        coords_to_transform if coords_to_transform is not None else source_coords
    )

    center_source = np.mean(source_coords, axis=0, keepdims=True)
    center_target = np.mean(target_coords, axis=0, keepdims=True)

    centered_source = source_coords - center_source
    centered_target = target_coords - center_target

    cov = centered_source.T @ centered_target
    u, _, vh = np.linalg.svd(cov)

    rot = vh.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vh[-1, :] *= -1.0
        rot = vh.T @ u.T

    transformed = (pts_to_apply - center_source) @ rot.T + center_target
    return transformed


def orient_coords_along_z(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Orient 3D coordinates such that the principal axis aligns with the Z-axis.

    Uses Singular Value Decomposition (SVD) of the centered covariance matrix:
    .. math::
        \text{Cov} = \frac{1}{N} X_c^T X_c = U \Sigma V^T

    Parameters
    ----------
    coords : np.ndarray
        Points to orient, shape `(N, 3)`.

    Returns
    -------
    oriented_coords : np.ndarray
        Oriented coordinates, shape `(N, 3)`.
    rot_matrix : np.ndarray
        Orthonormal rotation matrix applied, shape `(3, 3)`.
    """
    center = np.mean(coords, axis=0, keepdims=True)
    centered = coords - center
    cov = centered.T @ centered
    u, _, _ = np.linalg.svd(cov)

    axis_z = u[:, 0]
    axis_y = u[:, 1]
    axis_x = np.cross(axis_y, axis_z)
    norm_x = np.linalg.norm(axis_x)
    if norm_x > EPSILON:
        axis_x /= norm_x
    else:
        axis_x = np.array([1.0, 0.0, 0.0])
    axis_y = np.cross(axis_z, axis_x)

    rot = np.stack([axis_x, axis_y, axis_z], axis=0)
    oriented = (coords - center) @ rot.T
    return oriented, rot


def orient_binder_vertical_right(
    coords: np.ndarray,
    is_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Derive rotation and center orienting binder helix vertically on the right.

    Aligns the principal axis of the binder atoms along the vertical Z-axis,
    and rotates the target-to-binder vector onto the horizontal X-axis such that
    the binder is situated at $x > 0$ (right side of image) and the target at
    $x < 0$ (left side of image).

    Parameters
    ----------
    coords : np.ndarray
        Reference frame atomic coordinates, shape `(N, 3)`.
    is_target : np.ndarray
        Boolean mask of fixed target atoms, shape `(N,)`.

    Returns
    -------
    oriented_coords : np.ndarray
        Transformed coordinates, shape `(N, 3)`.
    rot_matrix : np.ndarray
        Orthonormal rotation matrix applied, shape `(3, 3)`.
    center : np.ndarray
        Centering offset applied prior to rotation, shape `(3,)`.
    """
    binder_pts = coords[~is_target]
    target_pts = coords[is_target]
    if len(binder_pts) == 0 or len(target_pts) == 0:
        center = np.mean(coords, axis=0) if len(coords) > 0 else np.zeros(3)
        return coords, np.eye(3), center

    c_b = np.mean(binder_pts, axis=0)
    c_t = np.mean(target_pts, axis=0)
    cov_b = (binder_pts - c_b).T @ (binder_pts - c_b)
    u_b, _, _ = np.linalg.svd(cov_b)
    axis_z = u_b[:, 0]

    v_tb = c_b - c_t
    v_x = v_tb - np.dot(v_tb, axis_z) * axis_z
    norm_x = np.linalg.norm(v_x)
    axis_x = v_x / norm_x if norm_x > EPSILON else u_b[:, 1]
    axis_y = np.cross(axis_z, axis_x)
    norm_y = np.linalg.norm(axis_y)
    axis_y = axis_y / norm_y if norm_y > EPSILON else np.array([0.0, 1.0, 0.0])
    axis_x = np.cross(axis_y, axis_z)

    rot = np.stack([axis_x, axis_y, axis_z], axis=0)
    center = (np.min(coords, axis=0) + np.max(coords, axis=0)) / 2.0
    oriented = (coords - center) @ rot.T
    return oriented, rot, center


def resolve_filepath(pattern_or_path: str) -> Path:
    """Resolve a file path or glob pattern to a single existing file.

    Parameters
    ----------
    pattern_or_path : str
        Direct path or glob string pattern.

    Returns
    -------
    Path
        Resolved file path.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    path_obj = Path(pattern_or_path)
    if path_obj.is_file():
        return path_obj.resolve()

    parent_dir = path_obj.parent if path_obj.parent != Path() else Path()
    pattern_name = path_obj.name
    matches = sorted(parent_dir.glob(pattern_name))
    if not matches:
        msg = f"No trajectory file matched pattern: {pattern_or_path}"
        raise FileNotFoundError(msg)
    return matches[0].resolve()


def load_trajectory_data(
    file_path: Path,
    *,
    use_x0: bool = False,
) -> Dict[str, np.ndarray]:
    r"""Load diffusion trajectory archive (.npz).

    Parameters
    ----------
    file_path : Path
        Path to .npz trajectory archive.
    use_x0 : bool, default False
        Whether to load denoised $\hat{x}_0$ coordinates instead of noisy $x_t$.

    Returns
    -------
    Dict[str, np.ndarray]
        Loaded trajectory features:
        - `coords`: shape `(T, N, 3)`
        - `is_target`: shape `(N,)`
        - `asym_id`: shape `(N,)`
        - `atom_pad_mask`: shape `(N,)`
        - `step_indices`: shape `(T,)`
    """
    with np.load(file_path, allow_pickle=True) as data:
        if use_x0 and "x0_coords" in data and data["x0_coords"] is not None:
            coords = data["x0_coords"]
            step_indices = (
                data["x0_step_indices"]
                if "x0_step_indices" in data
                else (
                    data["step_indices"]
                    if "step_indices" in data
                    else np.arange(coords.shape[0])
                )
            )
        elif "coords" in data and data["coords"] is not None:
            coords = data["coords"]
            step_indices = (
                data["step_indices"]
                if "step_indices" in data
                else np.arange(coords.shape[0])
            )
        elif "x0_coords" in data and data["x0_coords"] is not None:
            coords = data["x0_coords"]
            step_indices = (
                data["x0_step_indices"]
                if "x0_step_indices" in data
                else np.arange(coords.shape[0])
            )
        else:
            msg = f"No coordinate arrays found in {file_path}."
            raise KeyError(msg)

        num_atoms = coords.shape[1]
        is_target = (
            data["is_target"].astype(bool)
            if "is_target" in data
            else np.zeros(num_atoms, dtype=bool)
        )
        asym_id = (
            data["asym_id"]
            if "asym_id" in data
            else np.zeros(num_atoms, dtype=int)
        )
        atom_pad_mask = (
            data["atom_pad_mask"].astype(bool)
            if "atom_pad_mask" in data
            else np.ones(num_atoms, dtype=bool)
        )

    return {
        "coords": coords,
        "is_target": is_target,
        "asym_id": asym_id,
        "atom_pad_mask": atom_pad_mask,
        "step_indices": step_indices,
    }


def compute_diffusion_times(
    step_indices: Optional[np.ndarray],
    num_frames: int,
) -> np.ndarray:
    r"""Compute diffusion time parameter $t \in [1.0, 0.0]$ for trajectory frames.

    Parameters
    ----------
    step_indices : np.ndarray, optional
        1D array of diffusion step indices, shape `(T,)`.
    num_frames : int
        Total number of trajectory frames $T$.

    Returns
    -------
    np.ndarray
        Array of normalized diffusion time values in $[0.0, 1.0]$, shape `(T,)`.
    """
    if step_indices is not None and len(step_indices) > 0:
        max_step = (
            float(step_indices[-2])
            if (len(step_indices) > 1 and step_indices[-1] > step_indices[-2])
            else float(step_indices[-1])
        )
        if max_step <= 0.0:
            max_step = 1.0
        return np.clip(1.0 - (step_indices.astype(float) / max_step), 0.0, 1.0)
    if num_frames <= 1:
        return np.zeros(num_frames, dtype=float)
    return np.linspace(1.0, 0.0, num_frames)


def select_coalescence_frames(
    coords: np.ndarray,
    is_target: Optional[np.ndarray] = None,
    step_indices: Optional[np.ndarray] = None,
    *,
    num_timepoints: int = 5,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
) -> List[int]:
    r"""Select trajectory frames that capture structural coalescence.

    Computes RMSD of designed atoms relative to the converged final frame,
    identifies the earliest frame matching `t_start`, and selects intermediate
    coalescence frames via geometric RMSD progression or linear time decay.

    Parameters
    ----------
    coords : np.ndarray
        Trajectory coordinates, shape `(T, N, 3)`.
    is_target : np.ndarray, optional
        Boolean mask of fixed target atoms, shape `(N,)`.
    step_indices : np.ndarray, optional
        Array of diffusion step indices corresponding to frames, shape `(T,)`.
    num_timepoints : int, default 5
        Number of timepoints to select.
    t_start : float, default 0.9
        Initial diffusion time $t \in [0.0, 1.0]$ for the first frame.
    spacing_mode : str, default "rmsd"
        Spacing strategy: "rmsd" for geometric RMSD progression or
        "linear_t" for linear time decay.

    Returns
    -------
    List[int]
        Monotonically increasing frame indices spanning the coalescence phase.
    """
    total_frames = coords.shape[0]
    if total_frames <= num_timepoints:
        return list(range(total_frames))

    t_arr = compute_diffusion_times(step_indices, total_frames)

    mask = (
        ~is_target
        if (is_target is not None and np.any(~is_target))
        else np.ones(coords.shape[1], dtype=bool)
    )
    ref_coords = coords[-1][mask]
    c_ref = ref_coords - np.mean(ref_coords, axis=0, keepdims=True)

    rmsds: List[float] = []
    for i in range(total_frames):
        pts = coords[i][mask]
        c_pts = pts - np.mean(pts, axis=0, keepdims=True)
        cov = c_pts.T @ c_ref
        u, _, vh = np.linalg.svd(cov)
        rot = vh.T @ u.T
        if np.linalg.det(rot) < 0.0:
            vh[-1, :] *= -1.0
            rot = vh.T @ u.T
        pts_aligned = c_pts @ rot.T
        rmsd = float(np.sqrt(np.mean((pts_aligned - c_ref) ** 2)))
        rmsds.append(rmsd)

    rmsd_arr = np.array(rmsds)

    start_idx = int(np.argmin(np.abs(t_arr - t_start)))
    end_idx = total_frames - 1

    if spacing_mode == "linear_t":
        target_ts = np.linspace(t_arr[start_idx], 0.0, num_timepoints)
        chosen: List[int] = [start_idx]
        last_idx = start_idx
        for target_t in target_ts[1:-1]:
            remaining = range(last_idx + 1, end_idx)
            if not remaining:
                break
            best = min(remaining, key=lambda i: abs(t_arr[i] - target_t))
            chosen.append(best)
            last_idx = best
    else:
        min_target = max(float(rmsd_arr[end_idx - 1]), 2.0)
        target_levels = np.geomspace(
            max(rmsd_arr[start_idx], min_target + 0.1),
            min_target,
            num_timepoints - 1,
        )
        chosen = [start_idx]
        last_idx = start_idx
        for lvl in target_levels[1:]:
            remaining = range(last_idx + 1, end_idx)
            if not remaining:
                break
            best = min(remaining, key=lambda i: abs(rmsd_arr[i] - lvl))
            chosen.append(best)
            last_idx = best

    chosen.append(end_idx)
    chosen = sorted(dict.fromkeys(chosen))
    if len(chosen) < num_timepoints:
        for f in range(total_frames):
            if f not in chosen:
                chosen.append(f)
            if len(chosen) == num_timepoints:
                break
        chosen.sort()

    return chosen[:num_timepoints]


def prepare_binder_trajectory(
    traj_data: Dict[str, np.ndarray],
    *,
    num_timepoints: int = 5,
    frame_indices: Optional[List[int]] = None,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
    mainchain_only: bool = False,
    ca_only: bool = False,
    align_target: bool = True,
    align_vertical_right: bool = False,
    target_all_atoms: bool = True,
) -> Tuple[List[np.ndarray], List[float], np.ndarray, np.ndarray]:
    r"""Extract and align key timepoints for traditional binder design.

    Parameters
    ----------
    traj_data : Dict[str, np.ndarray]
        Loaded trajectory dictionary.
    num_timepoints : int, default 5
        Number of timepoints to extract.
    frame_indices : List[int], optional
        Explicit frame indices to extract. If None, auto-selects frames
        spanning the structural coalescence phase.
    t_start : float, default 0.9
        Initial diffusion time $t \in [0.0, 1.0]$ for the first frame.
    spacing_mode : str, default "rmsd"
        Frame spacing strategy: "rmsd" or "linear_t".
    mainchain_only : bool, default False
        Whether to render only main-chain backbone atoms (N, CA, C, O).
    ca_only : bool, default False
        Whether to render only alpha-carbon (CA) atoms.
    align_target : bool, default True
        Whether to rigidly align target atoms across all timepoints to the
        final denoised frame.
    align_vertical_right : bool, default False
        Whether to align the binder helix vertically along Z on the right (+X)
        with the target on the left (-X).
    target_all_atoms : bool, default True
        Whether to preserve all heavy atoms for target context while applying
        backbone/CA filtering only to the designed binder atoms.

    Returns
    -------
    frames : List[np.ndarray]
        List of aligned coordinates for each selected timepoint,
        shape `(N_valid, 3)`.
    t_vals : List[float]
        Diffusion time parameter $t \in [1.0, 0.0]$ corresponding to each frame.
    is_target_valid : np.ndarray
        Boolean mask of target atoms for valid atoms, shape `(N_valid,)`.
    asym_id_valid : np.ndarray
        Chain ID for valid atoms, shape `(N_valid,)`.
    """
    coords = traj_data["coords"]
    step_indices = traj_data.get("step_indices")
    total_frames = coords.shape[0]
    all_t_vals = compute_diffusion_times(step_indices, total_frames)

    valid_mask = traj_data["atom_pad_mask"].astype(bool).copy()
    is_target_raw = traj_data["is_target"].astype(bool)

    if mainchain_only or ca_only:
        atom_indices = np.arange(len(valid_mask))
        mc_filter = (
            (atom_indices % ATOM14_NUM_ATOMS == CA_ATOM_OFFSET)
            if ca_only
            else (atom_indices % ATOM14_NUM_ATOMS < BACKBONE_ATOMS_PER_RES)
        )
        if target_all_atoms:
            is_binder = ~is_target_raw
            valid_mask[is_binder] = valid_mask[is_binder] & mc_filter[is_binder]
        else:
            valid_mask = valid_mask & mc_filter

    is_target_valid = traj_data["is_target"][valid_mask]
    asym_id_valid = traj_data["asym_id"][valid_mask]

    if frame_indices is not None:
        selected_indices = frame_indices
    else:
        selected_indices = select_coalescence_frames(
            coords[:, valid_mask],
            is_target=is_target_valid,
            step_indices=step_indices,
            num_timepoints=num_timepoints,
            t_start=t_start,
            spacing_mode=spacing_mode,
        )

    t_vals = [float(all_t_vals[idx]) for idx in selected_indices]
    ref_frame_raw = coords[-1][valid_mask]
    target_mask = is_target_valid

    if align_vertical_right:
        _, rot_align, center_align = orient_binder_vertical_right(
            ref_frame_raw, target_mask
        )

    aligned_frames: List[np.ndarray] = []
    for idx in selected_indices:
        frame_raw = coords[idx][valid_mask]
        if align_target and target_mask.any():
            aligned = kabsch_rigid_align(
                source_coords=frame_raw[target_mask],
                target_coords=ref_frame_raw[target_mask],
                coords_to_transform=frame_raw,
            )
        else:
            aligned = frame_raw

        if align_vertical_right:
            aligned = (aligned - center_align) @ rot_align.T

        aligned_frames.append(aligned)

    return aligned_frames, t_vals, is_target_valid, asym_id_valid



def prepare_helical_trajectory(
    traj_data: Dict[str, np.ndarray],
    *,
    num_timepoints: int = 5,
    frame_indices: Optional[List[int]] = None,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
    mainchain_only: bool = False,
    ca_only: bool = False,
    align_to_axis: bool = True,
    axial_strain: float = 0.0,
    poisson_ratio: float = 0.35,
    deformation_mode: str = "rigid_centroid",
    return_atom_types: bool = False,
) -> Tuple[Any, ...]:
    r"""Extract and align key timepoints for symmetry-driven helical design.

    Parameters
    ----------
    traj_data : Dict[str, np.ndarray]
        Loaded trajectory dictionary.
    num_timepoints : int, default 5
        Number of timepoints to extract.
    frame_indices : List[int], optional
        Explicit frame indices to extract. If None, auto-selects frames
        spanning the structural coalescence phase.
    t_start : float, default 0.9
        Initial diffusion time $t \in [0.0, 1.0]$ for the first frame.
    spacing_mode : str, default "rmsd"
        Frame spacing strategy: "rmsd" or "linear_t".
    mainchain_only : bool, default False
        Whether to render only main-chain backbone atoms (N, CA, C, O).
    ca_only : bool, default False
        Whether to render only alpha-carbon (CA) atoms.
    align_to_axis : bool, default True
        Whether to align the principal helical axis along the vertical Z-axis
        and align all frames to the final structure.
    axial_strain : float, default 0.0
        Engineering strain applied along the principal z-axis.
    poisson_ratio : float, default 0.35
        Effective Poisson ratio governing transverse strain in rigid centroid mode.
    deformation_mode : str, default "rigid_centroid"
        Deformation method: "rigid_centroid" (preserves intra-chain helices)
        or "affine" (1D coordinate scaling).

    Returns
    -------
    frames : List[np.ndarray]
        List of aligned coordinates for each selected timepoint,
        shape `(N_valid, 3)`.
    t_vals : List[float]
        Diffusion time parameter $t \in [1.0, 0.0]$ corresponding to each frame.
    chain_indices : np.ndarray
        0-indexed integer ordering of subunits along screw axis,
        shape `(N_valid,)`.
    asym_id_valid : np.ndarray
        Original chain ID for valid atoms, shape `(N_valid,)`.
    """
    coords = traj_data["coords"]
    step_indices = traj_data.get("step_indices")
    total_frames = coords.shape[0]
    all_t_vals = compute_diffusion_times(step_indices, total_frames)

    valid_mask = traj_data["atom_pad_mask"].copy() > 0.5
    if mainchain_only or ca_only:
        atom_indices = np.arange(len(valid_mask))
        mc_filter = (
            (atom_indices % ATOM14_NUM_ATOMS == CA_ATOM_OFFSET)
            if ca_only
            else (atom_indices % ATOM14_NUM_ATOMS < BACKBONE_ATOMS_PER_RES)
        )
        valid_mask = valid_mask & mc_filter

    asym_id_valid = traj_data["asym_id"][valid_mask]

    if frame_indices is not None:
        selected_indices = frame_indices
    else:
        selected_indices = select_coalescence_frames(
            coords[:, valid_mask],
            is_target=None,
            step_indices=step_indices,
            num_timepoints=num_timepoints,
            t_start=t_start,
            spacing_mode=spacing_mode,
        )

    t_vals = [float(all_t_vals[idx]) for idx in selected_indices]
    final_raw = coords[-1][valid_mask]

    if align_to_axis:
        final_oriented, _ = orient_coords_along_z(final_raw)
        aligned_frames: List[np.ndarray] = []
        for idx in selected_indices:
            frame_raw = coords[idx][valid_mask]
            aligned = kabsch_rigid_align(
                source_coords=frame_raw,
                target_coords=final_raw,
                coords_to_transform=frame_raw,
            )
            frame_oriented = kabsch_rigid_align(
                source_coords=final_raw,
                target_coords=final_oriented,
                coords_to_transform=aligned,
            )
            aligned_frames.append(frame_oriented)
    else:
        aligned_frames = [coords[idx][valid_mask] for idx in selected_indices]
        final_oriented = aligned_frames[-1]

    if axial_strain != 0.0:
        unique_chains_strain = np.unique(asym_id_valid)
        strained_frames: List[np.ndarray] = []
        if deformation_mode == "rigid_centroid" and len(unique_chains_strain) > 1:
            ref_centroids = np.array([
                np.mean(final_oriented[asym_id_valid == cid], axis=0)
                for cid in unique_chains_strain
            ], dtype=np.float64)
            assembly_center = np.mean(ref_centroids, axis=0)

            chain_deltas = {}
            for cid, c_k in zip(unique_chains_strain, ref_centroids):
                delta = np.zeros(3, dtype=np.float64)
                delta[0] = -(c_k[0] - assembly_center[0]) * poisson_ratio * axial_strain
                delta[1] = -(c_k[1] - assembly_center[1]) * poisson_ratio * axial_strain
                delta[2] = (c_k[2] - assembly_center[2]) * axial_strain
                chain_deltas[cid] = delta

            for frame in aligned_frames:
                f = frame.copy()
                for cid, delta in chain_deltas.items():
                    f[asym_id_valid == cid] += delta
                strained_frames.append(f)
        else:
            z_ref = float(np.mean(final_oriented[:, 2]))
            for frame in aligned_frames:
                f = frame.copy()
                f[:, 2] = z_ref + (f[:, 2] - z_ref) * (1.0 + axial_strain)
                strained_frames.append(f)
        aligned_frames = strained_frames
        final_oriented = aligned_frames[-1]

    unique_chains = np.unique(asym_id_valid)
    chain_mean_z = {
        cid: float(np.mean(final_oriented[asym_id_valid == cid, 2]))
        for cid in unique_chains
    }
    sorted_chains = sorted(unique_chains, key=lambda cid: chain_mean_z[cid])
    chain_to_order = {cid: idx for idx, cid in enumerate(sorted_chains)}
    chain_indices = np.array(
        [chain_to_order[cid] for cid in asym_id_valid], dtype=int
    )

    if return_atom_types:
        raw_atom_indices = np.arange(len(traj_data["atom_pad_mask"]))[valid_mask]
        mod14 = raw_atom_indices % ATOM14_NUM_ATOMS
        atom_types = {
            "is_ca": (mod14 == CA_ATOM_OFFSET),
            "is_backbone": (mod14 < BACKBONE_ATOMS_PER_RES),
            "is_sidechain": (mod14 >= BACKBONE_ATOMS_PER_RES),
        }
        return aligned_frames, t_vals, chain_indices, asym_id_valid, atom_types

    return aligned_frames, t_vals, chain_indices, asym_id_valid


def _style_3d_axes(
    ax: Axes3D,
    center: np.ndarray,
    span: float,
    elev: float,
    azim: float,
    dist: float = 7.0,
) -> None:
    """Apply uniform bounding box, camera zoom, and clean styling to 3D axes."""
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("whitesmoke")
    ax.yaxis.pane.set_edgecolor("whitesmoke")
    ax.zaxis.pane.set_edgecolor("whitesmoke")
    ax.grid(visible=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _plot_binder_row(
    fig: plt.Figure,
    binder_traj: Dict[str, np.ndarray],
    *,
    row_idx: int,
    total_rows: int,
    num_timepoints: int,
    frame_indices: Optional[List[int]] = None,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
    mainchain_only: bool = False,
    ca_only: bool = False,
    elev: float,
    azim: float,
    dist: float = 7.0,
    target_color: str,
    target_alpha: float = 0.35,
    static_target: bool = True,
    target_all_atoms: bool = True,
    binder_color: str,
    point_size_target: float,
    point_size_binder: float,
    box_scale: float = 1.0,
    show_row_label: bool = True,
    orient_vertical_right: bool = False,
) -> None:
    """Plot traditional binder design point-cloud trajectory row."""
    b_frames, b_t_vals, b_is_target, _ = prepare_binder_trajectory(
        binder_traj,
        num_timepoints=num_timepoints,
        frame_indices=frame_indices,
        t_start=t_start,
        spacing_mode=spacing_mode,
        mainchain_only=mainchain_only,
        ca_only=ca_only,
        align_target=True,
        align_vertical_right=orient_vertical_right,
        target_all_atoms=target_all_atoms,
    )

    ref_frame = b_frames[-1]
    ref_target_pts = ref_frame[b_is_target]
    b_min = np.min(ref_frame, axis=0)
    b_max = np.max(ref_frame, axis=0)
    b_center = (b_min + b_max) / 2.0
    b_span = max(float(np.max(b_max - b_min) * 0.55 * box_scale), 1.0)

    for col_idx in range(len(b_frames)):
        ax = fig.add_subplot(
            total_rows,
            len(b_frames),
            row_idx * len(b_frames) + col_idx + 1,
            projection="3d",
        )
        ax.set_facecolor("white")

        frame_pts = b_frames[col_idx]
        target_pts = ref_target_pts if static_target else frame_pts[b_is_target]
        binder_pts = frame_pts[~b_is_target]

        if len(target_pts) > 0:
            ax.scatter(
                target_pts[:, 0],
                target_pts[:, 1],
                target_pts[:, 2],
                c=target_color,
                s=point_size_target,
                alpha=target_alpha,
                edgecolors="none",
                marker="o",
                label="Target Context",
                rasterized=True,
            )

        if len(binder_pts) > 0:
            edge_c = "black" if (mainchain_only or ca_only) else "none"
            lw = 0.25 if (mainchain_only or ca_only) else 0.0
            ax.scatter(
                binder_pts[:, 0],
                binder_pts[:, 1],
                binder_pts[:, 2],
                c=binder_color,
                s=point_size_binder,
                alpha=0.85,
                edgecolors=edge_c,
                linewidths=lw,
                label="Binder Subunit",
                rasterized=True,
            )

        t_val = b_t_vals[col_idx]
        if col_idx == 0:
            col_title = f"$t = {t_val:.2f}$" + "\n(Dispersed Cloud)"
        elif col_idx == len(b_frames) - 1:
            col_title = f"$t = {t_val:.2f}$" + "\n(Coalesced Binder)"
        else:
            col_title = f"$t = {t_val:.2f}$"
        ax.set_title(col_title, fontsize=11, pad=4, weight="bold")

        _style_3d_axes(ax, b_center, b_span, elev, azim, dist=dist)

        if col_idx == 0:
            if show_row_label:
                ax.text2D(
                    -0.18,
                    0.5,
                    "Traditional Binder Design\n(Fixed Target Context)",
                    transform=ax.transAxes,
                    fontsize=11,
                    weight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                    color="#1E293B",
                )
            legend = ax.legend(
                loc="upper left",
                bbox_to_anchor=(-0.05, 1.05),
                frameon=True,
                facecolor="white",
                edgecolor="lightgray",
                fontsize=9,
            )
            for handle in legend.legend_handles:
                handle.set_sizes([30])


def get_discrete_colormap(
    cmap_name: str,
    num_colors: int,
) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    """Create a discrete ListedColormap and BoundaryNorm for qualitative styling."""
    effective_cmap_name = cmap_name
    if effective_cmap_name == "tab10" and num_colors > TAB10_MAX_COLORS:
        effective_cmap_name = "tab20"

    base_cmap = colormaps[effective_cmap_name]
    if hasattr(base_cmap, "colors") and len(base_cmap.colors) >= num_colors:
        colors_list = [
            base_cmap.colors[i % len(base_cmap.colors)]
            for i in range(num_colors)
        ]
    elif hasattr(base_cmap, "resampled"):
        resampled = base_cmap.resampled(num_colors)
        colors_list = [resampled(i) for i in range(num_colors)]
    else:
        colors_list = [
            base_cmap(i / max(1, num_colors - 1))
            for i in range(num_colors)
        ]

    discrete_cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(
        boundaries=np.arange(num_colors + 1) - 0.5,
        ncolors=num_colors,
    )
    return discrete_cmap, norm


def _plot_helical_row(
    fig: plt.Figure,
    helical_traj: Dict[str, np.ndarray],
    *,
    row_idx: int,
    total_rows: int,
    num_timepoints: int,
    frame_indices: Optional[List[int]] = None,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
    mainchain_only: bool = False,
    ca_only: bool = False,
    elev: float,
    azim: float,
    dist: float = 7.0,
    cmap_name: str,
    point_size: float,
    show_titles: bool,
    box_scale: float = 1.0,
    show_row_label: bool = True,
    show_colorbar: bool = True,
    axial_strain: float = 0.0,
    poisson_ratio: float = 0.35,
    deformation_mode: str = "rigid_centroid",
    sidechain_alpha: float = 0.50,
    row_title: Optional[str] = None,
    final_frame_subtitle: Optional[str] = None,
    fixed_center: Optional[np.ndarray] = None,
    fixed_span: Optional[float] = None,
) -> None:
    """Plot symmetry-driven helical design point-cloud trajectory row."""
    h_frames, h_t_vals, h_chains, _, atom_types = prepare_helical_trajectory(
        helical_traj,
        num_timepoints=num_timepoints,
        frame_indices=frame_indices,
        t_start=t_start,
        spacing_mode=spacing_mode,
        mainchain_only=mainchain_only,
        ca_only=ca_only,
        align_to_axis=True,
        axial_strain=axial_strain,
        poisson_ratio=poisson_ratio,
        deformation_mode=deformation_mode,
        return_atom_types=True,
    )

    if fixed_center is not None and fixed_span is not None:
        h_center = fixed_center
        h_span = fixed_span
    else:
        ref_frame = h_frames[-1]
        h_min = np.min(ref_frame, axis=0)
        h_max = np.max(ref_frame, axis=0)
        h_center = (h_min + h_max) / 2.0
        h_span = max(float(np.max(h_max - h_min) * 0.55 * box_scale), 1.0)

    num_chains = len(np.unique(h_chains))
    cmap, norm = get_discrete_colormap(cmap_name, num_chains)
    chain_colors = cmap(norm(h_chains))

    for col_idx in range(len(h_frames)):
        ax = fig.add_subplot(
            total_rows,
            len(h_frames),
            row_idx * len(h_frames) + col_idx + 1,
            projection="3d",
        )
        ax.set_facecolor("white")

        frame_pts = h_frames[col_idx]
        if (mainchain_only or ca_only) or sidechain_alpha >= 1.0 or atom_types is None:
            edge_c = "black" if (mainchain_only or ca_only) else "none"
            lw = 0.25 if (mainchain_only or ca_only) else 0.0
            ax.scatter(
                frame_pts[:, 0],
                frame_pts[:, 1],
                frame_pts[:, 2],
                c=chain_colors,
                s=point_size,
                alpha=0.85,
                edgecolors=edge_c,
                linewidths=lw,
                rasterized=True,
            )
        else:
            is_sc = atom_types["is_sidechain"]
            is_ca = atom_types["is_ca"]
            is_other_bb = atom_types["is_backbone"] & (~is_ca)

            # 1. Side chains with ~50% transparency
            if np.any(is_sc):
                ax.scatter(
                    frame_pts[is_sc, 0],
                    frame_pts[is_sc, 1],
                    frame_pts[is_sc, 2],
                    c=chain_colors[is_sc],
                    s=point_size * 0.75,
                    alpha=sidechain_alpha,
                    edgecolors="none",
                    rasterized=True,
                )

            # 2. Non-CA backbone atoms (N, C, O) with high opacity
            if np.any(is_other_bb):
                ax.scatter(
                    frame_pts[is_other_bb, 0],
                    frame_pts[is_other_bb, 1],
                    frame_pts[is_other_bb, 2],
                    c=chain_colors[is_other_bb],
                    s=point_size * 0.9,
                    alpha=0.90,
                    edgecolors="none",
                    rasterized=True,
                )

            # 3. CA backbone atoms with solid opacity and crisp outline
            if np.any(is_ca):
                ax.scatter(
                    frame_pts[is_ca, 0],
                    frame_pts[is_ca, 1],
                    frame_pts[is_ca, 2],
                    c=chain_colors[is_ca],
                    s=point_size * 1.35,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.35,
                    rasterized=True,
                )

        if show_titles:
            t_val = h_t_vals[col_idx]
            if final_frame_subtitle and col_idx == len(h_frames) - 1:
                col_title = f"$t = {t_val:.2f}$\n({final_frame_subtitle})"
            else:
                col_title = f"$t = {t_val:.2f}$"
            ax.set_title(col_title, fontsize=11, pad=6, weight="bold")

        _style_3d_axes(ax, h_center, h_span, elev, azim, dist=dist)

        if col_idx == 0 and show_row_label:
            if row_title:
                effective_title = row_title
            elif axial_strain != 0.0:
                pct = abs(int(round(axial_strain * 100)))
                mode = "Tension" if axial_strain > 0 else "Compression"
                effective_title = f"{pct}% {mode}"
            else:
                effective_title = "Unstrained"
            ax.text2D(
                -0.24,
                0.5,
                effective_title,
                transform=ax.transAxes,
                fontsize=11,
                weight="bold",
                va="center",
                ha="center",
                rotation=90,
                color="#1E293B",
            )

    if show_colorbar:
        cbar_height = 0.65 if total_rows == 1 else 0.32
        cbar_bottom = 0.16 if total_rows == 1 else 0.12
        cbar_ax = fig.add_axes([0.915, cbar_bottom, 0.012, cbar_height])
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm,
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Subunit Index Along Screw Axis", fontsize=9, weight="bold")
        cbar.set_ticks(np.arange(num_chains))
        cbar.set_ticklabels([str(i + 1) for i in range(num_chains)])


def plot_point_cloud_comparison(
    binder_traj: Optional[Dict[str, np.ndarray]],
    helical_traj: Optional[Dict[str, np.ndarray]],
    output_path: Path,
    *,
    strained_helical_traj: Optional[Dict[str, np.ndarray]] = None,
    axial_strain: float = 0.0,
    show_strain_comparison: bool = False,
    tri_state_deformation: bool = False,
    compression_strain: Optional[float] = None,
    tension_strain: Optional[float] = None,
    lock_deformation_bounds: bool = True,
    helical_row_title: Optional[str] = None,
    strained_row_title: Optional[str] = None,
    compression_row_title: Optional[str] = None,
    tension_row_title: Optional[str] = None,
    num_timepoints: int = 5,
    frame_indices: Optional[List[int]] = None,
    t_start: float = 0.9,
    spacing_mode: str = "rmsd",
    mainchain_only: bool = False,
    ca_only: bool = False,
    poisson_ratio: float = 0.35,
    deformation_mode: str = "rigid_centroid",
    sidechain_alpha: float = 0.50,
    elev: float = 20.0,
    azim: float = 35.0,
    camera_dist: float = 7.0,
    binder_elev: Optional[float] = None,
    binder_azim: Optional[float] = None,
    helical_elev: Optional[float] = None,
    helical_azim: Optional[float] = None,
    point_size_target: float = 8.0,
    point_size_binder: float = 22.0,
    point_size_helical: float = 16.0,
    target_color: str = "#94A3B8",
    target_alpha: float = 0.35,
    static_target: bool = True,
    target_all_atoms: bool = True,
    orient_vertical_right: bool = False,
    binder_color: str = "#FF6B6B",
    helical_cmap_name: str = "tab10",
    box_scale: float = 1.0,
    show_row_labels: bool = True,
    show_colorbar: bool = True,
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """Render comparison figure of diffusion point-cloud trajectories."""
    has_binder = binder_traj is not None
    has_helical = helical_traj is not None
    has_tri_state = (
        (tri_state_deformation or (compression_strain is not None and tension_strain is not None))
        and has_helical
    )
    has_strained = (
        ((strained_helical_traj is not None) or (show_strain_comparison and has_helical))
        and not has_tri_state
    )

    n_helical_rows = (
        3 if has_tri_state else ((1 if has_helical else 0) + (1 if has_strained else 0))
    )
    n_rows = (1 if has_binder else 0) + n_helical_rows

    if n_rows == 0:
        msg = "At least one trajectory (binder, helical, or strained_helical) must be provided."
        raise ValueError(msg)

    out_path = Path(output_path)

    if figsize is None:
        width = 3.0 * num_timepoints + (0.8 if show_colorbar else 0.2)
        if n_rows == 1:
            height = 3.6
        elif n_rows == 2:
            height = 6.8
        elif n_rows == 3:
            height = 9.8
        else:
            height = 3.2 * n_rows
        eff_figsize = (width, height)
    else:
        eff_figsize = figsize

    fig = plt.figure(figsize=eff_figsize, dpi=dpi, facecolor="white")
    current_row = 0

    shared_center: Optional[np.ndarray] = None
    shared_span: Optional[float] = None
    if lock_deformation_bounds and has_helical and (has_tri_state or has_strained or axial_strain != 0.0):
        h_frames_ref, _, _, _ = prepare_helical_trajectory(
            helical_traj,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            align_to_axis=True,
            axial_strain=0.0,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
        )
        final_ref = h_frames_ref[-1]
        h_min = np.min(final_ref, axis=0)
        h_max = np.max(final_ref, axis=0)
        shared_center = (h_min + h_max) / 2.0

        if has_tri_state:
            eff_ten_m = tension_strain if tension_strain is not None else (axial_strain or 0.15)
            eff_comp_m = abs(compression_strain) if compression_strain is not None else (axial_strain or 0.15)
            max_strain_m = max(eff_ten_m, eff_comp_m, 0.0)
        elif has_strained:
            max_strain_m = abs(axial_strain) if axial_strain != 0.0 else 0.20
        else:
            max_strain_m = abs(axial_strain)
        dilation = 1.0 + max_strain_m
        shared_span = max(float(np.max(h_max - h_min) * 0.55 * dilation * box_scale), 1.0)

    if has_binder and binder_traj is not None:
        if orient_vertical_right:
            b_elev = binder_elev if binder_elev is not None else 5.0
            b_azim = binder_azim if binder_azim is not None else -75.0
        else:
            b_elev = binder_elev if binder_elev is not None else elev
            b_azim = binder_azim if binder_azim is not None else azim
        _plot_binder_row(
            fig,
            binder_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=b_elev,
            azim=b_azim,
            dist=camera_dist,
            target_color=target_color,
            target_alpha=target_alpha,
            static_target=static_target,
            target_all_atoms=target_all_atoms,
            binder_color=binder_color,
            point_size_target=point_size_target,
            point_size_binder=point_size_binder,
            box_scale=box_scale,
            show_row_label=show_row_labels,
            orient_vertical_right=orient_vertical_right,
        )
        current_row += 1

    if has_tri_state and helical_traj is not None:
        h_elev = helical_elev if helical_elev is not None else elev
        h_azim = helical_azim if helical_azim is not None else azim

        eff_ten = tension_strain if tension_strain is not None else (axial_strain or 0.15)
        raw_comp = compression_strain if compression_strain is not None else (axial_strain or 0.15)
        eff_comp = -abs(raw_comp)
        comp_pct = abs(int(round(eff_comp * 100)))
        ten_pct = abs(int(round(eff_ten * 100)))

        # Row 1: Equilibrium / Relaxed
        rel_title = helical_row_title or "Unstrained"
        _plot_helical_row(
            fig,
            helical_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=h_elev,
            azim=h_azim,
            dist=camera_dist,
            cmap_name=helical_cmap_name,
            point_size=point_size_helical,
            show_titles=(current_row == 0),
            box_scale=box_scale,
            show_row_label=show_row_labels,
            show_colorbar=show_colorbar,
            axial_strain=0.0,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
            sidechain_alpha=sidechain_alpha,
            row_title=rel_title,
            final_frame_subtitle=None,
            fixed_center=shared_center,
            fixed_span=shared_span,
        )
        current_row += 1

        # Row 2: Compressive Strain
        comp_title = compression_row_title or f"{comp_pct}% Compression"
        _plot_helical_row(
            fig,
            helical_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=h_elev,
            azim=h_azim,
            dist=camera_dist,
            cmap_name=helical_cmap_name,
            point_size=point_size_helical,
            show_titles=(current_row == 0),
            box_scale=box_scale,
            show_row_label=show_row_labels,
            show_colorbar=False,
            axial_strain=eff_comp,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
            sidechain_alpha=sidechain_alpha,
            row_title=comp_title,
            final_frame_subtitle=None,
            fixed_center=shared_center,
            fixed_span=shared_span,
        )
        current_row += 1

        # Row 3: Tensile Strain
        ten_title = tension_row_title or f"{ten_pct}% Tension"
        _plot_helical_row(
            fig,
            helical_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=h_elev,
            azim=h_azim,
            dist=camera_dist,
            cmap_name=helical_cmap_name,
            point_size=point_size_helical,
            show_titles=(current_row == 0),
            box_scale=box_scale,
            show_row_label=show_row_labels,
            show_colorbar=False,
            axial_strain=eff_ten,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
            sidechain_alpha=sidechain_alpha,
            row_title=ten_title,
            final_frame_subtitle=None,
            fixed_center=shared_center,
            fixed_span=shared_span,
        )
        current_row += 1

    elif has_helical and helical_traj is not None:
        h_elev = helical_elev if helical_elev is not None else elev
        h_azim = helical_azim if helical_azim is not None else azim
        # If comparison is requested, the primary helical row is relaxed (strain = 0.0)
        eff_helical_strain = 0.0 if has_strained else axial_strain
        eff_row_title = (
            helical_row_title
            or (
                "Symmetry-Driven Helical Design\n(Equilibrium / Relaxed)"
                if has_strained
                else None
            )
        )
        _plot_helical_row(
            fig,
            helical_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=h_elev,
            azim=h_azim,
            dist=camera_dist,
            cmap_name=helical_cmap_name,
            point_size=point_size_helical,
            show_titles=(current_row == 0),
            box_scale=box_scale,
            show_row_label=show_row_labels,
            show_colorbar=show_colorbar,
            axial_strain=eff_helical_strain,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
            sidechain_alpha=sidechain_alpha,
            row_title=eff_row_title,
            fixed_center=shared_center,
            fixed_span=shared_span,
        )
        current_row += 1

    if has_strained:
        h_elev = helical_elev if helical_elev is not None else elev
        h_azim = helical_azim if helical_azim is not None else azim

        source_strained_traj = (
            strained_helical_traj
            if strained_helical_traj is not None
            else helical_traj
        )
        strain_to_apply = (
            0.0 if strained_helical_traj is not None else (axial_strain or 0.20)
        )
        sign_str = (
            f"ε = {axial_strain:+.2f}"
            if axial_strain != 0.0
            else "Simulated Lattice Strain"
        )
        eff_strained_title = (
            strained_row_title
            or f"Symmetry-Driven Helical Design\n(Axially Strained, {sign_str})"
        )

        _plot_helical_row(
            fig,
            source_strained_traj,
            row_idx=current_row,
            total_rows=n_rows,
            num_timepoints=num_timepoints,
            frame_indices=frame_indices,
            t_start=t_start,
            spacing_mode=spacing_mode,
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            elev=h_elev,
            azim=h_azim,
            dist=camera_dist,
            cmap_name=helical_cmap_name,
            point_size=point_size_helical,
            show_titles=(current_row == 0),
            box_scale=box_scale,
            show_row_label=show_row_labels,
            show_colorbar=False,  # Avoid duplicate colorbar
            axial_strain=strain_to_apply,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
            sidechain_alpha=sidechain_alpha,
            row_title=eff_strained_title,
            final_frame_subtitle=f"Strained Helix, {sign_str}",
            fixed_center=shared_center,
            fixed_span=shared_span,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    left_margin = 0.08 if show_row_labels else 0.02
    right_margin = 0.90 if show_colorbar else 0.98
    plt.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=0.90,
        bottom=0.06,
        wspace=0.02,
        hspace=0.14,
    )
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure successfully written to {out_path}")


def plot_multi_design_piezo_comparison(
    helical_trajs: List[Dict[str, np.ndarray]],
    output_path: Path,
    *,
    design_labels: Optional[List[str]] = None,
    axial_strain: float = 0.15,
    mainchain_only: bool = False,
    ca_only: bool = False,
    poisson_ratio: float = 0.35,
    deformation_mode: str = "rigid_centroid",
    sidechain_alpha: float = 0.50,
    elev: float = 20.0,
    azim: float = 35.0,
    camera_dist: float = 7.0,
    point_size: float = 20.0,
    helical_cmap_name: str = "tab10",
    box_scale: float = 1.0,
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """Render multi-panel comparative figure comparing multiple helical design options under piezo deformation.

    Each row corresponds to an independent design candidate.
    Columns correspond to the tri-state piezoelectric deformation counter-screen:
    - Column 1: Equilibrium / Relaxed State (eps = 0.00)
    - Column 2: Compressive Strain Screen (eps = -axial_strain)
    - Column 3: Tensile Strain Screen (eps = +axial_strain)
    """
    num_designs = len(helical_trajs)
    if num_designs == 0:
        msg = "At least one helical trajectory must be provided."
        raise ValueError(msg)

    strains = [0.0, -abs(axial_strain), abs(axial_strain)]
    strain_pct = abs(int(round(axial_strain * 100)))
    col_subtitles = [
        "Unstrained",
        f"{strain_pct}% Compression",
        f"{strain_pct}% Tension",
    ]

    out_path = Path(output_path)
    eff_figsize = figsize if figsize is not None else (11.0, 3.4 * num_designs)
    fig = plt.figure(figsize=eff_figsize, dpi=dpi, facecolor="white")

    for row_idx, traj in enumerate(helical_trajs):
        d_label = (
            design_labels[row_idx]
            if (design_labels and row_idx < len(design_labels))
            else f"Design Option {row_idx + 1}"
        )

        # Prepare un-strained final frame to get fixed center and span
        h_frames_ref, _, h_chains, _ = prepare_helical_trajectory(
            traj,
            num_timepoints=2,
            frame_indices=[-1],
            mainchain_only=mainchain_only,
            ca_only=ca_only,
            align_to_axis=True,
            axial_strain=0.0,
            poisson_ratio=poisson_ratio,
            deformation_mode=deformation_mode,
        )
        final_ref = h_frames_ref[-1]
        h_min = np.min(final_ref, axis=0)
        h_max = np.max(final_ref, axis=0)
        h_center = (h_min + h_max) / 2.0
        dilation = 1.0 + abs(axial_strain)
        h_span = max(float(np.max(h_max - h_min) * 0.55 * dilation * box_scale), 1.0)

        num_chains = len(np.unique(h_chains))
        cmap, norm = get_discrete_colormap(helical_cmap_name, num_chains)
        chain_colors = cmap(norm(h_chains))

        for col_idx, strain_val in enumerate(strains):
            ax = fig.add_subplot(
                num_designs,
                len(strains),
                row_idx * len(strains) + col_idx + 1,
                projection="3d",
            )
            ax.set_facecolor("white")

            h_frames, _, _, _, atom_types = prepare_helical_trajectory(
                traj,
                num_timepoints=2,
                frame_indices=[-1],
                mainchain_only=mainchain_only,
                ca_only=ca_only,
                align_to_axis=True,
                axial_strain=strain_val,
                poisson_ratio=poisson_ratio,
                deformation_mode=deformation_mode,
                return_atom_types=True,
            )
            frame_pts = h_frames[-1]

            if (mainchain_only or ca_only) or sidechain_alpha >= 1.0 or atom_types is None:
                edge_c = "black" if (mainchain_only or ca_only) else "none"
                lw = 0.25 if (mainchain_only or ca_only) else 0.0
                ax.scatter(
                    frame_pts[:, 0],
                    frame_pts[:, 1],
                    frame_pts[:, 2],
                    c=chain_colors,
                    s=point_size,
                    alpha=0.85,
                    edgecolors=edge_c,
                    linewidths=lw,
                    rasterized=True,
                )
            else:
                is_sc = atom_types["is_sidechain"]
                is_ca = atom_types["is_ca"]
                is_other_bb = atom_types["is_backbone"] & (~is_ca)

                # 1. Side chains with ~50% transparency
                if np.any(is_sc):
                    ax.scatter(
                        frame_pts[is_sc, 0],
                        frame_pts[is_sc, 1],
                        frame_pts[is_sc, 2],
                        c=chain_colors[is_sc],
                        s=point_size * 0.75,
                        alpha=sidechain_alpha,
                        edgecolors="none",
                        rasterized=True,
                    )

                # 2. Non-CA backbone atoms (N, C, O) with high opacity
                if np.any(is_other_bb):
                    ax.scatter(
                        frame_pts[is_other_bb, 0],
                        frame_pts[is_other_bb, 1],
                        frame_pts[is_other_bb, 2],
                        c=chain_colors[is_other_bb],
                        s=point_size * 0.9,
                        alpha=0.90,
                        edgecolors="none",
                        rasterized=True,
                    )

                # 3. CA backbone atoms with solid opacity and crisp outline
                if np.any(is_ca):
                    ax.scatter(
                        frame_pts[is_ca, 0],
                        frame_pts[is_ca, 1],
                        frame_pts[is_ca, 2],
                        c=chain_colors[is_ca],
                        s=point_size * 1.35,
                        alpha=1.0,
                        edgecolors="black",
                        linewidths=0.35,
                        rasterized=True,
                    )

            if row_idx == 0:
                ax.set_title(col_subtitles[col_idx], fontsize=12, pad=6, weight="bold")

            _style_3d_axes(ax, h_center, h_span, elev, azim, dist=camera_dist)

            if col_idx == 0:
                ax.text2D(
                    -0.24,
                    0.5,
                    d_label,
                    transform=ax.transAxes,
                    fontsize=11,
                    weight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                    color="#1E293B",
                )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Subunit Index Along Screw Axis", fontsize=9, weight="bold")
    cbar.set_ticks(np.arange(num_chains))
    cbar.set_ticklabels([str(i + 1) for i in range(num_chains)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(
        left=0.10,
        right=0.90,
        top=0.90,
        bottom=0.06,
        wspace=0.04,
        hspace=0.15,
    )
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Multi-design comparison figure successfully written to {out_path}")


def main() -> None:
    """CLI entrypoint for diffusion trajectory figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparison figure of diffusion trajectories."
    )
    parser.add_argument(
        "--binder",
        type=str,
        default=None,
        help="Path or glob pattern to binder trajectory archive",
    )
    parser.add_argument(
        "--helical",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) or glob pattern to helical trajectory archive(s)",
    )
    parser.add_argument(
        "--strained_helical",
        type=str,
        default=None,
        help="Path or glob pattern to strained helical trajectory archive",
    )
    parser.add_argument(
        "--axial_strain",
        type=float,
        default=0.0,
        help="Axial strain (dL/L) applied along the vertical Z axis for visualization or simulation (default: 0.0, e.g. 0.20)",
    )
    parser.add_argument(
        "--show_strain_comparison",
        action="store_true",
        default=False,
        help="Show a separate strained helical row in addition to the relaxed helical row",
    )
    parser.add_argument(
        "--tri_state_deformation",
        "--piezo_deformation",
        action="store_true",
        default=False,
        help="Show tri-state piezoelectric screening deformation (Equilibrium, Compressive Strain, Tensile Strain)",
    )
    parser.add_argument(
        "--multi_design_comparison",
        action="store_true",
        default=False,
        help="Compare multiple helical design candidates under tri-state deformation",
    )
    parser.add_argument(
        "--design_labels",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for each design candidate when comparing multiple designs",
    )
    parser.add_argument(
        "--compression_strain",
        type=float,
        default=None,
        help="Compressive strain magnitude (e.g. -0.15 or 0.15; default: negative of axial_strain or -0.15)",
    )
    parser.add_argument(
        "--tension_strain",
        type=float,
        default=None,
        help="Tensile strain magnitude (e.g. 0.15; default: positive axial_strain or +0.15)",
    )
    parser.add_argument(
        "--lock_deformation_bounds",
        action="store_true",
        default=True,
        help="Lock 3D coordinate bounding box across relaxed, compressed, and tensioned rows (default: True)",
    )
    parser.add_argument(
        "--no_lock_deformation_bounds",
        action="store_false",
        dest="lock_deformation_bounds",
        help="Disable shared bounding box across deformation rows",
    )
    parser.add_argument(
        "--poisson_ratio",
        type=float,
        default=0.35,
        help="Poisson's ratio for transverse strain under axial deformation (default: 0.35)",
    )
    parser.add_argument(
        "--deformation_mode",
        type=str,
        default="rigid_centroid",
        choices=["rigid_centroid", "affine"],
        help="Piezo deformation mode: 'rigid_centroid' (preserves helices) or 'affine' (1D scaling; default: rigid_centroid)",
    )
    parser.add_argument(
        "--sidechain_alpha",
        type=float,
        default=0.50,
        help="Alpha transparency for sidechain atoms in all-atom views (default: 0.50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/diffusion_points_comparison.png"),
        help="Destination path for output figure",
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit trajectory frame indices (e.g. --frame_indices 4 6 7 12 21)",
    )
    parser.add_argument(
        "--t_start",
        type=float,
        default=0.9,
        help="Initial diffusion time parameter in [0.0, 1.0] for the first frame (default: 0.9)",
    )
    parser.add_argument(
        "--spacing_mode",
        type=str,
        choices=["rmsd", "linear_t"],
        default="rmsd",
        help="Intermediate frame selection strategy: 'rmsd' (geometric progression) or 'linear_t'",
    )
    parser.add_argument(
        "--camera_dist",
        type=float,
        default=7.0,
        help="3D camera distance zoom factor (smaller = zoomed in, default: 7.0)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Explicit figure size in inches (default: dynamically computed from row/column count)",
    )
    parser.add_argument(
        "--hide_row_labels",
        action="store_true",
        default=False,
        help="Hide vertical row description labels on the left margin",
    )
    parser.add_argument(
        "--no_colorbar",
        action="store_true",
        default=False,
        help="Suppress colorbar for helical subunit chains",
    )
    parser.add_argument(
        "--use_x0",
        action="store_true",
        default=False,
        help="Plot denoised coordinates (x_0) instead of noisy point clouds (x_t)",
    )
    parser.add_argument(
        "--mainchain_only",
        "--backbone_only",
        action="store_true",
        dest="mainchain_only",
        help="Render only main-chain backbone atoms (N, CA, C, O)",
    )
    parser.add_argument(
        "--ca_only",
        action="store_true",
        help="Render only alpha-carbon (CA) atoms",
    )
    parser.add_argument(
        "--helical_cmap",
        type=str,
        default="tab10",
        dest="helical_cmap_name",
        help="Colormap for helical subunits (e.g. 'tab10', 'tab20', 'Paired', 'Set1')",
    )
    parser.add_argument(
        "--box_scale",
        type=float,
        default=1.0,
        help="Viewport scaling multiplier relative to the final assembled structure (default: 1.0)",
    )
    parser.add_argument(
        "--num_timepoints",
        type=int,
        default=5,
        help="Number of diffusion timepoints to plot across each row",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="3D camera elevation angle in degrees",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=35.0,
        help="3D camera azimuth angle in degrees",
    )
    parser.add_argument(
        "--binder_elev",
        type=float,
        default=None,
        help="Elevation angle override for traditional binder row",
    )
    parser.add_argument(
        "--binder_azim",
        type=float,
        default=None,
        help="Azimuth angle override for traditional binder row",
    )
    parser.add_argument(
        "--helical_elev",
        type=float,
        default=None,
        help="Elevation angle override for helical row",
    )
    parser.add_argument(
        "--helical_azim",
        type=float,
        default=None,
        help="Azimuth angle override for helical row",
    )
    parser.add_argument(
        "--point_size_target",
        type=float,
        default=16.0,
        help="Point size for target atoms (default: 16.0)",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="#94A3B8",
        help="Color for target context points (default: #94A3B8)",
    )
    parser.add_argument(
        "--target_alpha",
        type=float,
        default=0.35,
        help="Transparency alpha for target atoms (default: 0.35)",
    )
    parser.add_argument(
        "--orient_vertical_right",
        "--orient_helix_right",
        action="store_true",
        default=False,
        dest="orient_vertical_right",
        help="Orient complex with vertical binder helix on the right",
    )
    parser.add_argument(
        "--static_target",
        "--non_diffused_target",
        action="store_true",
        default=True,
        dest="static_target",
        help="Render fixed, non-diffused target across all frames (default: True)",
    )
    parser.add_argument(
        "--no_static_target",
        action="store_false",
        dest="static_target",
        help="Allow target atoms to drift/diffuse across timepoints",
    )
    parser.add_argument(
        "--target_all_atoms",
        action="store_true",
        default=True,
        dest="target_all_atoms",
        help="Keep all heavy atoms for target context (default: True)",
    )
    parser.add_argument(
        "--no_target_all_atoms",
        action="store_false",
        dest="target_all_atoms",
        help="Filter target atoms using the same modular filter as the binder",
    )
    parser.add_argument(
        "--point_size_binder",
        type=float,
        default=22.0,
        help="Point size for binder atoms",
    )
    parser.add_argument(
        "--point_size_helical",
        type=float,
        default=16.0,
        help="Point size for helical subunit atoms",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in DPI",
    )

    args = parser.parse_args()

    if (
        args.binder is None
        and args.helical is None
        and args.strained_helical is None
    ):
        parser.error(
            "At least one of --binder, --helical, or --strained_helical must be specified."
        )

    use_x0 = args.use_x0

    custom_figsize = tuple(args.figsize) if args.figsize is not None else None

    binder_traj = None
    if args.binder is not None:
        binder_file = resolve_filepath(args.binder)
        print(f"Loading binder trajectory: {binder_file}")
        binder_traj = load_trajectory_data(binder_file, use_x0=use_x0)

    helical_traj = None
    if args.helical is not None:
        helical_files = []
        for h_spec in args.helical:
            try:
                helical_files.append(resolve_filepath(h_spec))
            except Exception:
                helical_files.append(Path(h_spec))

        if args.multi_design_comparison or len(helical_files) > 1:
            print(
                f"Loading {len(helical_files)} helical trajectories for multi-design comparison..."
            )
            loaded_trajs = [
                load_trajectory_data(f, use_x0=use_x0) for f in helical_files
            ]
            plot_multi_design_piezo_comparison(
                helical_trajs=loaded_trajs,
                output_path=args.output,
                design_labels=args.design_labels,
                axial_strain=args.axial_strain or 0.15,
                mainchain_only=args.mainchain_only,
                ca_only=args.ca_only,
                poisson_ratio=args.poisson_ratio,
                deformation_mode=args.deformation_mode,
                sidechain_alpha=args.sidechain_alpha,
                elev=args.elev,
                azim=args.azim,
                camera_dist=args.camera_dist,
                point_size=args.point_size_helical,
                helical_cmap_name=args.helical_cmap_name,
                box_scale=args.box_scale,
                dpi=args.dpi,
                figsize=custom_figsize,
            )
            return

        helical_file = helical_files[0]
        print(f"Loading helical trajectory: {helical_file}")
        helical_traj = load_trajectory_data(helical_file, use_x0=use_x0)

    strained_helical_traj = None
    if args.strained_helical is not None:
        strained_file = resolve_filepath(args.strained_helical)
        print(f"Loading strained helical trajectory: {strained_file}")
        strained_helical_traj = load_trajectory_data(strained_file, use_x0=use_x0)

    plot_point_cloud_comparison(
        binder_traj=binder_traj,
        helical_traj=helical_traj,
        strained_helical_traj=strained_helical_traj,
        axial_strain=args.axial_strain,
        show_strain_comparison=args.show_strain_comparison,
        tri_state_deformation=args.tri_state_deformation,
        compression_strain=args.compression_strain,
        tension_strain=args.tension_strain,
        lock_deformation_bounds=args.lock_deformation_bounds,
        poisson_ratio=args.poisson_ratio,
        deformation_mode=args.deformation_mode,
        sidechain_alpha=args.sidechain_alpha,
        output_path=args.output,
        num_timepoints=args.num_timepoints,
        frame_indices=args.frame_indices,
        t_start=args.t_start,
        spacing_mode=args.spacing_mode,
        mainchain_only=args.mainchain_only,
        ca_only=args.ca_only,
        elev=args.elev,
        azim=args.azim,
        camera_dist=args.camera_dist,
        binder_elev=args.binder_elev,
        binder_azim=args.binder_azim,
        helical_elev=args.helical_elev,
        helical_azim=args.helical_azim,
        point_size_target=args.point_size_target,
        point_size_binder=args.point_size_binder,
        point_size_helical=args.point_size_helical,
        target_color=args.target_color,
        target_alpha=args.target_alpha,
        static_target=args.static_target,
        target_all_atoms=args.target_all_atoms,
        orient_vertical_right=args.orient_vertical_right,
        helical_cmap_name=args.helical_cmap_name,
        box_scale=args.box_scale,
        show_row_labels=(not args.hide_row_labels),
        show_colorbar=(not args.no_colorbar),
        dpi=args.dpi,
        figsize=custom_figsize,
    )


if __name__ == "__main__":
    main()
