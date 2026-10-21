"""CPU-only geometric metrics for assembled peptide materials."""

# ruff: noqa: INP001, PLR2004

from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from scipy.spatial import Delaunay, QhullError  # type: ignore[import-untyped]

VDW_RADIUS_ANGSTROM = 1.7
"""Flat approximate van der Waals radius used for pore-clearance estimates."""


def _validate_points(points: torch.Tensor, dimensions: int, name: str) -> None:
    if points.ndim != 2 or points.shape[1] != dimensions:
        message = f"{name} must have shape [N, {dimensions}]"
        raise ValueError(message)
    if not torch.isfinite(points).all():
        message = f"{name} must contain only finite coordinates"
        raise ValueError(message)


def cage_metrics(
    chain_coms: torch.Tensor,
    atom_coords: torch.Tensor,
    vdw_radius: float = VDW_RADIUS_ANGSTROM,
) -> dict[str, float]:
    """Calculate radial cage uniformity and approximate central void radius."""
    _validate_points(chain_coms, 3, "chain_coms")
    _validate_points(atom_coords, 3, "atom_coords")
    if chain_coms.shape[0] < 2:
        raise ValueError("cage metrics require at least two chain COMs")
    if atom_coords.shape[0] < 1:
        raise ValueError("cage_void_radius requires at least one atom coordinate")
    if vdw_radius < 0:
        raise ValueError("vdw_radius must be non-negative")

    centroid = chain_coms.mean(dim=0)
    radial_distances = torch.linalg.vector_norm(chain_coms - centroid, dim=1)
    return {
        "cage_void_radius": void_radius(chain_coms, atom_coords, vdw_radius),
        "cage_sphericity_rmsd": float(radial_distances.std(correction=0).item()),
    }


def void_radius(
    chain_coms: torch.Tensor,
    atom_coords: torch.Tensor,
    vdw_radius: float = VDW_RADIUS_ANGSTROM,
) -> float:
    """Estimate the clearance around the centroid of an assembled material."""
    _validate_points(chain_coms, 3, "chain_coms")
    _validate_points(atom_coords, 3, "atom_coords")
    if chain_coms.shape[0] < 1:
        raise ValueError("void radius requires at least one chain COM")
    if atom_coords.shape[0] < 1:
        raise ValueError("void radius requires at least one atom coordinate")
    if vdw_radius < 0:
        raise ValueError("vdw_radius must be non-negative")
    centroid = chain_coms.mean(dim=0)
    nearest_atom_distance = torch.linalg.vector_norm(
        atom_coords - centroid, dim=1
    ).min()
    return float(nearest_atom_distance.item() - vdw_radius)


def sheet_planarity(ca_coords: torch.Tensor) -> float:
    """Return planarity index from the eigenvalue spectrum of CA covariance."""
    _validate_points(ca_coords, 3, "ca_coords")
    if ca_coords.shape[0] < 3:
        return 0.0
    centered = ca_coords - ca_coords.mean(dim=0)
    covariance = centered.T @ centered / ca_coords.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total_variance = eigenvalues.sum()
    if total_variance <= torch.finfo(ca_coords.dtype).eps:
        return 0.0
    smallest_eigenvalue = eigenvalues[0]
    return float((1.0 - 3.0 * smallest_eigenvalue / total_variance).item())


def pore_aspect_ratio(
    chain_coms_xy: torch.Tensor,
    atom_coords_xy: torch.Tensor,
    vdw_radius: float = VDW_RADIUS_ANGSTROM,
) -> float | None:
    """Estimate interior mesh-pore aspect ratio from Delaunay pore centers.

    Finite patches and their Delaunay triangulation are only an approximate
    proxy for periodic mesh pores. Boundary triangles are excluded, and
    duplicate chain COM positions (e.g. the three chains in each C3 node) are
    collapsed before triangulation.
    """
    _validate_points(chain_coms_xy, 2, "chain_coms_xy")
    _validate_points(atom_coords_xy, 2, "atom_coords_xy")
    if atom_coords_xy.shape[0] == 0 or vdw_radius < 0:
        return None
    unique_coms = torch.unique(chain_coms_xy, dim=0)
    if unique_coms.shape[0] < 4:
        return None
    # Triangulation and distance checks are CPU-only and intentionally kept
    # outside the model's GPU execution path.
    unique_coms = unique_coms.detach().cpu()
    atom_coords_xy = atom_coords_xy.detach().cpu().to(dtype=unique_coms.dtype)
    try:
        triangulation = Delaunay(unique_coms.numpy())
    except QhullError:
        return None

    coords = atom_coords_xy
    interior_clearances: list[float] = []
    for simplex, neighbors in zip(triangulation.simplices, triangulation.neighbors):
        if (neighbors < 0).any():
            continue
        triangle = unique_coms[torch.as_tensor(simplex, device=unique_coms.device)]
        pore_center = triangle.mean(dim=0)
        clearance = (
            torch.linalg.vector_norm(coords - pore_center, dim=1).min().item()
            - vdw_radius
        )
        if clearance > 0:
            interior_clearances.append(float(clearance))

    if not interior_clearances:
        return None
    min_clearance = min(interior_clearances)
    return max(interior_clearances) / min_clearance


def backbone_ca_coords(
    atom_coords: torch.Tensor,
    atom_chain_ids: torch.Tensor,
    backbone_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract CA coordinates from N/CA/C/O backbone atoms, chain by chain."""
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
        raise ValueError("atom_coords must have shape [N, 3]")
    if atom_chain_ids.ndim != 1 or backbone_mask.ndim != 1:
        raise ValueError("atom_chain_ids and backbone_mask must be one-dimensional")
    n_atoms = min(atom_coords.shape[0], atom_chain_ids.shape[0], backbone_mask.shape[0])
    atom_coords = atom_coords[:n_atoms]
    atom_chain_ids = atom_chain_ids[:n_atoms]
    backbone_mask = backbone_mask[:n_atoms].bool()
    ca_by_chain = []
    for chain_id in torch.unique(atom_chain_ids[backbone_mask]):
        chain_coords = atom_coords[backbone_mask & (atom_chain_ids == chain_id)]
        if chain_coords.shape[0] % 4 != 0:
            continue
        ca_by_chain.append(chain_coords.reshape(-1, 4, 3)[:, 1, :])
    if not ca_by_chain:
        return atom_coords.new_empty((0, 3))
    return torch.cat(ca_by_chain, dim=0)


def cylinder_metrics(
    chain_coms: torch.Tensor,
    ca_coords: torch.Tensor,
    target_radius: float | None = None,
    vdw_radius: float = VDW_RADIUS_ANGSTROM,
) -> dict[str, float]:
    r"""Calculate cylindrical barrel geometry metrics.

    Identifies the principal cylinder axis from chain COMs, projects CA coordinates
    into the cross-sectional plane, and evaluates radial deviation against
    the target radius.

    Parameters
    ----------
    chain_coms : torch.Tensor
        Chain center-of-mass coordinates with shape ``[K, 3]``.
    ca_coords : torch.Tensor
        Residue CA coordinates with shape ``[L, 3]``.
    target_radius : float | None
        Target cylinder radius in Angstroms. If None, read from
        ``MAT_TARGET_RADIUS`` or mean radius.
    vdw_radius : float
        van der Waals clearance radius.

    Returns
    -------
    dict[str, float]
        Dictionary containing ``cylinder_radius_rmsd``, ``neg_cylinder_radius_rmsd``,
        ``cylinder_mean_radius``, and ``cylinder_pore_radius``.
    """
    _validate_points(chain_coms, 3, "chain_coms")
    _validate_points(ca_coords, 3, "ca_coords")
    if chain_coms.shape[0] < 3:
        raise ValueError("cylinder metrics require at least three chain COMs")
    if ca_coords.shape[0] < 1:
        raise ValueError("cylinder metrics require at least one CA coordinate")

    import os

    if target_radius is None:
        env_r = os.environ.get("MAT_TARGET_RADIUS")
        target_radius = float(env_r) if env_r is not None else None

    centroid = chain_coms.mean(dim=0)
    coms_centered = chain_coms - centroid

    _, _, vh = torch.linalg.svd(coms_centered)
    u_axis = vh[0]
    v_axis = vh[1]

    ca_centered = ca_coords - centroid
    x_proj = ca_centered @ u_axis
    y_proj = ca_centered @ v_axis
    radii = torch.sqrt(x_proj**2 + y_proj**2)

    mean_r = float(radii.mean().item())
    ref_radius = (
        float(target_radius)
        if target_radius is not None and target_radius > 0
        else mean_r
    )
    rmsd = float(torch.sqrt(torch.mean((radii - ref_radius) ** 2)).item())
    pore_r = float(radii.min().item() - vdw_radius)

    return {
        "cylinder_radius_rmsd": rmsd,
        "neg_cylinder_radius_rmsd": -rmsd,
        "cylinder_mean_radius": mean_r,
        "cylinder_pore_radius": pore_r,
    }


def compute_material_metrics(
    topology: str,
    chain_coms: torch.Tensor,
    backbone_coords: torch.Tensor,
    ca_coords: torch.Tensor,
) -> dict[str, float]:
    """Dispatch applicable material metrics for one topology."""
    result: dict[str, float] = {}
    if topology in {"cage_tetrahedral", "cage_octahedral"}:
        cage_values = cage_metrics(chain_coms, backbone_coords)
        result.update(cage_values)
        result["neg_cage_sphericity_rmsd"] = -cage_values["cage_sphericity_rmsd"]
    elif topology == "nanotube":
        result["cage_void_radius"] = void_radius(chain_coms, backbone_coords)
    elif topology == "bilayer_sheet":
        result["sheet_planarity"] = sheet_planarity(ca_coords)
    elif topology == "hexagonal_mesh":
        result["sheet_planarity"] = sheet_planarity(ca_coords)
        aspect_ratio = pore_aspect_ratio(chain_coms[:, :2], backbone_coords[:, :2])
        if aspect_ratio is not None:
            result["pore_aspect_ratio"] = aspect_ratio
            result["neg_pore_aspect_ratio"] = -aspect_ratio
    elif topology == "cylindrical":
        result.update(cylinder_metrics(chain_coms, ca_coords))
    return result


def _extract_chain_backbone_and_com(
    coords: torch.Tensor,
    chain_ids: torch.Tensor,
    bb_mask: torch.Tensor,
    partner_slots: dict[int, int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Extract per-chain backbone atom coordinates and centers of mass."""
    bb_dict: dict[int, torch.Tensor] = {}
    com_dict: dict[int, torch.Tensor] = {}
    n_atoms = min(coords.shape[0], chain_ids.shape[0], bb_mask.shape[0])
    coords = coords[:n_atoms]
    chain_ids = chain_ids[:n_atoms]
    bb_mask = bb_mask[:n_atoms].bool()

    for c in sorted(partner_slots.keys()):
        c_mask = bb_mask & (chain_ids == c)
        c_pts = coords[c_mask]
        if c_pts.shape[0] > 0:
            bb_dict[c] = c_pts
            com_dict[c] = c_pts.mean(dim=0)
    return bb_dict, com_dict


def _count_interface_contacts(
    chain_bb: dict[int, torch.Tensor],
    chain_coms: dict[int, torch.Tensor],
    partner_slots: dict[int, int],
    contact_distance: float,
) -> tuple[int, int]:
    """Count heteromeric and homomeric interface contact edges between chains."""
    valid_chains = [c for c in sorted(partner_slots.keys()) if c in chain_bb]
    n_hetero = 0
    n_homo = 0
    for i in range(len(valid_chains)):
        c_i = valid_chains[i]
        bb_i = chain_bb[c_i]
        slot_i = partner_slots[c_i]
        for j in range(i + 1, len(valid_chains)):
            c_j = valid_chains[j]
            bb_j = chain_bb[c_j]
            slot_j = partner_slots[c_j]
            if (
                torch.linalg.vector_norm(chain_coms[c_i] - chain_coms[c_j]).item()
                <= 25.0
            ):
                d_min = torch.cdist(bb_i, bb_j).min().item()
                if d_min <= contact_distance:
                    if slot_i != slot_j:
                        n_hetero += 1
                    else:
                        n_homo += 1
    return n_hetero, n_homo


def _align_phase(  # noqa: C901, PLR0912, PLR0915
    refold_bb: dict[int, torch.Tensor],
    refold_coms: dict[int, torch.Tensor],
    ideal_bb: dict[int, torch.Tensor],
    ideal_coms: dict[int, torch.Tensor],
    partner_slots_refold: dict[int, int],
    partner_slots_ideal: dict[int, int],
    target_slot_mapping: dict[int, int],
) -> tuple[float, int, list[tuple[int, int]]]:
    """Find SE(3)-invariant partner-constrained alignment, RMSD, and matched pairs."""
    chains_r = [c for c in sorted(partner_slots_refold.keys()) if c in refold_bb]
    chains_i = [c for c in sorted(partner_slots_ideal.keys()) if c in ideal_bb]
    if not chains_r or not chains_i:
        return 0.0, 0, []

    coms_r_list = [refold_coms[c] for c in chains_r]
    coms_i_list = [ideal_coms[c] for c in chains_i]
    coms_r = torch.stack(coms_r_list)
    coms_i = torch.stack(coms_i_list)

    center_r = coms_r.mean(dim=0)
    center_i = coms_i.mean(dim=0)
    cr = coms_r - center_r
    ci = coms_i - center_i

    device = coms_r.device
    dtype = coms_r.dtype

    candidate_rotations: list[torch.Tensor] = [torch.eye(3, device=device, dtype=dtype)]

    if len(chains_r) >= 3 and len(chains_i) == len(chains_r):
        h0 = cr.T @ ci
        u0, _, vh0 = torch.linalg.svd(h0)
        d0 = torch.sign(torch.det(u0 @ vh0))
        d_mat0 = torch.eye(3, device=device, dtype=dtype)
        d_mat0[2, 2] = d0
        r0 = u0 @ d_mat0 @ vh0
        candidate_rotations.append(r0)
        rx = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
        ry = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))
        rz = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype))
        for flip in (rx, ry, rz):
            candidate_rotations.append(r0 @ flip)
            candidate_rotations.append(flip)
    else:
        rx = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
        ry = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))
        rz = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype))
        candidate_rotations.extend([rx, ry, rz])

    best_rmsd = float("inf")
    best_mismatches = 0
    best_pairs: list[tuple[int, int]] = []

    for r_init in candidate_rotations:
        rot_cr = cr @ r_init

        matched_pairs: list[tuple[int, int]] = []
        for s in (0, 1):
            s_target = target_slot_mapping.get(s, s)
            sub_r_indices = [
                idx for idx, c in enumerate(chains_r) if partner_slots_refold[c] == s
            ]
            sub_i_indices = [
                idx
                for idx, c in enumerate(chains_i)
                if partner_slots_ideal[c] == s_target
            ]
            if sub_r_indices and sub_i_indices:
                sub_costs = (
                    torch.cdist(rot_cr[sub_r_indices], ci[sub_i_indices]).cpu().numpy()
                )
                row_sub, col_sub = linear_sum_assignment(sub_costs)
                for r_idx, c_idx in zip(row_sub, col_sub):
                    matched_pairs.append(
                        (chains_r[sub_r_indices[r_idx]], chains_i[sub_i_indices[c_idx]])
                    )

        costs_all = torch.cdist(rot_cr, ci).cpu().numpy()
        row_all, col_all = linear_sum_assignment(costs_all)
        mismatch_count = sum(
            1
            for r_idx, c_idx in zip(row_all, col_all)
            if target_slot_mapping.get(partner_slots_refold[chains_r[r_idx]])
            != partner_slots_ideal[chains_i[c_idx]]
        )

        if not matched_pairs:
            continue

        aligned_r: list[torch.Tensor] = []
        aligned_i: list[torch.Tensor] = []
        for u, v in matched_pairs:
            pts_u = refold_bb[u]
            pts_v = ideal_bb[v]
            if pts_u.shape[0] == pts_v.shape[0]:
                aligned_r.append(pts_u)
                aligned_i.append(pts_v)

        if aligned_r:
            cat_r = torch.cat(aligned_r)
            cat_i = torch.cat(aligned_i)
            p = cat_r - cat_r.mean(dim=0)
            q = cat_i - cat_i.mean(dim=0)
            h = p.T @ q
            u_mat, _, vh_mat = torch.linalg.svd(h)
            d_sign = torch.sign(torch.det(u_mat @ vh_mat))
            d_rot = torch.eye(3, device=device, dtype=dtype)
            d_rot[2, 2] = d_sign
            r_opt = u_mat @ d_rot @ vh_mat
            p_aligned = p @ r_opt
            cand_rmsd = float(torch.sqrt(torch.mean((p_aligned - q) ** 2)).item())
            if cand_rmsd < best_rmsd:
                best_rmsd = cand_rmsd
                best_mismatches = mismatch_count
                best_pairs = list(matched_pairs)

    if math.isinf(best_rmsd):
        best_rmsd = 99.0

    return best_rmsd, best_mismatches, best_pairs


def compute_two_chain_alternating_metrics(
    coords_refold: torch.Tensor,
    coords_ideal: torch.Tensor,
    chain_ids_refold: torch.Tensor,
    chain_ids_ideal: torch.Tensor,
    partner_slots_refold: dict[int, int],
    partner_slots_ideal: dict[int, int],
    bb_mask_refold: torch.Tensor,
    bb_mask_ideal: torch.Tensor,
    lambda_penalty: float = 5.0,
    contact_distance: float = 5.0,
) -> dict[str, float]:
    r"""Compute phase-invariant partner RMSD and contact-based alternation metrics.

    Evaluates:
    1. Interface contact graphs of refolded and ideal structures:
       .. math::
           f_{\text{alt}} = \min\left(
               1.0, \frac{f_{\text{alt}}^{\text{refold}}}{f_{\text{alt}}^{\text{ideal}}}
           \right)
    2. Phase-invariant partner-constrained Kabsch RMSD across phases
       :math:`\phi \in \{0, 1\}`:
       .. math::
           \text{RMSD}_{\text{geom}} = \min(\text{RMSD}_{\phi=0}, \text{RMSD}_{\phi=1})
    3. Penalized alternating primary RMSD:
       .. math::
           \text{RMSD}_{\text{primary}} = \text{RMSD}_{\text{geom}} +
           \lambda \cdot (1.0 - f_{\text{alt}})

    Parameters
    ----------
    coords_refold : torch.Tensor
        Refolded structure atom coordinates with shape ``[N_refold, 3]``.
    coords_ideal : torch.Tensor
        Ideal designed lattice atom coordinates with shape ``[N_ideal, 3]``.
    chain_ids_refold : torch.Tensor
        Refolded atom chain identifiers with shape ``[N_refold]``.
    chain_ids_ideal : torch.Tensor
        Ideal atom chain identifiers with shape ``[N_ideal]``.
    partner_slots_refold : dict[int, int]
        Mapping from refolded chain ID to partner slot index (0 or 1).
    partner_slots_ideal : dict[int, int]
        Mapping from ideal chain ID to partner slot index (0 or 1).
    bb_mask_refold : torch.Tensor
        Backbone atom mask for refolded structure with shape ``[N_refold]``.
    bb_mask_ideal : torch.Tensor
        Backbone atom mask for ideal structure with shape ``[N_ideal]``.
    lambda_penalty : float, default=5.0
        Penalty weighting in Angstroms for alternation deficit:
        :math:`\lambda (1 - f_{\text{alt}})`.
    contact_distance : float, default=5.0
        Maximum inter-chain backbone atom distance in Angstroms defining
        an interface contact edge.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - ``lattice_rmsd_refolded``: Primary penalized alternating RMSD in Angstroms.
        - ``lattice_rmsd_geom``: Minimum coordinate RMSD across phases in Angstroms.
        - ``lattice_rmsd_phase_0``: Coordinate RMSD under phase 0 alignment
          in Angstroms.
        - ``lattice_rmsd_phase_1``: Coordinate RMSD under phase 1 alignment
          in Angstroms.
        - ``two_chain_alternation_ratio``: Normalized alternation score
          :math:`f_{\text{alt}} \in [0.0, 1.0]`.
        - ``two_chain_heteromeric_contacts``: Count of heteromeric interfaces
          in refolded complex.
        - ``two_chain_homomeric_contacts``: Count of homomeric interfaces
          in refolded complex.
        - ``partner_mismatch_count``: Minimum partner mismatches across phases.
    """
    refold_bb, refold_com = _extract_chain_backbone_and_com(
        coords_refold, chain_ids_refold, bb_mask_refold, partner_slots_refold
    )
    ideal_bb, ideal_com = _extract_chain_backbone_and_com(
        coords_ideal, chain_ids_ideal, bb_mask_ideal, partner_slots_ideal
    )

    rmsd_phase_0, mismatches_0, _ = _align_phase(
        refold_bb,
        refold_com,
        ideal_bb,
        ideal_com,
        partner_slots_refold,
        partner_slots_ideal,
        {0: 0, 1: 1},
    )
    rmsd_phase_1, mismatches_1, _ = _align_phase(
        refold_bb,
        refold_com,
        ideal_bb,
        ideal_com,
        partner_slots_refold,
        partner_slots_ideal,
        {0: 1, 1: 0},
    )

    rmsd_geom = min(rmsd_phase_0, rmsd_phase_1)
    partner_mismatches = mismatches_0 if rmsd_phase_0 <= rmsd_phase_1 else mismatches_1

    n_hetero_refold, n_homo_refold = _count_interface_contacts(
        refold_bb, refold_com, partner_slots_refold, contact_distance
    )
    n_hetero_ideal, n_homo_ideal = _count_interface_contacts(
        ideal_bb, ideal_com, partner_slots_ideal, contact_distance
    )

    tot_ideal = n_hetero_ideal + n_homo_ideal
    f_alt_ideal = float(n_hetero_ideal / tot_ideal) if tot_ideal > 0 else 1.0

    tot_refold = n_hetero_refold + n_homo_refold
    f_alt_refold = float(n_hetero_refold / tot_refold) if tot_refold > 0 else 1.0

    if f_alt_ideal > 0.0:
        alternation_ratio = float(min(1.0, f_alt_refold / f_alt_ideal))
    else:
        alternation_ratio = 1.0

    penalty = float(lambda_penalty * (1.0 - alternation_ratio))
    rmsd_primary = float(rmsd_geom + penalty)

    return {
        "lattice_rmsd_refolded": rmsd_primary,
        "lattice_rmsd_geom": rmsd_geom,
        "lattice_rmsd_phase_0": rmsd_phase_0,
        "lattice_rmsd_phase_1": rmsd_phase_1,
        "two_chain_alternation_ratio": alternation_ratio,
        "two_chain_heteromeric_contacts": float(n_hetero_refold),
        "two_chain_homomeric_contacts": float(n_homo_refold),
        "partner_mismatch_count": float(partner_mismatches),
    }


def _strand_direction(bb_coords: torch.Tensor) -> torch.Tensor:
    r"""Compute unit N->C backbone direction vector.

    .. math::
        \vec{v} = \frac{\langle \vec{x}_{\text{tail}} \rangle -
        \langle \vec{x}_{\text{head}} \rangle}{\|\langle \vec{x}_{\text{tail}} \rangle -
        \langle \vec{x}_{\text{head}} \rangle\|_2}

    Parameters
    ----------
    bb_coords : torch.Tensor
        Backbone coordinates of shape ``[N_atoms, 3]``.

    Returns
    -------
    torch.Tensor
        Unit vector of shape ``[3]``.
    """
    if bb_coords.shape[0] < 2:
        return torch.tensor(
            [0.0, 1.0, 0.0], device=bb_coords.device, dtype=bb_coords.dtype
        )
    k = max(1, bb_coords.shape[0] // 4)
    v = bb_coords[-k:].mean(dim=0) - bb_coords[:k].mean(dim=0)
    norm = torch.linalg.norm(v)
    if norm > 1e-8:
        return v / norm
    return v


def _kabsch_rmsd(p_pts: torch.Tensor, q_pts: torch.Tensor) -> float:
    r"""Compute coordinate RMSD after optimal rigid superposition (Kabsch algorithm).

    .. math::
        \text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|(\vec{p}_i - \bar{\vec{p}}) R -
        (\vec{q}_i - \bar{\vec{q}})\|^2}

    Parameters
    ----------
    p_pts : torch.Tensor
        Mobile coordinates of shape ``[N, 3]``.
    q_pts : torch.Tensor
        Reference coordinates of shape ``[N, 3]``.

    Returns
    -------
    float
        Superimposed RMSD in Angstroms.
    """
    if p_pts.shape[0] != q_pts.shape[0] or p_pts.shape[0] == 0:
        return 99.0
    p = p_pts - p_pts.mean(dim=0)
    q = q_pts - q_pts.mean(dim=0)
    h = p.T @ q
    u, _, vh = torch.linalg.svd(h)
    d_sign = torch.sign(torch.det(u @ vh))
    d_rot = torch.eye(3, device=p.device, dtype=p.dtype)
    d_rot[2, 2] = d_sign
    r_opt = u @ d_rot @ vh
    p_aligned = p @ r_opt
    return float(torch.sqrt(torch.mean((p_aligned - q) ** 2)).item())


def compute_double_tape_neatness_metrics(  # noqa: C901, PLR0912, PLR0915
    coords_refold: torch.Tensor,
    coords_ideal: torch.Tensor,
    chain_ids_refold: torch.Tensor,
    chain_ids_ideal: torch.Tensor,
    partner_slots_refold: dict[int, int],
    partner_slots_ideal: dict[int, int],
    bb_mask_refold: torch.Tensor,
    bb_mask_ideal: torch.Tensor,
    lambda_penalty: float = 5.0,
    contact_distance: float = 5.0,
) -> dict[str, float]:
    r"""Compute supercoiling-invariant double-layer neatness and polarity metrics.

    Evaluates:
    1. Target-aware in-sheet strand polarity match against ideal design:
       .. math::
           \text{Match}_{\ell, k} = \mathbb{I}\left(
               (\vec{v}_{\ell, k}^{\text{refold}} \cdot
               \vec{v}_{\ell, k+1}^{\text{refold}})
               \cdot
               (\vec{v}_{\ell, k}^{\text{ideal}} \cdot
               \vec{v}_{\ell, k+1}^{\text{ideal}}) > 0
           \right)
       .. math::
           \text{PolarityMatch} = \frac{1}{2(K-1)} \sum_{\ell \in \{0, 1\}}
           \sum_{k=0}^{K-2} \text{Match}_{\ell, k}
    2. Layer-segregated intra-sheet heterotypic alternation ratio:
       .. math::
           \text{IntraSheetAlternation} = \frac{n_{\text{hetero}}^{\text{intra}}}{
               n_{\text{hetero}}^{\text{intra}} + n_{\text{homo}}^{\text{intra}} +
               \epsilon}
    3. Supercoiling-invariant overlapping 2x2 unit-cell Kabsch RMSD:
       .. math::
           \text{LocalStepRMSD} = \frac{1}{K-1} \sum_{k=0}^{K-2}
           \text{RMSD}_k(U_k^{\text{refold}}, U_k^{\text{ideal}})
    4. Bilayer thickness mean :math:`D_{\text{mean}}` and
       uniformity :math:`D_{\text{std}}`:
       .. math::
           d_k = \|\text{COM}(C_{0, k}) - \text{COM}(C_{1, k})\|_2
    5. Penalized local double-tape RMSD:
       .. math::
           \text{RMSD}_{\text{neat}} = \text{LocalStepRMSD} +
           \lambda (1 - \text{PolarityMatch}) +
           \lambda (1 - \text{IntraSheetAlternation})
    6. Continuous composite neatness score:
       .. math::
           S_{\text{neat}} = \text{PolarityMatch} \cdot
           \text{IntraSheetAlternation} \cdot
           \exp\left(-\frac{\text{LocalStepRMSD}}{2.5}\right) \cdot
           \exp\left(-\frac{D_{\text{std}}}{1.0}\right)

    Parameters
    ----------
    coords_refold : torch.Tensor
        Refolded structure atom coordinates with shape ``[N_refold, 3]``.
    coords_ideal : torch.Tensor
        Ideal designed lattice atom coordinates with shape ``[N_ideal, 3]``.
    chain_ids_refold : torch.Tensor
        Refolded atom chain identifiers with shape ``[N_refold]``.
    chain_ids_ideal : torch.Tensor
        Ideal atom chain identifiers with shape ``[N_ideal]``.
    partner_slots_refold : dict[int, int]
        Mapping from refolded chain ID to partner slot index (0 or 1).
    partner_slots_ideal : dict[int, int]
        Mapping from ideal chain ID to partner slot index (0 or 1).
    bb_mask_refold : torch.Tensor
        Backbone atom mask for refolded structure with shape ``[N_refold]``.
    bb_mask_ideal : torch.Tensor
        Backbone atom mask for ideal structure with shape ``[N_ideal]``.
    lambda_penalty : float, default=5.0
        Penalty scaling in Angstroms for polarity and alternation violations.
    contact_distance : float, default=5.0
        Maximum backbone distance in Angstroms defining an in-sheet interface contact.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - ``local_double_tape_rmsd``: Supercoiling-invariant penalized local RMSD.
        - ``local_step_rmsd``: Average 2x2 unit-cell Kabsch RMSD in Angstroms.
        - ``max_step_rmsd``: Maximum 2x2 unit-cell Kabsch RMSD in Angstroms.
        - ``strand_polarity_target_match``: Fraction of in-sheet pairs matching
          polarity.
        - ``strand_polarity_mean_alignment``: Mean continuous alignment dot product.
        - ``intra_sheet_alternation_ratio``: In-sheet alternation score in [0.0, 1.0].
        - ``intra_sheet_heteromeric_contacts``: Count of in-sheet heteromeric contacts.
        - ``intra_sheet_homomeric_contacts``: Count of in-sheet homomeric contacts.
        - ``bilayer_dist_mean``: Mean bilayer thickness across steps in Angstroms.
        - ``bilayer_dist_std``: Standard deviation of bilayer thickness in Angstroms.
        - ``double_tape_neatness_score``: Composite neatness quality score.
    """
    refold_bb, refold_com = _extract_chain_backbone_and_com(
        coords_refold, chain_ids_refold, bb_mask_refold, partner_slots_refold
    )
    ideal_bb, ideal_com = _extract_chain_backbone_and_com(
        coords_ideal, chain_ids_ideal, bb_mask_ideal, partner_slots_ideal
    )

    chains_i = sorted([c for c in partner_slots_ideal if c in ideal_bb])
    chains_r = sorted([c for c in partner_slots_refold if c in refold_bb])
    num_chains = len(chains_i)
    num_steps = num_chains // 2

    if num_chains < 4 or len(chains_r) < 4 or num_steps < 2:
        return {
            "local_double_tape_rmsd": 0.0,
            "local_step_rmsd": 0.0,
            "max_step_rmsd": 0.0,
            "strand_polarity_target_match": 1.0,
            "strand_polarity_mean_alignment": 1.0,
            "intra_sheet_alternation_ratio": 1.0,
            "intra_sheet_heteromeric_contacts": 0.0,
            "intra_sheet_homomeric_contacts": 0.0,
            "bilayer_dist_mean": 0.0,
            "bilayer_dist_std": 0.0,
            "double_tape_neatness_score": 1.0,
        }

    rmsd_0, _, pairs_0 = _align_phase(
        refold_bb, refold_com, ideal_bb, ideal_com,
        partner_slots_refold, partner_slots_ideal, {0: 0, 1: 1}
    )
    rmsd_1, _, pairs_1 = _align_phase(
        refold_bb, refold_com, ideal_bb, ideal_com,
        partner_slots_refold, partner_slots_ideal, {0: 1, 1: 0}
    )
    best_pairs = pairs_0 if rmsd_0 <= rmsd_1 else pairs_1
    ideal_to_refold = {i: r for r, i in best_pairs}

    # 1. Target-aware in-sheet strand polarity match
    polarity_matches: list[float] = []
    polarity_alignments: list[float] = []
    for ell in (0, 1):
        for k in range(num_steps - 1):
            ik = chains_i[2 * k + ell]
            ik_next = chains_i[2 * (k + 1) + ell]
            rk = ideal_to_refold.get(ik, ik)
            rk_next = ideal_to_refold.get(ik_next, ik_next)
            if rk not in refold_bb or rk_next not in refold_bb:
                continue
            v_i_k = _strand_direction(ideal_bb[ik])
            v_i_next = _strand_direction(ideal_bb[ik_next])
            dot_i = float(torch.dot(v_i_k, v_i_next).item())

            v_r_k = _strand_direction(refold_bb[rk])
            v_r_next = _strand_direction(refold_bb[rk_next])
            dot_r = float(torch.dot(v_r_k, v_r_next).item())

            match = float((dot_r * dot_i) > 0.0)
            polarity_matches.append(match)
            norm_dot_i = 1.0 if dot_i >= 0.0 else -1.0
            clamped_dot_r = max(-1.0, min(1.0, dot_r))
            polarity_alignments.append((1.0 + norm_dot_i * clamped_dot_r) / 2.0)

    strand_polarity_target_match = (
        sum(polarity_matches) / len(polarity_matches) if polarity_matches else 1.0
    )
    strand_polarity_mean_alignment = (
        sum(polarity_alignments) / len(polarity_alignments)
        if polarity_alignments
        else 1.0
    )

    # 2. Layer-segregated intra-sheet heterotypic alternation
    n_hetero_intra = 0
    n_homo_intra = 0
    for ell in (0, 1):
        sheet_chains = [
            ideal_to_refold.get(chains_i[2 * k + ell], chains_i[2 * k + ell])
            for k in range(num_steps)
        ]
        for a_idx in range(len(sheet_chains)):
            ca = sheet_chains[a_idx]
            if ca not in refold_bb:
                continue
            bb_a = refold_bb[ca]
            slot_a = partner_slots_refold.get(ca, 0)
            for b_idx in range(a_idx + 1, len(sheet_chains)):
                cb = sheet_chains[b_idx]
                if cb not in refold_bb:
                    continue
                bb_b = refold_bb[cb]
                slot_b = partner_slots_refold.get(cb, 0)
                if torch.linalg.norm(refold_com[ca] - refold_com[cb]) <= 25.0:
                    d_min = float(torch.min(torch.cdist(bb_a, bb_b)).item())
                    if d_min <= contact_distance:
                        if slot_a != slot_b:
                            n_hetero_intra += 1
                        else:
                            n_homo_intra += 1

    tot_intra = n_hetero_intra + n_homo_intra
    intra_sheet_alternation_ratio = (
        float(n_hetero_intra / tot_intra) if tot_intra > 0 else 1.0
    )

    # 3. Supercoiling-invariant overlapping 2x2 unit-cell Kabsch RMSD
    step_rmsds: list[float] = []
    for k in range(num_steps - 1):
        i0 = chains_i[2 * k]
        i1 = chains_i[2 * k + 1]
        i2 = chains_i[2 * (k + 1)]
        i3 = chains_i[2 * (k + 1) + 1]

        r0 = ideal_to_refold.get(i0, i0)
        r1 = ideal_to_refold.get(i1, i1)
        r2 = ideal_to_refold.get(i2, i2)
        r3 = ideal_to_refold.get(i3, i3)

        if (
            r0 in refold_bb
            and r1 in refold_bb
            and r2 in refold_bb
            and r3 in refold_bb
            and i0 in ideal_bb
            and i1 in ideal_bb
            and i2 in ideal_bb
            and i3 in ideal_bb
        ):
            ideal_unit = torch.cat(
                [ideal_bb[i0], ideal_bb[i1], ideal_bb[i2], ideal_bb[i3]]
            )
            refold_unit = torch.cat(
                [refold_bb[r0], refold_bb[r1], refold_bb[r2], refold_bb[r3]]
            )
            refold_swap = torch.cat(
                [refold_bb[r1], refold_bb[r0], refold_bb[r3], refold_bb[r2]]
            )

            rmsd_direct = _kabsch_rmsd(refold_unit, ideal_unit)
            rmsd_swap = _kabsch_rmsd(refold_swap, ideal_unit)
            step_rmsds.append(min(rmsd_direct, rmsd_swap))

    local_step_rmsd = (
        sum(step_rmsds) / len(step_rmsds) if step_rmsds else 0.0
    )
    max_step_rmsd = max(step_rmsds) if step_rmsds else 0.0

    # 4. Bilayer thickness mean and uniformity
    bilayer_dists: list[float] = []
    for k in range(num_steps):
        i0 = chains_i[2 * k]
        i1 = chains_i[2 * k + 1]
        r0 = ideal_to_refold.get(i0, i0)
        r1 = ideal_to_refold.get(i1, i1)
        if r0 in refold_com and r1 in refold_com:
            dist = float(torch.linalg.norm(refold_com[r0] - refold_com[r1]).item())
            bilayer_dists.append(dist)

    if bilayer_dists:
        dists_tensor = torch.tensor(bilayer_dists, dtype=torch.float32)
        bilayer_dist_mean = float(dists_tensor.mean().item())
        bilayer_dist_std = float(dists_tensor.std(correction=0).item())
    else:
        bilayer_dist_mean = 0.0
        bilayer_dist_std = 0.0

    # 5. Penalized local double-tape RMSD and neatness score
    polarity_penalty = lambda_penalty * (1.0 - strand_polarity_target_match)
    alternation_penalty = lambda_penalty * (1.0 - intra_sheet_alternation_ratio)
    local_double_tape_rmsd = float(
        local_step_rmsd + polarity_penalty + alternation_penalty
    )

    double_tape_neatness_score = float(
        strand_polarity_target_match
        * intra_sheet_alternation_ratio
        * math.exp(-local_step_rmsd / 2.5)
        * math.exp(-bilayer_dist_std / 1.0)
    )

    return {
        "local_double_tape_rmsd": local_double_tape_rmsd,
        "local_step_rmsd": local_step_rmsd,
        "max_step_rmsd": max_step_rmsd,
        "strand_polarity_target_match": strand_polarity_target_match,
        "strand_polarity_mean_alignment": strand_polarity_mean_alignment,
        "intra_sheet_alternation_ratio": intra_sheet_alternation_ratio,
        "intra_sheet_heteromeric_contacts": float(n_hetero_intra),
        "intra_sheet_homomeric_contacts": float(n_homo_intra),
        "bilayer_dist_mean": bilayer_dist_mean,
        "bilayer_dist_std": bilayer_dist_std,
        "double_tape_neatness_score": double_tape_neatness_score,
    }
