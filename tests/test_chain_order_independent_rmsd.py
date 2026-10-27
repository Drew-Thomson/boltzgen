"""Tests for exact SE(3)-invariant, chain-permutation-independent RMSD."""

import time
import torch

from boltzgen.task.analyze.material_metrics import compute_chain_order_independent_rmsd


def _generate_synthetic_cyclic_bundle(
    k_chains: int = 6,
    res_per_chain: int = 20,
    radius: float = 10.0,
    antiparallel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic coordinates for an idealized cyclic helical bundle.

    Parameters
    ----------
    k_chains : int
        Number of chains in the ring.
    res_per_chain : int
        Residues per chain (4 backbone atoms N, CA, C, O per residue).
    radius : float
        Radius of bundle in Angstroms.
    antiparallel : bool
        Whether adjacent chains alternate strand direction along Z.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        coords of shape [k_chains * res_per_chain * 4, 3],
        chain_ids of shape [k_chains * res_per_chain * 4],
        bb_mask of shape [k_chains * res_per_chain * 4].
    """
    coords_list: list[torch.Tensor] = []
    chain_ids_list: list[int] = []
    bb_mask_list: list[bool] = []

    for k in range(k_chains):
        angle = 2.0 * 3.141592653589793 * k / k_chains
        cx = radius * torch.cos(torch.tensor(angle))
        cy = radius * torch.sin(torch.tensor(angle))

        # Direction along Z
        direction = -1.0 if (antiparallel and k % 2 == 1) else 1.0

        rot_k = torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
            [0.0, 0.0, 1.0],
        ])

        for r in range(res_per_chain):
            z = direction * (r * 1.5)
            # 4 backbone atoms per residue in local frame rotated by rot_k
            ca = torch.tensor([cx, cy, z])
            n = ca + torch.tensor([0.0, 0.5, -0.6 * direction]) @ rot_k.T
            c = ca + torch.tensor([0.0, -0.5, 0.6 * direction]) @ rot_k.T
            o = c + torch.tensor([0.4, 0.0, 0.2 * direction]) @ rot_k.T

            res_coords = torch.stack([n, ca, c, o])
            coords_list.append(res_coords)
            chain_ids_list.extend([k] * 4)
            bb_mask_list.extend([True] * 4)

    coords = torch.cat(coords_list, dim=0).float()
    chain_ids = torch.tensor(chain_ids_list, dtype=torch.long)
    bb_mask = torch.tensor(bb_mask_list, dtype=torch.bool)
    return coords, chain_ids, bb_mask


def test_chain_order_independent_rmsd_identical_structures():
    torch.manual_seed(42)
    coords, chain_ids, bb_mask = _generate_synthetic_cyclic_bundle(k_chains=6)
    rmsd, matched_pairs = compute_chain_order_independent_rmsd(
        coords_refold=coords,
        coords_ideal=coords,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        topology="cyclic",
    )
    assert abs(rmsd) < 1e-4
    # All 6 chains are matched bijectively
    refold_matched = sorted([p[0] for p in matched_pairs])
    ideal_matched = sorted([p[1] for p in matched_pairs])
    assert refold_matched == list(range(6))
    assert ideal_matched == list(range(6))


def test_chain_order_independent_rmsd_cyclic_shift():
    torch.manual_seed(42)
    coords, chain_ids, bb_mask = _generate_synthetic_cyclic_bundle(k_chains=6, antiparallel=False)

    # Shift chains cyclically: chain 0 becomes chain 1, 1 becomes 2, ..., 5 becomes 0
    shifted_coords_list = []
    shifted_ids_list = []
    atoms_per_chain = 20 * 4

    # Permute blocks: new chain 0 gets old chain 2, new chain 1 gets old chain 3, etc.
    perm = [2, 3, 4, 5, 0, 1]
    for new_k, old_k in enumerate(perm):
        start = old_k * atoms_per_chain
        end = start + atoms_per_chain
        shifted_coords_list.append(coords[start:end])
        shifted_ids_list.extend([new_k] * atoms_per_chain)

    shifted_coords = torch.cat(shifted_coords_list, dim=0)
    shifted_ids = torch.tensor(shifted_ids_list, dtype=torch.long)

    rmsd, matched_pairs = compute_chain_order_independent_rmsd(
        coords_refold=shifted_coords,
        coords_ideal=coords,
        chain_ids_refold=shifted_ids,
        chain_ids_ideal=chain_ids,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        topology="cyclic",
    )
    assert abs(rmsd) < 1e-4
    refold_matched = sorted([p[0] for p in matched_pairs])
    ideal_matched = sorted([p[1] for p in matched_pairs])
    assert refold_matched == list(range(6))
    assert ideal_matched == list(range(6))


def test_chain_order_independent_rmsd_antiparallel_flip():
    torch.manual_seed(42)
    coords, chain_ids, bb_mask = _generate_synthetic_cyclic_bundle(k_chains=6, antiparallel=True)

    # Rotate the entire complex by 180 degrees around X axis: (x, y, z) -> (x, -y, -z)
    # In an antiparallel bundle, rotating 180 deg around in-plane axis flips UP to DOWN
    flip_x = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    flipped_coords = coords @ flip_x

    # Permute chain IDs arbitrarily
    perm = [3, 4, 5, 0, 1, 2]
    atoms_per_chain = 20 * 4
    permuted_coords_list = []
    permuted_ids_list = []
    for new_k, old_k in enumerate(perm):
        start = old_k * atoms_per_chain
        end = start + atoms_per_chain
        permuted_coords_list.append(flipped_coords[start:end])
        permuted_ids_list.extend([new_k] * atoms_per_chain)

    refold_coords = torch.cat(permuted_coords_list, dim=0)
    refold_ids = torch.tensor(permuted_ids_list, dtype=torch.long)

    rmsd, _ = compute_chain_order_independent_rmsd(
        coords_refold=refold_coords,
        coords_ideal=coords,
        chain_ids_refold=refold_ids,
        chain_ids_ideal=chain_ids,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        topology="cyclic",
    )
    assert abs(rmsd) < 1e-4


def test_chain_order_independent_rmsd_distorted_reflection():
    torch.manual_seed(42)
    coords, chain_ids, bb_mask = _generate_synthetic_cyclic_bundle(k_chains=6)
    # Add a known geometric shift of 2.0 A along X to all atoms
    noisy_coords = coords.clone()
    noisy_coords[:, 0] += 2.0  # Rigid shift only -> RMSD remains 0 after Kabsch

    rmsd, _ = compute_chain_order_independent_rmsd(
        coords_refold=noisy_coords,
        coords_ideal=coords,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert abs(rmsd) < 1e-4

    # Non-rigid deformation
    distorted = coords.clone()
    distorted[::2] += 3.0
    dist_rmsd, _ = compute_chain_order_independent_rmsd(
        coords_refold=distorted,
        coords_ideal=coords,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert dist_rmsd > 1.0


def test_chain_order_independent_rmsd_performance_benchmark():
    torch.manual_seed(42)
    coords, chain_ids, bb_mask = _generate_synthetic_cyclic_bundle(k_chains=6, res_per_chain=25)

    # Warmup
    compute_chain_order_independent_rmsd(
        coords, coords, chain_ids, chain_ids, bb_mask, bb_mask
    )

    t0 = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        compute_chain_order_independent_rmsd(
            coords, coords, chain_ids, chain_ids, bb_mask, bb_mask
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
    assert elapsed_ms < 50.0  # Easily completes in milliseconds
