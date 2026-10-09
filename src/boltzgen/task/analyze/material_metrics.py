"""CPU-only geometric metrics for assembled peptide materials."""

# ruff: noqa: INP001, PLR2004

from __future__ import annotations

import torch
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
    return result
