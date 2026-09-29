"""Ideal chain lattices and local-frame transforms for material guidance."""

from __future__ import annotations

import math

import torch


def _rotation_z(angle: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a row-vector-compatible rotation matrix about Z."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _ring_points(count: int, radius: float, *, device: torch.device, dtype: torch.dtype):
    angles = torch.arange(count, device=device, dtype=dtype) * (2.0 * math.pi / count)
    points = torch.zeros((count, 3), device=device, dtype=dtype)
    points[:, 0] = radius * torch.cos(angles)
    points[:, 1] = radius * torch.sin(angles)
    rotations = torch.stack(
        [_rotation_z(float(angle), device=device, dtype=dtype) for angle in angles]
    )
    return points, rotations


def generate_ideal_lattice(
    topology: str,
    n_chains: int,
    asym_unit_size: int,
    params: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create target COMs and per-copy rotations for a material topology.

    Rotations map coordinates in the consensus/local frame into the ideal
    lattice frame. COM and rotation arrays have shapes ``[n_chains, 3]`` and
    ``[n_chains, 3, 3]`` respectively.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be positive")

    radius = float(params.get("target_radius", 15.0))
    dz = float(params.get("target_dz", 5.0))
    pitch = float(params.get("target_pitch", 4.8))
    identity = torch.eye(3, device=device, dtype=dtype)

    if topology == "cyclic":
        coms, rotations = _ring_points(n_chains, radius, device=device, dtype=dtype)
    elif topology == "linear_tape":
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 2] = (torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2) * pitch
        rotations = identity.expand(n_chains, -1, -1).clone()
    elif topology == "helical":
        angles = torch.arange(n_chains, device=device, dtype=dtype) * math.radians(
            float(params.get("target_angle", 30.0))
        )
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 0] = radius * torch.cos(angles)
        coms[:, 1] = radius * torch.sin(angles)
        coms[:, 2] = (torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2) * dz
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "open_arc":
        arc_radius = float(params.get("arc_radius", 100.0))
        spacing = float(params.get("target_arc_spacing", 10.0))
        step = 2.0 * math.asin(min(spacing / (2.0 * arc_radius), 1.0))
        angles = (torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2) * step
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 0] = arc_radius * torch.sin(angles)
        coms[:, 1] = arc_radius * (1.0 - torch.cos(angles))
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "double_tape":
        # Two offset rows, alternating chain indices between the layers.
        layer_dist = float(params.get("layer_dist", 10.0))
        units = math.ceil(n_chains / 2)
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            layer = i % 2
            step = i // 2
            coms[i, 1] = (layer - 0.5) * layer_dist
            coms[i, 2] = (step - (units - 1) / 2) * pitch
        rotations = identity.expand(n_chains, -1, -1).clone()
    elif topology == "bilayer_sheet":
        gx, gy = int(params.get("grid_dim_x", 2)), int(params.get("grid_dim_y", 2))
        if n_chains != 2 * gx * gy:
            raise ValueError(f"bilayer_sheet requires {2 * gx * gy} copies, got {n_chains}")
        row_pitch = float(params.get("row_pitch", 10.0))
        layer_dist = float(params.get("layer_dist", 10.0))
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            layer, position = divmod(i, gx * gy)
            x_idx, y_idx = divmod(position, gy)
            coms[i] = torch.tensor(
                [
                    (x_idx - (gx - 1) / 2) * row_pitch,
                    (layer - 0.5) * layer_dist,
                    (y_idx - (gy - 1) / 2) * pitch,
                ],
                device=device,
                dtype=dtype,
            )
        rotations = identity.expand(n_chains, -1, -1).clone()
    elif topology == "nanotube":
        ring_size = int(params.get("ring_size", 4))
        tiers = int(params.get("num_tiers", 4))
        if n_chains != ring_size * tiers:
            raise ValueError(f"nanotube requires {ring_size * tiers} copies, got {n_chains}")
        stagger = math.radians(float(params.get("chiral_stagger", 0.0)))
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        rotations = torch.empty((n_chains, 3, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            tier, slot = divmod(i, ring_size)
            angle = 2.0 * math.pi * slot / ring_size + tier * stagger
            coms[i] = torch.tensor(
                [radius * math.cos(angle), radius * math.sin(angle), (tier - (tiers - 1) / 2) * dz],
                device=device,
                dtype=dtype,
            )
            rotations[i] = _rotation_z(angle, device=device, dtype=dtype)
    elif topology == "multi_helical":
        starts = int(params.get("num_starts", 3))
        if n_chains % starts:
            raise ValueError(f"multi_helical requires copies divisible by num_starts ({starts})")
        angle_step = math.radians(float(params.get("target_angle", 30.0)))
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        rotations = torch.empty((n_chains, 3, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            strand, step = i % starts, i // starts
            angle = strand * 2.0 * math.pi / starts + step * angle_step
            coms[i] = torch.tensor(
                [radius * math.cos(angle), radius * math.sin(angle), (step - (n_chains // starts - 1) / 2) * dz],
                device=device,
                dtype=dtype,
            )
            rotations[i] = _rotation_z(angle, device=device, dtype=dtype)
    elif topology in ("cage_tetrahedral", "cage_octahedral", "hexagonal_mesh"):
        # Evenly spread directions provide stable deterministic starting
        # layouts for these compact closed/surface lattices.
        cage_radius = float(params.get("target_cage_radius", 20.0))
        if topology == "hexagonal_mesh":
            gx, gy = int(params.get("grid_dim_x", 2)), int(params.get("grid_dim_y", 2))
            if n_chains != 3 * gx * gy:
                raise ValueError(f"hexagonal_mesh requires {3 * gx * gy} copies, got {n_chains}")
            lattice = params.get("lattice_constant")
            if lattice is None:
                pore = params.get("pore_diameter")
                if pore is None:
                    raise ValueError("hexagonal_mesh requires pore_diameter or lattice_constant")
                lattice = float(pore) + 10.0
            coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
            for i in range(n_chains):
                position, orientation = divmod(i, 3)
                x, y = divmod(position, gy)
                coms[i, 0] = x * float(lattice)
                coms[i, 1] = y * float(lattice) * math.sqrt(3.0) / 2.0
                coms[i, 2] = 0.0
            rotations = identity.expand(n_chains, -1, -1).clone()
            angles = torch.arange(n_chains, device=device, dtype=dtype) % 3 * (math.pi / 3.0)
            rotations = torch.stack([_rotation_z(float(a), device=device, dtype=dtype) for a in angles])
        else:
            angles = torch.arange(n_chains, device=device, dtype=dtype) * (math.pi * (3.0 - math.sqrt(5.0)))
            y = 1.0 - 2.0 * (torch.arange(n_chains, device=device, dtype=dtype) + 0.5) / n_chains
            radial = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
            coms = torch.stack((radial * torch.cos(angles), radial * torch.sin(angles), y), dim=-1) * cage_radius
            rotations = identity.expand(n_chains, -1, -1).clone()
    else:
        raise ValueError(f"Unsupported material topology: {topology}")

    return coms, rotations


def fold_to_consensus_frame(
    coords: torch.Tensor, ideal_com: torch.Tensor, ideal_rotation: torch.Tensor
) -> torch.Tensor:
    """Map coordinates from an ideal placement into the shared local frame."""
    return torch.matmul(coords - ideal_com.unsqueeze(-2), ideal_rotation)


def unfold_from_consensus_frame(
    coords: torch.Tensor, ideal_com: torch.Tensor, ideal_rotation: torch.Tensor
) -> torch.Tensor:
    """Map coordinates from the shared local frame into an ideal placement."""
    return torch.matmul(coords, ideal_rotation.transpose(-1, -2)) + ideal_com.unsqueeze(-2)


def build_layout_guidance_masks(
    atom_asym_id: torch.Tensor,
    layout: dict,
    topology: str,
    atom_valid_mask: torch.Tensor | None = None,
):
    """Build protein-only topology masks and ligand masks from builder metadata."""
    chain_records = {
        int(record["asym_index"]): record for record in layout.get("chains", [])
    }
    present_ids = {int(chain_id) for chain_id in torch.unique(atom_asym_id).tolist()}
    missing = {
        asym_index
        for asym_index, record in chain_records.items()
        if record["role"] == "protein" and asym_index not in present_ids
    }
    if missing:
        raise ValueError(
            "Material layout does not match feature chain order; "
            f"missing asym_id(s) {sorted(missing)}"
        )

    protein_masks = {}
    ligand_masks = []
    for asym_index, record in chain_records.items():
        mask = atom_asym_id == asym_index
        if atom_valid_mask is not None:
            mask &= atom_valid_mask.bool()
        if record["role"] == "protein":
            if not torch.any(mask):
                raise ValueError(
                    f"Material layout protein asym_index {asym_index} has no valid atoms"
                )
            protein_masks[asym_index] = mask
        elif torch.any(mask):
            ligand_masks.append((asym_index, record, mask))

    guidance_masks = []
    for placement in layout.get("placements", []):
        protein_indices = [int(index) for index in placement["protein_asym_indices"]]
        if not protein_indices or any(index not in protein_masks for index in protein_indices):
            raise ValueError(
                f"Placement {placement.get('placement_index')} has no valid guided protein chain"
            )
        if topology == "double_tape":
            guidance_masks.extend(protein_masks[index] for index in protein_indices)
        else:
            unit_mask = torch.zeros_like(atom_asym_id, dtype=torch.bool)
            for index in protein_indices:
                unit_mask |= protein_masks[index]
            guidance_masks.append(unit_mask)
    return guidance_masks, protein_masks, ligand_masks


def fit_rigid_transform(source: torch.Tensor, target: torch.Tensor):
    """Least-squares row-vector transform mapping source coordinates to target."""
    n = min(source.shape[0], target.shape[0])
    if n == 0:
        raise ValueError("Cannot fit a rigid transform to empty coordinates")
    source, target = source[:n], target[:n]
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _, vh = torch.linalg.svd(source_centered.T @ target_centered)
    determinant = torch.sign(torch.det(u @ vh))
    correction = torch.eye(3, device=source.device, dtype=source.dtype)
    correction[2, 2] = determinant
    rotation = u @ correction @ vh
    translation = target_mean - source_mean @ rotation
    return rotation, translation
