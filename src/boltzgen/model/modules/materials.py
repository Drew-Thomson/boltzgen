"""Ideal chain lattices and local-frame transforms for material guidance."""

from __future__ import annotations

from itertools import permutations, product
import math
from typing import Any

import torch


def _signed_permutation_rotations(
    *, tetrahedral: bool = False
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Generate the 12 tetrahedral or 24 octahedral signed permutation matrices."""
    rotations = []
    even_perms = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}
    for p in permutations(range(3)):
        if tetrahedral and p not in even_perms:
            continue
        parity = 1 if p in even_perms else -1
        for signs in product([-1, 1], repeat=3):
            if tetrahedral and (signs[0] * signs[1] * signs[2] != 1):
                continue
            det = parity * signs[0] * signs[1] * signs[2]
            if det == 1:
                mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                for i in range(3):
                    mat[i][p[i]] = signs[i]
                rotations.append(tuple(tuple(row) for row in mat))
    return tuple(rotations)


TETRAHEDRAL_ROTATIONS = _signed_permutation_rotations(tetrahedral=True)
OCTAHEDRAL_ROTATIONS = _signed_permutation_rotations(tetrahedral=False)


def _rotation_z(
    angle: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a row-vector-compatible rotation matrix about Z."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _ring_points(
    count: int, radius: float, *, device: torch.device, dtype: torch.dtype
):
    angles = torch.arange(count, device=device, dtype=dtype) * (2.0 * math.pi / count)
    points = torch.zeros((count, 3), device=device, dtype=dtype)
    points[:, 0] = radius * torch.cos(angles)
    points[:, 1] = radius * torch.sin(angles)
    rotations = torch.stack(
        [_rotation_z(float(angle), device=device, dtype=dtype) for angle in angles]
    )
    return points, rotations


def double_tape_layer_position(chain_index: int) -> tuple[int, int]:
    """Map emitted double-tape chain order to its layer and axial position.

    Emitted chains alternate between the two layers at each axial position:
    0=(layer 0, position 0), 1=(layer 1, position 0), 2=(layer 0, position 1), ...
    Keep this mapping shared with antiparallel guidance so chain orientation
    cannot accidentally use a different (serpentine) order from the lattice.
    """
    if chain_index < 0:
        raise ValueError("chain_index must be non-negative")
    return chain_index % 2, chain_index // 2


def double_tape_consensus_slot_indices_from_layout(layout: dict) -> list[int]:
    """Map chains to their consensus slot based on template index in layout."""
    slots = []
    seen = {}
    for chain in layout.get("chains", []):
        if chain.get("role") != "protein":
            continue
        t_idx = chain.get("protein_template_index")
        if t_idx is None:
            raise ValueError("All proteins must have a protein_template_index")
        if t_idx not in seen:
            seen[t_idx] = len(seen)
        slots.append(seen[t_idx])
    return slots


def double_tape_antiparallel_rotation(
    chain_index: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return local rotation matrix for antiparallel double tapes.

    Preserves antiparallel polarity both within sheets (along the tape axis) and
    across sheets at the same axial step, while inverting the transverse axis (x)
    in Layer 1 so that odd-numbered residues face outward into solvent and even-numbered
    residues face inward toward the bilayer core in both sheets.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    layer, step = double_tape_layer_position(chain_index)
    r_xx = 1.0 if layer == 0 else -1.0
    r_yy = -1.0 if (layer + step) % 2 == 1 else 1.0
    r_zz = r_xx * r_yy
    return torch.diag(
        torch.tensor([r_xx, r_yy, r_zz], device=device, dtype=dtype)
    )


def strand_direction(coords: torch.Tensor) -> torch.Tensor:
    """Unit N->C direction of a chain (last quarter mean - first quarter mean)."""
    k = max(1, coords.shape[0] // 4)
    v = coords[-k:].mean(dim=0) - coords[:k].mean(dim=0)
    return v / (v.norm() + 1e-8)


def lock_slot_orientations(slot_coords: list[torch.Tensor]) -> list[torch.Tensor]:
    """Rotate each consensus slot 180° about the layer axis (x) so that its strand
    direction matches slot 0. With the (layer+step) parity flip this makes adjacent
    different-partner strands in a sheet antiparallel."""
    flip = torch.diag(
        torch.tensor(
            [1.0, -1.0, -1.0], dtype=slot_coords[0].dtype, device=slot_coords[0].device
        )
    )
    ref = strand_direction(slot_coords[0])
    out = [slot_coords[0]]
    for coords in slot_coords[1:]:
        if torch.dot(strand_direction(coords), ref) < 0:
            out.append(torch.matmul(coords, flip))
        else:
            out.append(coords)
    return out


def generate_ideal_lattice(
    topology: str,
    n_chains: int,
    asym_unit_size: int,
    params: dict,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create target COMs and per-copy rotations for a material topology.

    Rotations map coordinates in the consensus/local frame into the ideal
    lattice frame. For tape topologies with axial supercoiling/twist angle per
    step $\Delta\theta$, step $s$ has rotation:
    $$\mathbf{R}_z(\theta_s) = \begin{bmatrix}
    \cos\theta_s & -\sin\theta_s & 0 \\
    \sin\theta_s & \cos\theta_s & 0 \\
    0 & 0 & 1
    \end{bmatrix}$$
    with $\theta_s = (s - \frac{S-1}{2}) \Delta\theta$.

    Parameters
    ----------
    topology : str
        Topology type (e.g. ``'double_tape'``, ``'linear_tape'``).
    n_chains : int
        Number of protein chain copies.
    asym_unit_size : int
        Number of chains in the asymmetric unit.
    params : dict
        Topology parameters including ``target_pitch``, ``layer_dist``,
        ``layer_offset_z``, ``target_twist``, and optional noise offsets.
    device : torch.device | None
        Torch device.
    dtype : torch.dtype | None
        Torch floating point dtype.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Target COMs with shape ``[n_chains, 3]`` and copy rotations with shape
        ``[n_chains, 3, 3]``.
    """
    if device is None:
        device = kwargs.get("device", torch.device("cpu"))
    if dtype is None:
        dtype = kwargs.get("dtype", torch.float32)

    if n_chains < 1:
        raise ValueError("n_chains must be positive")

    radius = float(params.get("target_radius", 15.0))
    dz = float(params.get("target_dz", 5.0))
    pitch = float(params.get("strand_spacing") or params.get("target_pitch") or 4.8)
    identity = torch.eye(3, device=device, dtype=dtype)

    if topology == "cyclic":
        coms, rotations = _ring_points(n_chains, radius, device=device, dtype=dtype)
    elif topology == "cylindrical":
        u_size = max(1, asym_unit_size)
        if n_chains % u_size != 0:
            msg = (
                f"cylindrical requires n_chains divisible by asym_unit_size "
                f"({u_size}), got {n_chains}"
            )
            raise ValueError(msg)
        n_placements = n_chains // u_size
        base_step = 2.0 * math.pi / n_placements
        intra_angle_step = base_step / u_size

        stagger_dz = (
            float(params.get("target_dz") or 0.0)
            if u_size > 1
            else float(params.get("target_dz") or 0.0)
        )
        stagger_z_noise = float(params.get("stagger_z_noise") or 0.0)
        stagger_phi_noise = float(params.get("stagger_phi_noise") or 0.0)

        effective_dz = stagger_dz + stagger_z_noise
        effective_dphi = intra_angle_step + stagger_phi_noise

        target_angle_deg = float(params.get("target_angle") or 0.0)
        target_angle_rad = math.radians(target_angle_deg)
        raw_slide = params.get("slide_vector")
        slide_vector: list[float] | None = None
        if raw_slide is not None:
            if isinstance(raw_slide, torch.Tensor):
                slide_vector = [float(v.item()) for v in raw_slide]
            elif isinstance(raw_slide, (list, tuple)):
                slide_vector = [float(v) for v in raw_slide]

        angles = torch.zeros(n_chains, device=device, dtype=dtype)
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            p = i // u_size
            u = i % u_size
            u_offset = (
                (u - (u_size - 1) / 2.0)
                if u_size > 1
                else ((i % 2) - 0.5 if effective_dz != 0.0 else 0.0)
            )
            phi = p * base_step + (u_offset * effective_dphi if u_size > 1 else 0.0)
            z = u_offset * effective_dz

            if slide_vector is not None:
                if u_size > 1 and u < len(slide_vector):
                    s_u = slide_vector[u]
                elif u_size == 1 and (i % 2) < len(slide_vector):
                    s_u = slide_vector[i % 2]
                else:
                    s_u = 0.0

                flip_sign = -1.0 if (i % 2 == 1 and target_angle_rad != 0.0) else 1.0
                alpha_u = flip_sign * target_angle_rad

                z = z + s_u * math.cos(alpha_u)
                phi = phi + (s_u * math.sin(alpha_u)) / radius

            angles[i] = phi
            coms[i, 0] = radius * math.cos(phi)
            coms[i, 1] = radius * math.sin(phi)
            coms[i, 2] = z
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "linear_tape":
        target_twist = float(params.get("target_twist", 0.0))
        twist_offset = float(params.get("twist_offset") or 0.0)
        step_angle = math.radians(target_twist + twist_offset)
        steps = torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2
        angles = steps * step_angle
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 2] = steps * pitch
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "helical":
        angles = torch.arange(n_chains, device=device, dtype=dtype) * math.radians(
            float(params.get("target_angle", 30.0))
        )
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 0] = radius * torch.cos(angles)
        coms[:, 1] = radius * torch.sin(angles)
        coms[:, 2] = (
            torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2
        ) * dz
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "open_arc":
        arc_radius = float(params.get("arc_radius", 100.0))
        spacing = float(params.get("target_arc_spacing", 10.0))
        step = 2.0 * math.asin(min(spacing / (2.0 * arc_radius), 1.0))
        angles = (
            torch.arange(n_chains, device=device, dtype=dtype) - (n_chains - 1) / 2
        ) * step
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 0] = arc_radius * torch.sin(angles)
        coms[:, 1] = arc_radius * (1.0 - torch.cos(angles))
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
        )
    elif topology == "double_tape":
        # Two offset rows, alternating chain indices between the layers.
        layer_dist = float(params.get("layer_dist", 10.0))
        if params.get("layer_dist_noise") is not None:
            layer_dist = max(4.8, layer_dist + float(params["layer_dist_noise"]))
        offset_y = float(params.get("layer_offset_y_noise") or 0.0)
        nominal_offset_z = params.get("layer_offset_z")
        if nominal_offset_z is None:
            nominal_offset_z = 0.5 * pitch
        else:
            nominal_offset_z = float(nominal_offset_z)
        offset_z_noise = float(params.get("layer_offset_z_noise") or 0.0)
        offset_z = nominal_offset_z + offset_z_noise
        target_twist = float(params.get("target_twist", 0.0))
        twist_offset = float(params.get("twist_offset") or 0.0)
        step_angle = math.radians(target_twist + twist_offset)
        units = math.ceil(n_chains / 2)
        chain_idx = torch.arange(n_chains, device=device)
        layers = chain_idx % 2
        steps = chain_idx // 2
        layer_signs = layers.to(dtype=dtype) - 0.5
        step_offsets = steps.to(dtype=dtype) - (units - 1) / 2
        x_u = layer_signs * layer_dist
        y_u = layer_signs * offset_y
        z_u = step_offsets * pitch + layer_signs * offset_z
        theta_s = step_offsets * step_angle
        cos_theta = torch.cos(theta_s)
        sin_theta = torch.sin(theta_s)
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        coms[:, 0] = x_u * cos_theta - y_u * sin_theta
        coms[:, 1] = x_u * sin_theta + y_u * cos_theta
        coms[:, 2] = z_u
        rotations = torch.stack(
            [_rotation_z(float(a), device=device, dtype=dtype) for a in theta_s]
        )
    elif topology == "bilayer_sheet":
        gx, gy = int(params.get("grid_dim_x", 2)), int(params.get("grid_dim_y", 2))
        if n_chains != 2 * gx * gy:
            raise ValueError(
                f"bilayer_sheet requires {2 * gx * gy} copies, got {n_chains}"
            )
        row_pitch = float(params.get("row_pitch", 10.0))
        layer_dist = float(params.get("layer_dist", 10.0))
        if params.get("layer_dist_noise") is not None:
            layer_dist = max(4.8, layer_dist + float(params["layer_dist_noise"]))
        offset_x = float(params.get("layer_offset_y_noise") or 0.0)
        offset_y = float(params.get("layer_offset_z_noise") or 0.0)
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            layer, position = divmod(i, gx * gy)
            x_idx, y_idx = divmod(position, gy)
            layer_sign = layer - 0.5
            coms[i] = torch.tensor(
                [
                    layer_sign * layer_dist,
                    (x_idx - (gx - 1) / 2) * row_pitch + layer_sign * offset_x,
                    (y_idx - (gy - 1) / 2) * pitch + layer_sign * offset_y,
                ],
                device=device,
                dtype=dtype,
            )
        rotations = identity.expand(n_chains, -1, -1).clone()
    elif topology == "nanotube":
        ring_size = int(params.get("ring_size", 4))
        tiers = int(params.get("num_tiers", 4))
        if n_chains != ring_size * tiers:
            raise ValueError(
                f"nanotube requires {ring_size * tiers} copies, got {n_chains}"
            )
        stagger = math.radians(float(params.get("chiral_stagger", 0.0)))
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        rotations = torch.empty((n_chains, 3, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            tier, slot = divmod(i, ring_size)
            angle = 2.0 * math.pi * slot / ring_size + tier * stagger
            coms[i] = torch.tensor(
                [
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    (tier - (tiers - 1) / 2) * dz,
                ],
                device=device,
                dtype=dtype,
            )
            rotations[i] = _rotation_z(angle, device=device, dtype=dtype)
    elif topology == "multi_helical":
        starts = int(params.get("num_starts", 3))
        if n_chains % starts:
            raise ValueError(
                f"multi_helical requires copies divisible by num_starts ({starts})"
            )
        angle_step = math.radians(float(params.get("target_angle", 30.0)))
        coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
        rotations = torch.empty((n_chains, 3, 3), device=device, dtype=dtype)
        for i in range(n_chains):
            strand, step = i % starts, i // starts
            angle = strand * 2.0 * math.pi / starts + step * angle_step
            coms[i] = torch.tensor(
                [
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    (step - (n_chains // starts - 1) / 2) * dz,
                ],
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
                raise ValueError(
                    f"hexagonal_mesh requires {3 * gx * gy} copies, got {n_chains}"
                )
            lattice = params.get("lattice_constant")
            if lattice is None:
                pore = params.get("pore_diameter")
                if pore is None:
                    raise ValueError(
                        "hexagonal_mesh requires pore_diameter or lattice_constant"
                    )
                lattice = float(pore) + 10.0
            coms = torch.zeros((n_chains, 3), device=device, dtype=dtype)
            for i in range(n_chains):
                position, orientation = divmod(i, 3)
                x, y = divmod(position, gy)
                coms[i, 0] = x * float(lattice)
                coms[i, 1] = y * float(lattice) * math.sqrt(3.0) / 2.0
                coms[i, 2] = 0.0
            rotations = identity.expand(n_chains, -1, -1).clone()
            angles = (
                torch.arange(n_chains, device=device, dtype=dtype) % 3 * (math.pi / 3.0)
            )
            rotations = torch.stack(
                [_rotation_z(float(a), device=device, dtype=dtype) for a in angles]
            )
        else:
            angles = torch.arange(n_chains, device=device, dtype=dtype) * (
                math.pi * (3.0 - math.sqrt(5.0))
            )
            y = (
                1.0
                - 2.0
                * (torch.arange(n_chains, device=device, dtype=dtype) + 0.5)
                / n_chains
            )
            radial = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
            coms = (
                torch.stack(
                    (radial * torch.cos(angles), radial * torch.sin(angles), y), dim=-1
                )
                * cage_radius
            )
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
    return torch.matmul(coords, ideal_rotation.transpose(-1, -2)) + ideal_com.unsqueeze(
        -2
    )


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
                msg = (
                    f"Material layout protein asym_index {asym_index} "
                    "has no valid atoms"
                )
                raise ValueError(msg)
            protein_masks[asym_index] = mask
        elif torch.any(mask):
            ligand_masks.append((asym_index, record, mask))

    guidance_masks = []
    for placement in layout.get("placements", []):
        protein_indices = [int(index) for index in placement["protein_asym_indices"]]
        if not protein_indices or any(
            index not in protein_masks for index in protein_indices
        ):
            msg = (
                f"Placement {placement.get('placement_index')} has no valid guided "
                "protein chain"
            )
            raise ValueError(msg)
        if topology in ("double_tape", "cylindrical"):
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


def resolve_anchor_residue_index(
    spec: Any,
    seq_len: int,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> int:
    r"""Resolve a 0-indexed anchor residue from a user spec or uniform random sample.

    When ``spec`` is an integer or numeric string, it represents a 1-indexed residue
    position ($k \in [1, L]$), which is converted to 0-indexed and clamped to $[0, L - 1]$.
    When ``spec`` is ``'random'`` or unspecified, a position is drawn uniformly from
    the internal residues $[2, L - 3]$ (for $L > 4$) to avoid terminal unfolding.

    Parameters
    ----------
    spec : Any
        Residue identifier: 1-indexed int, numeric string, or ``'random'``.
    seq_len : int
        Total number of residues in the parent protein chain ($L$).
    device : torch.device | None
        Torch device for random sampling.
    generator : torch.Generator | None
        Optional PyTorch generator for reproducible sampling.

    Returns
    -------
    int
        0-indexed residue position in $[0, L - 1]$.
    """
    if seq_len <= 0:
        return 0
    if spec is not None and str(spec).strip().lower() not in ("random", "none"):
        try:
            val = int(spec)
            return max(0, min(val - 1, seq_len - 1))
        except (ValueError, TypeError):
            pass

    if seq_len > 4:
        low = 2
        high = seq_len - 3
    else:
        low = 0
        high = seq_len - 1

    if device is None:
        device = torch.device("cpu")
    sample = torch.randint(low, high + 1, (1,), device=device, generator=generator)
    return int(sample.item())


def compute_anchored_ligand_position(
    ca_coords: torch.Tensor,
    assembly_com: torch.Tensor,
    topology: str,
    anchor_distance: float = 4.0,
    r_kabsch: torch.Tensor | None = None,
    sidechain_vec: torch.Tensor | None = None,
    direction_mode: str = "outward",
) -> torch.Tensor:
    r"""Compute target position for a ligand anchored to a specific residue C_alpha.

    Calculates the spatial anchor position avoiding backbone penetration:
    $$\mathbf{x}_{\text{target}} = \mathbf{x}_{\text{C}_\alpha} + d_{\text{anchor}} \hat{\mathbf{u}}$$
    When sidechain vector $\mathbf{v}_{\text{sc}}$ is available, $\hat{\mathbf{u}}$ is directed
    away from the peptide backbone into solvent/interface space, preventing embedding.

    Parameters
    ----------
    ca_coords : torch.Tensor
        Guided C_alpha coordinates of shape ``(3,)``.
    assembly_com : torch.Tensor
        Center of mass of the complex of shape ``(3,)``.
    topology : str
        Assembly topology (e.g. ``'cyclic'``, ``'cylindrical'``, ``'double_tape'``).
    anchor_distance : float
        Target distance from C_alpha in Angstroms (default: 4.0).
    r_kabsch : torch.Tensor | None
        Kabsch rotation matrix mapping local frame to global frame.
    sidechain_vec : torch.Tensor | None
        Optional unnormalized vector pointing in the C_alpha -> C_beta direction.
    direction_mode : str
        Direction mode: ``'outward'`` (or ``'radial'``), ``'sidechain'``, or ``'auto'``.

    Returns
    -------
    torch.Tensor
        Target ligand position of shape ``(3,)``.
    """
    disp = ca_coords - assembly_com
    if topology in ("cage_tetrahedral", "cage_octahedral", "floating"):
        norm_disp = torch.norm(disp)
        u_radial = (
            disp / (norm_disp + 1e-6)
            if norm_disp > 1e-4
            else torch.tensor([1.0, 0.0, 0.0], device=ca_coords.device, dtype=ca_coords.dtype)
        )
    else:
        if r_kabsch is not None and r_kabsch.shape == (3, 3):
            v_axis = r_kabsch[:, 2]
        else:
            v_axis = torch.tensor([0.0, 0.0, 1.0], device=ca_coords.device, dtype=ca_coords.dtype)
        v_along_axis = torch.dot(disp, v_axis) * v_axis
        v_perp = disp - v_along_axis
        norm_perp = torch.norm(v_perp)
        if norm_perp > 1e-4:
            u_radial = v_perp / norm_perp
        else:
            norm_disp = torch.norm(disp)
            u_radial = (
                disp / (norm_disp + 1e-6)
                if norm_disp > 1e-4
                else torch.tensor([1.0, 0.0, 0.0], device=ca_coords.device, dtype=ca_coords.dtype)
            )

    mode = str(direction_mode).strip().lower()
    if sidechain_vec is not None and torch.norm(sidechain_vec) > 1e-4:
        u_sc = sidechain_vec / torch.norm(sidechain_vec)
        if mode == "sidechain":
            u_dir = u_sc
        else:
            # For 'outward' or 'auto', ensure displacement does not pass back into the helix backbone.
            cos_sim = torch.dot(u_sc, u_radial)
            if cos_sim >= 0.0:
                v_blend = u_sc + 0.4 * u_radial
                u_dir = v_blend / torch.norm(v_blend)
            else:
                # Inward-facing sidechain: project along the interface tangent perpendicular to u_radial
                v_tangent = u_sc - cos_sim * u_radial
                norm_tangent = torch.norm(v_tangent)
                if norm_tangent > 1e-4:
                    u_dir = v_tangent / norm_tangent
                else:
                    u_dir = u_sc
    else:
        u_dir = u_radial

    return ca_coords + float(anchor_distance) * u_dir


def resolve_protein_metal_clashes(
    metal_pos: torch.Tensor,
    protein_coords: torch.Tensor,
    assembly_com: torch.Tensor | None = None,
    min_distance: float = 2.8,
    max_iterations: int = 8,
    axis: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Iteratively repel metal position away from protein atoms to prevent embedding.

    If ``axis`` is provided, repulsions are constrained strictly along the axis:
    $$\Delta\mathbf{x} \parallel \hat{\mathbf{v}}_{\text{axis}}, \quad \Delta\mathbf{x} \cdot \mathbf{r}_\perp = 0$$
    ensuring axial ligands remain on the symmetry axis without transverse radial displacement.
    If ``axis`` is None, pushes outward in 3D:
    $$\mathbf{x}_{\text{metal}} \leftarrow \mathbf{x}_{\text{metal}} + \sum_{j \in \text{clash}} (d_{\min} - d_j) \frac{\mathbf{x}_{\text{metal}} - \mathbf{x}_j}{d_j}$$

    Parameters
    ----------
    metal_pos : torch.Tensor
        Proposed metal position of shape ``(3,)`` or ``(M, 3)``.
    protein_coords : torch.Tensor
        Coordinates of all atoms in the protein chain of shape ``(N, 3)``.
    assembly_com : torch.Tensor | None
        Optional assembly center of mass for breaking saddle points.
    min_distance : float
        Minimum clearance distance in Angstroms (default: 2.8).
    max_iterations : int
        Maximum push-out iterations.
    axis : torch.Tensor | None
        Optional symmetry unit axis of shape ``(3,)`` for 1D axial clash resolution.

    Returns
    -------
    torch.Tensor
        Repelled metal position free of steric overlap.
    """
    pos = metal_pos.clone()
    is_1d = pos.dim() == 1
    if is_1d:
        pos = pos.unsqueeze(0)

    u_axis = None
    if axis is not None:
        ax_norm = axis.norm()
        if ax_norm > 1e-6:
            u_axis = (axis / ax_norm).to(device=pos.device, dtype=pos.dtype)

    for _ in range(max_iterations):
        dists = torch.cdist(pos, protein_coords)
        clash_mask = dists < min_distance
        if not clash_mask.any():
            break
        for m_idx in range(pos.shape[0]):
            clashing = clash_mask[m_idx].nonzero(as_tuple=True)[0]
            if len(clashing) == 0:
                continue

            if u_axis is not None:
                # 1D Axial clash resolution: displace strictly parallel to u_axis
                pos_shift = 0.0
                neg_shift = 0.0
                ref_com = assembly_com if assembly_com is not None else protein_coords.mean(dim=0)
                for at_idx in clashing:
                    diff = pos[m_idx] - protein_coords[at_idx]
                    dz = torch.dot(diff, u_axis).item()
                    r_perp = diff - dz * u_axis
                    rho = r_perp.norm().item()
                    if rho < min_distance:
                        req_dz = math.sqrt(max(0.0, min_distance**2 - rho**2))
                        deficit = req_dz - abs(dz)
                        if deficit > 0.0:
                            if dz > 1e-4:
                                pos_shift = max(pos_shift, deficit)
                            elif dz < -1e-4:
                                neg_shift = max(neg_shift, deficit)
                            else:
                                com_dz = torch.dot(pos[m_idx] - ref_com, u_axis).item()
                                if com_dz >= 0.0:
                                    pos_shift = max(pos_shift, deficit)
                                else:
                                    neg_shift = max(neg_shift, deficit)

                if pos_shift > 0.0 and neg_shift > 0.0:
                    total_dz = pos_shift if pos_shift < neg_shift else -neg_shift
                elif pos_shift > 0.0:
                    total_dz = pos_shift
                elif neg_shift > 0.0:
                    total_dz = -neg_shift
                else:
                    total_dz = 0.0

                pos[m_idx] = pos[m_idx] + total_dz * u_axis
            else:
                push = torch.zeros(3, device=pos.device, dtype=pos.dtype)
                max_deficit = 0.0
                for at_idx in clashing:
                    diff = pos[m_idx] - protein_coords[at_idx]
                    d = dists[m_idx, at_idx].item()
                    if d > 1e-4:
                        deficit = min_distance - d
                        max_deficit = max(max_deficit, deficit)
                        push = push + (diff / d) * deficit
                push_norm = push.norm()
                if push_norm > 1e-4 and max_deficit > 0.0:
                    # Clamp step size to max penetration deficit to prevent multi-atom push explosion
                    step_size = min(push_norm.item(), max_deficit * 1.1)
                    push = (push / push_norm) * step_size
                elif push_norm < 1e-4:
                    ref_com = assembly_com if assembly_com is not None else protein_coords.mean(dim=0)
                    out_vec = pos[m_idx] - ref_com
                    out_norm = out_vec.norm()
                    if out_norm > 1e-4:
                        push = (out_vec / out_norm) * (min_distance - dists[m_idx].min().item())
                    else:
                        push = torch.tensor([1.0, 0.0, 0.0], device=pos.device, dtype=pos.dtype) * min_distance
                pos[m_idx] = pos[m_idx] + push
    return pos.squeeze(0) if is_1d else pos


def sample_stratified_axial_positions(
    count: int,
    z_range: tuple[float, float] | list[float] | float = 5.0,
    min_distance: float = 3.5,
    symmetric: bool = False,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> list[float]:
    r"""Sample randomly evenly distributed axial positions along the z-axis.

    Parameters
    ----------
    count : int
        Number of axial positions to sample ($K \ge 1$).
    z_range : tuple[float, float] | list[float] | float
        Either [z_min, z_max] or scalar h meaning [-h, h]. Default: 5.0.
    min_distance : float
        Minimum separation distance along the axis between consecutive positions.
    symmetric : bool
        If True and count == 2, samples z > 0 and pairs it with -z to preserve
        dihedral/inversion symmetry.
    generator : torch.Generator | None
        Optional PyTorch random generator for reproducibility.
    device : torch.device | None
        PyTorch device.

    Returns
    -------
    list[float]
        Sorted list of sampled axial z coordinates.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    if isinstance(z_range, (int, float)):
        z_min, z_max = -abs(float(z_range)), abs(float(z_range))
    else:
        z_min, z_max = float(z_range[0]), float(z_range[1])

    if z_min > z_max:
        z_min, z_max = z_max, z_min

    if count == 1:
        if symmetric or math.isclose(z_min, -z_max, abs_tol=1e-5):
            return [0.0]
        u = torch.rand(1, generator=generator, device=device).item()
        return [z_min + u * (z_max - z_min)]

    if symmetric and count == 2:
        half_span = z_max
        low = max(min_distance / 2.0, 0.5)
        high = max(low + 0.1, half_span)
        u = torch.rand(1, generator=generator, device=device).item()
        z = low + u * (high - low)
        return [-float(z), float(z)]

    span = z_max - z_min
    bin_w = span / count
    positions: list[float] = []

    for i in range(count):
        bin_start = z_min + i * bin_w
        bin_end = bin_start + bin_w
        margin = min(0.2 * bin_w, 0.5)
        low = bin_start + margin
        high = bin_end - margin
        if high <= low:
            low, high = bin_start, bin_end
        u = torch.rand(1, generator=generator, device=device).item()
        z = low + u * (high - low)
        positions.append(float(z))

    positions.sort()

    for i in range(1, count):
        if positions[i] - positions[i - 1] < min_distance:
            positions[i] = positions[i - 1] + min_distance

    if positions[-1] > z_max:
        overflow = positions[-1] - z_max
        for i in range(count):
            positions[i] -= overflow

    return [float(p) for p in positions]


def enforce_minimum_axial_separation(
    positions: torch.Tensor,
    axis: torch.Tensor,
    min_distance: float = 3.5,
) -> torch.Tensor:
    r"""Enforce minimum separation between distinct ligands along the symmetry axis.

    Parameters
    ----------
    positions : torch.Tensor
        Ligand center positions of shape ``(K, 3)``.
    axis : torch.Tensor
        Unit symmetry axis vector of shape ``(3,)``.
    min_distance : float
        Minimum required axial distance between distinct ligands.

    Returns
    -------
    torch.Tensor
        Adjusted positions of shape ``(K, 3)`` where $|z_i - z_{i-1}| \ge \text{min\_distance}$.
    """
    if positions.shape[0] <= 1:
        return positions

    u_axis = axis / (axis.norm() + 1e-8)
    pos = positions.clone()
    z_coords = torch.matmul(pos, u_axis)
    order = torch.argsort(z_coords)
    k = positions.shape[0]

    for _ in range(8):
        violated = False
        for idx in range(1, k):
            prev_idx = order[idx - 1]
            curr_idx = order[idx]
            dz = z_coords[curr_idx] - z_coords[prev_idx]
            if dz < min_distance:
                violated = True
                deficit = min_distance - dz
                shift = deficit / 2.0
                z_coords[prev_idx] -= shift
                z_coords[curr_idx] += shift
        if not violated:
            break

    perp = pos - torch.outer(torch.matmul(pos, u_axis), u_axis)
    return perp + torch.outer(z_coords, u_axis)


def compute_element_trace(
    ca_coords: torch.Tensor,
    secondary_structure: str | None = None,
) -> torch.Tensor:
    r"""Extract helical axis or strand trace from CA coordinates.

    For an $\alpha$-helix, CA atoms spiral around the central helical axis with
    radius $r_1 \approx 2.3\,\text{\AA}$ and $3.6$ residues per turn. A rolling
    4-residue filter filters out the helical oscillation to yield the exact central
    helical axis (Crick parameterization):
    $$\mathbf{t}_k = \frac{1}{4} \sum_{j=-1}^{2} \mathbf{c}_{k+j}$$
    For a $\beta$-strand, a rolling 2-residue filter smooths pleating:
    $$\mathbf{t}_k = \frac{1}{2} (\mathbf{c}_k + \mathbf{c}_{k+1})$$

    Parameters
    ----------
    ca_coords : torch.Tensor
        Residue CA coordinates with shape ``[L, 3]``.
    secondary_structure : str | None
        Secondary structure designation ('H' for helix, 'S' for sheet).
        If None or 'H', uses a 4-residue rolling filter.
        If 'S', uses a 2-residue rolling filter.

    Returns
    -------
    torch.Tensor
        Centerline trace coordinates with shape ``[L, 3]``.
    """
    l_res = ca_coords.shape[0]
    if l_res <= 1:
        return ca_coords.clone()

    ss_char = (secondary_structure or "H").upper()
    window = 2 if "S" in ss_char and "H" not in ss_char else min(4, l_res)

    if window <= 1 or l_res < window:
        return ca_coords.clone()

    # ca_coords: [L, 3] -> [3, 1, L]
    ca_t = ca_coords.T.unsqueeze(1)
    pad_left = (window - 1) // 2
    pad_right = window // 2
    padded = torch.nn.functional.pad(ca_t, (pad_left, pad_right), mode="replicate")
    kernel = torch.ones(
        (1, 1, window), device=ca_coords.device, dtype=ca_coords.dtype
    ) / float(window)
    trace_t = torch.nn.functional.conv1d(padded, kernel)
    return trace_t.squeeze(1).T


def align_consensus_to_cylinder(ref_coords: torch.Tensor) -> torch.Tensor:
    r"""Rigidly place the consensus element axis tangent to the cylinder.

    With the monomer centred on the local frame origin (radial $\hat{x}$, tangent
    $\hat{y}$, cylinder axis $\hat{z}$), the element axis $\mathbf{u}$ (PCA primary
    axis, signed N$\to$C) is rotated by the *minimal* rotation onto

    $$\mathbf{v} = \frac{(0, u_y, u_z)}{\lVert (0, u_y, u_z) \rVert},$$

    i.e. radial component removed. Crossing angle
    $\alpha = \arctan2(u_y, u_z)$, N$\to$C sense and roll about the axis are
    preserved. After placement at radius $R$ the axis is tangent to the cylinder
    (Crick parameterization); for $\alpha = 0$ it lies exactly on the surface.
    Pure rigid rotation: all internal geometry is invariant.

    Parameters
    ----------
    ref_coords : torch.Tensor
        Consensus monomer atom coordinates, shape ``[N, 3]``.

    Returns
    -------
    torch.Tensor
        Centred, rotated coordinates, shape ``[N, 3]``.
    """
    min_atoms = 3
    eps = 1e-3
    if ref_coords.shape[0] < min_atoms:
        return ref_coords.clone()

    centered = ref_coords - ref_coords.mean(dim=0)
    u_mat, _, _ = torch.linalg.svd(centered.T @ centered)
    u_axis = u_mat[:, 0] * torch.sign(
        torch.dot(strand_direction(centered), u_mat[:, 0]) + 1e-6
    )

    v_axis = torch.stack([torch.zeros_like(u_axis[0]), u_axis[1], u_axis[2]])
    if v_axis.norm() < eps:
        # Axis (near) radial: fall back to cylinder axis, keep N->C sense.
        v_axis = torch.zeros_like(u_axis)
        v_axis[2] = torch.sign(u_axis[2] + 1e-6)
    v_axis = v_axis / v_axis.norm()

    # Rodrigues: R = I + [k]_x + [k]_x^2 / (1 + c), k = u x v, c = u . v
    k_vec = torch.linalg.cross(u_axis, v_axis)
    cos_t = torch.dot(u_axis, v_axis)
    k_x = torch.zeros((3, 3), device=u_axis.device, dtype=u_axis.dtype)
    k_x[0, 1], k_x[0, 2] = -k_vec[2], k_vec[1]
    k_x[1, 0], k_x[1, 2] = k_vec[2], -k_vec[0]
    k_x[2, 0], k_x[2, 1] = -k_vec[1], k_vec[0]
    rot = (
        torch.eye(3, device=u_axis.device, dtype=u_axis.dtype)
        + k_x
        + (k_x @ k_x) / (1.0 + cos_t)
    )
    return torch.matmul(centered, rot.T)


def beta_strand_pleating_direction(
    coords: torch.Tensor,
    bb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Return transverse unit vector pointing towards odd-numbered sidechains.

    For a beta-strand, odd residues (1, 3, 5, ...) pleat towards one side of the
    sheet, and even residues (2, 4, ...) pleat towards the opposite side.
    Extracts C-beta vectors from backbone atoms (N, CA, C) if available, or CA pleating
    vectors $\mathbf{p}_i = \mathbf{c}_i - \frac{1}{2}(\mathbf{c}_{i-1} +
    \mathbf{c}_{i+1})$. Returns a unit vector pointing along the average odd sidechain
    direction.

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates of a peptide chain with shape ``[N, 3]``.
    bb_mask : torch.Tensor | None, default=None
        Boolean mask of shape ``[N]`` indicating backbone atoms (N, CA, C, O).
        If provided, only backbone atoms are used to compute the pleating
        direction.

    Returns
    -------
    torch.Tensor
        Unit vector with shape ``[3]`` pointing towards odd-numbered residue sidechains.
    """
    device = coords.device
    dtype = coords.dtype

    if bb_mask is not None:
        coords_bb = coords[bb_mask.bool()]
    else:
        coords_bb = coords

    n_bb = coords_bb.shape[0]

    is_pure_bb_4 = (
        n_bb >= 8
        and n_bb % 4 == 0
        and (
            bb_mask is not None
            or (coords_bb[1:] - coords_bb[:-1]).norm(dim=-1).mean() < 2.0
        )
    )

    if is_pure_bb_4:
        l_res = n_bb // 4
        bb = coords_bb.view(l_res, 4, 3)
        n_pts = bb[:, 0]
        ca_pts = bb[:, 1]
        c_pts = bb[:, 2]

        b = ca_pts - n_pts
        c_vec = c_pts - ca_pts
        b_norm = torch.linalg.norm(b - c_vec, dim=-1, keepdim=True) + 1e-8
        cross = torch.linalg.cross(b, c_vec)
        c_norm = torch.linalg.norm(cross, dim=-1, keepdim=True) + 1e-8

        vecs = -0.531018 * ((b - c_vec) / b_norm) + 1.20673 * (cross / c_norm)
        vecs = vecs / (torch.linalg.norm(vecs, dim=-1, keepdim=True) + 1e-8)

        odd_mean = vecs[0::2].mean(dim=0)
        even_mean = vecs[1::2].mean(dim=0)
        v_pleat = odd_mean - even_mean
    elif n_bb >= 3:
        p = coords_bb[1:-1] - 0.5 * (coords_bb[:-2] + coords_bb[2:])
        odd_p = p[1::2]
        even_p = p[0::2]
        if odd_p.shape[0] > 0 and even_p.shape[0] > 0:
            v_pleat = odd_p.mean(dim=0) - even_p.mean(dim=0)
        elif odd_p.shape[0] > 0:
            v_pleat = odd_p.mean(dim=0)
        elif even_p.shape[0] > 0:
            v_pleat = -even_p.mean(dim=0)
        else:
            v_pleat = torch.zeros(3, device=device, dtype=dtype)
    else:
        v_pleat = torch.zeros(3, device=device, dtype=dtype)

    norm = torch.linalg.norm(v_pleat)
    if norm > 1e-4:
        return v_pleat / norm
    return torch.zeros(3, device=device, dtype=dtype)


def align_consensus_to_double_tape(
    ref_coords: torch.Tensor,
    bb_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Rigidly align the consensus monomer element axis with the in-sheet axis (+Y).

    Centres the monomer at the origin, aligns its primary PCA axis $\mathbf{u}$
    (signed N$\to$C) with $\hat{y} = (0, 1, 0)$ via minimal Rodrigues rotation,
    and rolls about $Y$ so that backbone amide bonds lie coplanar with the
    in-sheet lateral axis ($\pm\hat{z}$), while odd-numbered residues face along
    $-\hat{x}$ (outward in Layer 0) and even-numbered residues face along $+\hat{x}$
    (inward toward the bilayer core).

    Parameters
    ----------
    ref_coords : torch.Tensor
        Consensus monomer atom coordinates, shape ``[N, 3]``.
    bb_mask : torch.Tensor | None, default=None
        Boolean mask of shape ``[N]`` indicating backbone atoms (N, CA, C, O).
        If provided, backbone geometry is used for amide and pleating roll
        alignment.

    Returns
    -------
    torch.Tensor
        Centred, aligned coordinates, shape ``[N, 3]``.
    """
    min_atoms = 3
    eps = 1e-6
    if ref_coords.shape[0] < min_atoms:
        return ref_coords - ref_coords.mean(dim=0)

    centered = ref_coords - ref_coords.mean(dim=0)
    u_mat, _, _ = torch.linalg.svd(centered.T @ centered)
    u_axis = u_mat[:, 0] * torch.sign(
        torch.dot(strand_direction(centered), u_mat[:, 0]) + 1e-6
    )

    v_axis = torch.tensor([0.0, 1.0, 0.0], device=u_axis.device, dtype=u_axis.dtype)
    k_vec = torch.linalg.cross(u_axis, v_axis)
    cos_t = torch.dot(u_axis, v_axis)

    if k_vec.norm() < eps or (1.0 + cos_t) < 1e-6:
        if cos_t < 0:
            rot = torch.diag(
                torch.tensor(
                    [1.0, -1.0, -1.0], device=u_axis.device, dtype=u_axis.dtype
                )
            )
        else:
            rot = torch.eye(3, device=u_axis.device, dtype=u_axis.dtype)
    else:
        k_x = torch.zeros((3, 3), device=u_axis.device, dtype=u_axis.dtype)
        k_x[0, 1], k_x[0, 2] = -k_vec[2], k_vec[1]
        k_x[1, 0], k_x[1, 2] = k_vec[2], -k_vec[0]
        k_x[2, 0], k_x[2, 1] = -k_vec[1], k_vec[0]
        rot = (
            torch.eye(3, device=u_axis.device, dtype=u_axis.dtype)
            + k_x
            + (k_x @ k_x) / (1.0 + cos_t)
        )
    aligned = torch.matmul(centered, rot.T)

    if bb_mask is not None:
        aligned_bb = aligned[bb_mask.bool()]
    else:
        aligned_bb = aligned

    n_bb = aligned_bb.shape[0]
    has_amide = False
    theta_co = None

    is_pure_bb_4 = (
        n_bb >= 8
        and n_bb % 4 == 0
        and (
            bb_mask is not None
            or (aligned_bb[1:] - aligned_bb[:-1]).norm(dim=-1).mean() < 2.0
        )
    )

    if is_pure_bb_4:
        l_res = n_bb // 4
        bb = aligned_bb.view(l_res, 4, 3)
        c_pts = bb[:, 2]
        o_pts = bb[:, 3]
        co = o_pts - c_pts
        co = co / (torch.linalg.norm(co, dim=-1, keepdim=True) + 1e-8)

        signs = torch.tensor(
            [(-1.0) ** i for i in range(l_res)],
            device=aligned.device,
            dtype=aligned.dtype,
        ).unsqueeze(1)
        co_alt = co * signs
        co_xz = co_alt[:, [0, 2]]
        mean_co_xz = co_xz.mean(dim=0)
        norm_co = torch.linalg.norm(mean_co_xz)
        if norm_co > 1e-4:
            mean_co_xz = mean_co_xz / norm_co
            theta_co = torch.atan2(mean_co_xz[0], mean_co_xz[1])
            has_amide = True
    elif (
        bb_mask is None
        and n_bb >= 9
        and n_bb % 3 == 0
        and (aligned_bb[1:] - aligned_bb[:-1]).norm(dim=-1).mean() < 2.0
    ):
        l_res = n_bb // 3
        bb = aligned_bb.view(l_res, 3, 3)
        ca_pts = bb[:-1, 1]
        c_pts = bb[:-1, 2]
        next_n_pts = bb[1:, 0]
        c_ca = ca_pts - c_pts
        c_ca = c_ca / (torch.linalg.norm(c_ca, dim=-1, keepdim=True) + 1e-8)
        c_n = next_n_pts - c_pts
        c_n = c_n / (torch.linalg.norm(c_n, dim=-1, keepdim=True) + 1e-8)
        co_est = -(c_ca + c_n)
        co_est = co_est / (torch.linalg.norm(co_est, dim=-1, keepdim=True) + 1e-8)

        signs = torch.tensor(
            [(-1.0) ** i for i in range(l_res - 1)],
            device=aligned.device,
            dtype=aligned.dtype,
        ).unsqueeze(1)
        co_alt = co_est * signs
        co_xz = co_alt[:, [0, 2]]
        mean_co_xz = co_xz.mean(dim=0)
        norm_co = torch.linalg.norm(mean_co_xz)
        if norm_co > 1e-4:
            mean_co_xz = mean_co_xz / norm_co
            theta_co = torch.atan2(mean_co_xz[0], mean_co_xz[1])
            has_amide = True

    if has_amide and theta_co is not None:
        v_pleat = beta_strand_pleating_direction(aligned, bb_mask=bb_mask)
        pi_val = torch.acos(
            torch.tensor(-1.0, device=aligned.device, dtype=aligned.dtype)
        )
        candidate_angles = [-theta_co, -theta_co + pi_val]
        best_angle = candidate_angles[0]
        for candidate_angle in candidate_angles:
            cos_a = torch.cos(candidate_angle)
            sin_a = torch.sin(candidate_angle)
            v_pleat_x = cos_a * v_pleat[0] + sin_a * v_pleat[2]
            if v_pleat_x < 0:
                best_angle = candidate_angle
                break

        cos_r = torch.cos(best_angle)
        sin_r = torch.sin(best_angle)
        r_roll = torch.tensor(
            [[cos_r, 0.0, sin_r], [0.0, 1.0, 0.0], [-sin_r, 0.0, cos_r]],
            device=aligned.device,
            dtype=aligned.dtype,
        )
        aligned = torch.matmul(aligned, r_roll.T)
    else:
        v_pleat = beta_strand_pleating_direction(aligned, bb_mask=bb_mask)
        r_xz = torch.sqrt(v_pleat[0] ** 2 + v_pleat[2] ** 2)
        if r_xz > 1e-4:
            u_x = v_pleat[0] / r_xz
            u_z = v_pleat[2] / r_xz
            r_roll = torch.tensor(
                [[-u_x, 0.0, -u_z], [0.0, 1.0, 0.0], [u_z, 0.0, -u_x]],
                device=aligned.device,
                dtype=aligned.dtype,
            )
            aligned = torch.matmul(aligned, r_roll.T)

    return aligned


def align_consensus_to_linear_tape(ref_coords: torch.Tensor) -> torch.Tensor:
    r"""Rigidly align the consensus monomer element axis with the tape strand axis (+Y).

    Centres the monomer at the origin and aligns its primary PCA axis $\mathbf{u}$
    (signed N$\to$C) with $\hat{y} = (0, 1, 0)$ via minimal Rodrigues rotation:

    $$\mathbf{k} = \mathbf{u} \times \hat{y}, \quad c = \mathbf{u} \cdot \hat{y}$$

    This places the $\beta$-strand backbone parallel to the strand direction ($Y$),
    perpendicular to the tape growth axis ($Z$), preserving internal monomer geometry.

    Parameters
    ----------
    ref_coords : torch.Tensor
        Consensus monomer atom coordinates, shape ``[N, 3]``.

    Returns
    -------
    torch.Tensor
        Centred, aligned coordinates, shape ``[N, 3]``.
    """
    return align_consensus_to_double_tape(ref_coords)


def wrap_consensus_to_cylinder(
    coords: torch.Tensor,
    ideal_com: torch.Tensor,
    radius: float,
    *,
    atom_token_ids: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Wrap flat consensus element coordinates onto the cylinder surface.

    Places the secondary structural element axis along a geodesic on the cylinder
    $\rho = R$ (a straight line on the developed/unrolled cylinder surface), with
    azimuth $\phi(s) = \phi_0 + \frac{s \sin\alpha}{R}$ and axial height
    $z(s) = z_0 + s \cos\alpha$. Each residue is rotated rigidly by its local
    azimuth $\phi_k$, preserving 100% of intra-residue 3D geometry while
    constraining the helical axis to $\rho \equiv R$:

    $$\phi_k = \phi_0 + \frac{a_{k, y}}{R}$$
    $$\mathbf{r}_k = \left( R \cos\phi_k, R \sin\phi_k, z_0 + a_{k, z} \right)$$
    $$\mathbf{X}_{j} = \mathbf{r}_k + \Delta \mathbf{x}_j \mathbf{R}_z(\phi_k)^T$$

    Parameters
    ----------
    coords : torch.Tensor
        Consensus monomer atom coordinates with shape ``[N, 3]``.
    ideal_com : torch.Tensor
        Ideal center of mass on the cylinder, shape ``[3]`` or ``[1, 3]``.
    radius : float
        Target cylinder radius $R > 0$ (Å).
    atom_token_ids : torch.Tensor | None
        Residue token ID for each atom with shape ``[N]``. If None, residues are
        assumed to be 4-atom blocks (N, CA, C, O), or 1 CA atom per residue if N < 4.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Atomic coordinates on the cylinder with shape ``[N, 3]``.
    """
    n_atoms = coords.shape[0]
    min_atoms = 3
    if n_atoms < min_atoms or radius <= 0:
        return coords

    com_vec = ideal_com.squeeze()
    phi_val = torch.atan2(com_vec[1], com_vec[0])
    z_val = com_vec[2]

    if atom_token_ids is not None and atom_token_ids.shape[0] == n_atoms:
        _, inverse = torch.unique(atom_token_ids, return_inverse=True)
        l_res = int(inverse.max().item()) + 1
        first = torch.full((l_res,), n_atoms, dtype=torch.long, device=coords.device)
        first = first.scatter_reduce(
            0,
            inverse,
            torch.arange(n_atoms, device=coords.device),
            reduce="amin",
        )
        counts = torch.bincount(inverse, minlength=l_res)
        ca_idx = torch.where(counts > 1, first + 1, first)
    elif n_atoms % 4 == 0:
        l_res = n_atoms // 4
        ca_idx = torch.arange(1, n_atoms, 4, device=coords.device)
        inverse = torch.arange(l_res, device=coords.device).repeat_interleave(4)
    else:
        l_res = n_atoms
        ca_idx = torch.arange(n_atoms, device=coords.device)
        inverse = ca_idx

    ca = coords[ca_idx]
    centre = ca.mean(dim=0)
    _, _, vh = torch.linalg.svd(ca - centre)
    u_axis = vh[0]
    if torch.dot(u_axis, ca[-1] - ca[0]) < 0:
        u_axis = -u_axis

    v_axis = torch.stack([torch.zeros_like(u_axis[0]), u_axis[1], u_axis[2]])
    if v_axis.norm() > eps:
        v_axis = v_axis / v_axis.norm()
    else:
        v_axis = torch.zeros_like(u_axis)
        v_axis[2] = torch.sign(u_axis[2] + eps)

    s_k = (ca - centre) @ v_axis
    axis_pts = centre + torch.outer(s_k, v_axis)

    phi_k = phi_val + axis_pts[:, 1] / radius
    r_k = torch.stack(
        [
            radius * torch.cos(phi_k),
            radius * torch.sin(phi_k),
            z_val + axis_pts[:, 2],
        ],
        dim=1,
    )

    rel = coords - axis_pts[inverse]
    phi_atoms = phi_k[inverse]
    cos_p = torch.cos(phi_atoms).unsqueeze(1)
    sin_p = torch.sin(phi_atoms).unsqueeze(1)

    x_rot = rel[:, 0:1] * cos_p - rel[:, 1:2] * sin_p
    y_rot = rel[:, 0:1] * sin_p + rel[:, 1:2] * cos_p
    z_rot = rel[:, 2:3]
    return r_k[inverse] + torch.cat([x_rot, y_rot, z_rot], dim=1)


def unwrap_cylinder_to_consensus(
    coords: torch.Tensor,
    ideal_com: torch.Tensor,
    radius: float,
    *,
    atom_token_ids: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Unroll coordinates from the cylinder surface into the flat consensus frame.

    Inverse of :func:`wrap_consensus_to_cylinder`. Maps the curved cylinder surface
    isometrically onto the flat $(u, v) = (R\phi, z)$ plane so consensus averaging
    occurs in the developable 2D plane:

    $$x_j = \rho_j - R,\quad y_j = R \cdot \Delta\phi_j,\quad z_j = Z_j - z_0$$

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates on the cylinder, shape ``[N, 3]``.
    ideal_com : torch.Tensor
        Ideal center of mass on the cylinder, shape ``[3]`` or ``[1, 3]``.
    radius : float
        Target cylinder radius $R > 0$ (Å).
    atom_token_ids : torch.Tensor | None
        Residue token ID for each atom with shape ``[N]``.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Atomic coordinates in the consensus frame with shape ``[N, 3]``.
    """
    n_atoms = coords.shape[0]
    min_atoms = 3
    if n_atoms < min_atoms or radius <= 0:
        return coords

    com_vec = ideal_com.squeeze()
    phi_val = torch.atan2(com_vec[1], com_vec[0])
    z_val = com_vec[2]

    if atom_token_ids is not None and atom_token_ids.shape[0] == n_atoms:
        _, inverse = torch.unique(atom_token_ids, return_inverse=True)
        l_res = int(inverse.max().item()) + 1
        first = torch.full((l_res,), n_atoms, dtype=torch.long, device=coords.device)
        first = first.scatter_reduce(
            0,
            inverse,
            torch.arange(n_atoms, device=coords.device),
            reduce="amin",
        )
        counts = torch.bincount(inverse, minlength=l_res)
        ca_idx = torch.where(counts > 1, first + 1, first)
    elif n_atoms % 4 == 0:
        l_res = n_atoms // 4
        ca_idx = torch.arange(1, n_atoms, 4, device=coords.device)
        inverse = torch.arange(l_res, device=coords.device).repeat_interleave(4)
    else:
        l_res = n_atoms
        ca_idx = torch.arange(n_atoms, device=coords.device)
        inverse = ca_idx

    ca = coords[ca_idx]
    centre = ca.mean(dim=0)
    _, _, vh = torch.linalg.svd(ca - centre)
    u_axis = vh[0]
    if torch.dot(u_axis, ca[-1] - ca[0]) < 0:
        u_axis = -u_axis

    s_k = (ca - centre) @ u_axis
    axis_3d = centre + torch.outer(s_k, u_axis)

    rho_axis = torch.sqrt(axis_3d[:, 0] ** 2 + axis_3d[:, 1] ** 2 + eps)
    phi_axis = torch.atan2(axis_3d[:, 1], axis_3d[:, 0])
    dphi_axis = (
        torch.remainder(phi_axis - phi_val + math.pi, 2 * math.pi) - math.pi
    )

    axis_flat = torch.stack(
        [
            rho_axis - radius,
            radius * dphi_axis,
            axis_3d[:, 2] - z_val,
        ],
        dim=1,
    )

    rel_3d = coords - axis_3d[inverse]
    phi_atoms = phi_axis[inverse]
    cos_p = torch.cos(phi_atoms).unsqueeze(1)
    sin_p = torch.sin(phi_atoms).unsqueeze(1)

    rel_x = rel_3d[:, 0:1] * cos_p + rel_3d[:, 1:2] * sin_p
    rel_y = -rel_3d[:, 0:1] * sin_p + rel_3d[:, 1:2] * cos_p
    rel_z = rel_3d[:, 2:3]
    return axis_flat[inverse] + torch.cat([rel_x, rel_y, rel_z], dim=1)


def project_axis_to_cylinder(
    coords: torch.Tensor,
    radius: float,
    *,
    atom_token_ids: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Move the element axis onto the cylinder $\rho = R$ by rigid residue shifts.

    The axis is the least-squares line through the CA atoms,
    $\mathbf{a}_k = \mathbf{c} + s_k \mathbf{u}$ with
    $s_k = (\mathbf{x}^{CA}_k - \mathbf{c}) \cdot \mathbf{u}$ (smooth: no
    helical-phase jitter). Every atom of residue $k$ is translated by

    $$\Delta \mathbf{r}_k = (R - \rho_k)\,\hat{\boldsymbol{\rho}}_k,\quad
    \rho_k = \lVert (a_{k,x}, a_{k,y}) \rVert,$$

    so the axis wraps onto the surface (a helix on the cylinder for a tilted
    element) while each residue stays rigid. The shifts vary slowly along the
    axis, so inter-residue bonds change by $\ll 0.1$ Å for tangent axes.

    Parameters
    ----------
    coords : torch.Tensor
        Atom coordinates, shape ``[N, 3]``, cylinder axis = $z$ through origin.
    radius : float
        Target radius $R > 0$ (Å).
    atom_token_ids : torch.Tensor | None
        Residue id per atom, shape ``[N]``; atoms of a residue are contiguous with
        CA second (N, CA, C, O, ...). ``None`` treats each atom as a CA.
    eps : float
        Numerical floor for $\rho_k$.

    Returns
    -------
    torch.Tensor
        Shifted coordinates, shape ``[N, 3]``.
    """
    n_atoms = coords.shape[0]
    min_atoms = 3
    if n_atoms < min_atoms or radius <= 0:
        return coords

    if atom_token_ids is not None and atom_token_ids.shape[0] == n_atoms:
        _, inverse = torch.unique(atom_token_ids, return_inverse=True)
        l_res = int(inverse.max().item()) + 1
        first = torch.full((l_res,), n_atoms, dtype=torch.long, device=coords.device)
        first = first.scatter_reduce(
            0,
            inverse,
            torch.arange(n_atoms, device=coords.device),
            reduce="amin",
        )
        counts = torch.bincount(inverse, minlength=l_res)
        ca_idx = torch.where(counts > 1, first + 1, first)
    else:
        inverse = torch.arange(n_atoms, device=coords.device)
        ca_idx = inverse

    ca = coords[ca_idx]  # [L, 3]
    centre = ca.mean(dim=0)
    u_axis = torch.linalg.svd((ca - centre).T @ (ca - centre))[0][:, 0]  # [3]
    s_k = (ca - centre) @ u_axis  # [L]
    s_k = s_k / (s_k.abs().max() + eps)
    # Quadratic axis a(s) = c0 + c1 s + c2 s^2 per coordinate: smooth, follows any
    # existing curvature (so repeated projection is idempotent) and averages out the
    # helical orbit of the CA atoms.
    degree = min(2, ca.shape[0] - 1)
    vander = torch.stack([s_k**p for p in range(degree + 1)], dim=1)  # [L, d+1]
    coef = torch.linalg.lstsq(vander, ca).solution  # [d+1, 3]
    axis_pts = vander @ coef  # [L, 3]

    rho = torch.sqrt(axis_pts[:, 0] ** 2 + axis_pts[:, 1] ** 2 + eps)  # [L]
    shift = torch.zeros_like(axis_pts)
    shift[:, 0] = (radius - rho) * axis_pts[:, 0] / rho
    shift[:, 1] = (radius - rho) * axis_pts[:, 1] / rho
    return coords + shift[inverse]


def constrain_trace_to_cylinder(
    coords: torch.Tensor,
    radius: float,
    *,
    atom_token_ids: torch.Tensor | None = None,
    secondary_structure: str | None = None,
    max_displacement: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Constrain peptide secondary structural element trace to lie on cylinder surface.

    Translates each residue rigidly in the radial plane by the radial error of its
    helical axis / strand trace, preserving internal 3D secondary structure geometry:
    $$\rho_k = \sqrt{X_{\text{trace}, k}^2 + Y_{\text{trace}, k}^2 + \epsilon}$$
    $$\Delta \mathbf{r}_k = (R - \rho_k) \hat{\boldsymbol{\rho}}_k$$
    $$\mathbf{X}'_{k, \text{atom}} = \mathbf{X}_{k, \text{atom}} + \Delta \mathbf{r}_k$$

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates with shape ``[N, 3]`` in the cylinder-aligned frame.
    radius : float
        Target cylinder radius in Angstroms ($R > 0$).
    atom_token_ids : torch.Tensor | None
        Residue token ID for each atom with shape ``[N]``. If None, residues are
        assumed to be 4-atom blocks (N, CA, C, O), or 1 CA atom per residue if N < 4.
    secondary_structure : str | None
        Secondary structure designation ('H' for helix, 'S' for sheet).
    max_displacement : float | None
        Optional maximum radial displacement per residue in Angstroms to clamp.
    eps : float
        Small constant for numerical stability (avoid division by zero).

    Returns
    -------
    torch.Tensor
        Constrained atomic coordinates with shape ``[N, 3]``.
    """
    n_atoms = coords.shape[0]
    if n_atoms == 0 or radius <= 0:
        return coords

    out = coords.clone()

    if atom_token_ids is not None and atom_token_ids.shape[0] == n_atoms:
        unique_tokens, inverse_indices = torch.unique(
            atom_token_ids, return_inverse=True
        )
        l_res = unique_tokens.shape[0]
        ca_coords = torch.zeros((l_res, 3), device=coords.device, dtype=coords.dtype)
        for r_idx in range(l_res):
            atom_indices = (inverse_indices == r_idx).nonzero(as_tuple=True)[0]
            ca_idx = atom_indices[1] if atom_indices.shape[0] > 1 else atom_indices[0]
            ca_coords[r_idx] = coords[ca_idx]

        trace = compute_element_trace(
            ca_coords, secondary_structure=secondary_structure
        )
        rho = torch.sqrt(trace[:, 0] ** 2 + trace[:, 1] ** 2 + eps)
        unit_radial = torch.zeros((l_res, 3), device=coords.device, dtype=coords.dtype)
        unit_radial[:, 0] = trace[:, 0] / rho
        unit_radial[:, 1] = trace[:, 1] / rho

        delta_rho = radius - rho
        if max_displacement is not None and max_displacement > 0.0:
            delta_rho = torch.clamp(delta_rho, -max_displacement, max_displacement)

        delta_r = unit_radial * delta_rho.unsqueeze(1)
        out += delta_r[inverse_indices]
    elif n_atoms % 4 == 0:
        l_res = n_atoms // 4
        ca_coords = coords.reshape(l_res, 4, 3)[:, 1, :]
        trace = compute_element_trace(
            ca_coords, secondary_structure=secondary_structure
        )
        rho = torch.sqrt(trace[:, 0] ** 2 + trace[:, 1] ** 2 + eps)
        unit_radial = torch.zeros((l_res, 3), device=coords.device, dtype=coords.dtype)
        unit_radial[:, 0] = trace[:, 0] / rho
        unit_radial[:, 1] = trace[:, 1] / rho

        delta_rho = radius - rho
        if max_displacement is not None and max_displacement > 0.0:
            delta_rho = torch.clamp(delta_rho, -max_displacement, max_displacement)

        delta_r = unit_radial * delta_rho.unsqueeze(1)
        delta_r_atoms = delta_r.unsqueeze(1).expand(-1, 4, -1).reshape(n_atoms, 3)
        out += delta_r_atoms
    else:
        ca_coords = coords
        trace = compute_element_trace(
            ca_coords, secondary_structure=secondary_structure
        )
        rho = torch.sqrt(trace[:, 0] ** 2 + trace[:, 1] ** 2 + eps)
        unit_radial = torch.zeros_like(coords)
        unit_radial[:, 0] = trace[:, 0] / rho
        unit_radial[:, 1] = trace[:, 1] / rho

        delta_rho = radius - rho
        if max_displacement is not None and max_displacement > 0.0:
            delta_rho = torch.clamp(delta_rho, -max_displacement, max_displacement)

        out += unit_radial * delta_rho.unsqueeze(1)

    return out
