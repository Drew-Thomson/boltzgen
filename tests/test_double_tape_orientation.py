import pytest
import torch
from boltzgen.model.modules.materials import (
    strand_direction,
    lock_slot_orientations,
    double_tape_antiparallel_rotation,
    double_tape_layer_position,
)

def test_strand_direction():
    coords_pos = torch.zeros((10, 3))
    coords_pos[:, 2] = torch.linspace(0, 10, 10)
    dir_pos = strand_direction(coords_pos)
    assert torch.allclose(dir_pos, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

    coords_neg = torch.zeros((10, 3))
    coords_neg[:, 2] = torch.linspace(10, 0, 10)
    dir_neg = strand_direction(coords_neg)
    assert torch.allclose(dir_neg, torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)

def test_lock_slot_orientations():
    coords0 = torch.zeros((10, 3))
    coords0[:, 2] = torch.linspace(0, 10, 10)
    coords1 = torch.zeros((10, 3))
    coords1[:, 2] = torch.linspace(10, 0, 10)
    
    locked = lock_slot_orientations([coords0, coords1])
    dir0 = strand_direction(locked[0])
    dir1 = strand_direction(locked[1])
    assert torch.dot(dir0, dir1) > 0.99

    locked_same = lock_slot_orientations([coords0, coords0.clone()])
    assert torch.allclose(locked_same[1], coords0)

def test_simulated_heteromer_fold_flip_unfold():
    device = torch.device("cpu")
    dtype = torch.float32

    n_chains = 16

    # Strand runs along Y (the strand axis in double_tape)
    M_A = torch.zeros((10, 3))
    M_A[:, 1] = torch.linspace(0, 10, 10)
    M_B = torch.zeros((10, 3))
    M_B[:, 1] = torch.linspace(0, 10, 10)

    slots = [M_A, M_B]
    locked_slots = lock_slot_orientations(slots)

    layer_chains = {0: {}, 1: {}}

    for i in range(n_chains):
        layer, step = double_tape_layer_position(i)
        slot_idx = step % 2

        flip = double_tape_antiparallel_rotation(i, device=device, dtype=dtype)
        unfolded_coords = torch.matmul(locked_slots[slot_idx], flip.transpose(-1, -2))
        layer_chains[layer][step] = unfolded_coords

    # Check in-sheet neighbors alternate
    for layer in (0, 1):
        for step in range(7):
            dir_i = strand_direction(layer_chains[layer][step])
            dir_j = strand_direction(layer_chains[layer][step + 1])
            dot = torch.dot(dir_i, dir_j)
            assert dot < -0.99, f"Layer {layer} in-sheet parallel at step {step}"

    # Check cross-layer pairs are antiparallel
    for step in range(8):
        dir_0 = strand_direction(layer_chains[0][step])
        dir_1 = strand_direction(layer_chains[1][step])
        dot = torch.dot(dir_0, dir_1)
        assert dot < -0.99, f"Cross-layer parallel at step {step}"


def test_double_tape_even_inside_odd_outside() -> None:
    """Verify that residues 2, 4, 6, 8 point inward (toward center) and
    residues 1, 3, 5, 7, 9 point outward for all chains in both layers."""
    from boltzgen.model.modules.materials import align_consensus_to_double_tape

    # 9-residue beta strand with alternating sidechain pleating
    bb_coords = []
    for i in range(9):
        y = i * 3.5
        sign = 1.0 if i % 2 == 0 else -1.0
        n = [0.2 * sign, y - 1.2, 0.0]
        ca = [0.0, y, 0.2 * sign]
        c = [-0.2 * sign, y + 1.2, 0.0]
        o = [-0.5 * sign, y + 1.2, 0.5 * sign]
        bb_coords.extend([n, ca, c, o])

    coords = torch.tensor(bb_coords, dtype=torch.float32)
    aligned = align_consensus_to_double_tape(coords)
    l_res = 9
    bb = aligned.view(l_res, 4, 3)

    for chain_idx in range(16):
        layer, _ = double_tape_layer_position(chain_idx)
        r_rot = double_tape_antiparallel_rotation(
            chain_idx, device=torch.device("cpu"), dtype=torch.float32
        )
        assert torch.isclose(torch.det(r_rot), torch.tensor(1.0))

        unfolded_bb = bb @ r_rot.T
        n_p, ca_p, c_p = unfolded_bb[:, 0], unfolded_bb[:, 1], unfolded_bb[:, 2]
        b_u = ca_p - n_p
        c_u = c_p - ca_p
        b_n = torch.linalg.norm(b_u - c_u, dim=-1, keepdim=True) + 1e-8
        cr_u = torch.linalg.cross(b_u, c_u)
        c_n = torch.linalg.norm(cr_u, dim=-1, keepdim=True) + 1e-8
        v_u = -0.531018 * ((b_u - c_u) / b_n) + 1.20673 * (cr_u / c_n)
        v_u = v_u / (torch.linalg.norm(v_u, dim=-1, keepdim=True) + 1e-8)

        # Inward normal points toward center (X=0): Layer 0 (+X), Layer 1 (-X)
        inward_normal = (
            torch.tensor([1.0, 0.0, 0.0])
            if layer == 0
            else torch.tensor([-1.0, 0.0, 0.0])
        )
        dots = v_u @ inward_normal
        odd_dots = dots[0::2]  # residues 1, 3, 5, 7, 9
        even_dots = dots[1::2]  # residues 2, 4, 6, 8

        # Odd residues must point outward (away from center -> dot < 0)
        assert (odd_dots < 0).all(), (
            f"Chain {chain_idx} (layer {layer}) odd residues not outward: {odd_dots}"
        )
        # Even residues must point inward (toward center -> dot > 0)
        assert (even_dots > 0).all(), (
            f"Chain {chain_idx} (layer {layer}) even residues not inward: {even_dots}"
        )


def test_align_consensus_to_double_tape_arbitrary_rotations() -> None:
    from boltzgen.model.modules.materials import align_consensus_to_double_tape

    # Synthetic 9-mer strand coordinates (~26 A long along an arbitrary axis)
    torch.manual_seed(42)
    coords = torch.zeros((9, 3))
    coords[:, 0] = torch.linspace(-13.0, 13.0, 9)

    # Apply random 3D rotation
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated_coords = torch.matmul(coords, q.T)

    aligned = align_consensus_to_double_tape(rotated_coords)
    assert aligned.shape == (9, 3)

    # Check centered
    assert torch.allclose(aligned.mean(dim=0), torch.zeros(3), atol=1e-5)

    # Check primary PCA axis lies along Y
    u_mat, _, _ = torch.linalg.svd(aligned.T @ aligned)
    assert abs(float(u_mat[1, 0])) > 0.999

    # Check N->C direction points along +Y
    dir_aligned = strand_direction(aligned)
    assert torch.allclose(dir_aligned, torch.tensor([0.0, 1.0, 0.0]), atol=1e-3)


def test_align_consensus_to_double_tape_prevents_layer_overlap() -> None:
    from boltzgen.model.modules.materials import (
        align_consensus_to_double_tape,
        generate_ideal_lattice,
    )

    # 9-residue extended strand (~26 A long, wider than 8.5 A layer spacing)
    torch.manual_seed(42)
    strand = torch.zeros((9, 3))
    # Intentionally oriented along X (bilayer normal)
    strand[:, 0] = torch.linspace(-13.0, 13.0, 9)

    aligned = align_consensus_to_double_tape(strand)

    # Verify extent along X is collapsed to 0 for linear model
    assert float(aligned[:, 0].abs().max()) < 1e-4

    # Generate ideal lattice for double_tape (16 chains, 8.5 A layer distance)
    coms, _ = generate_ideal_lattice(
        "double_tape",
        16,
        1,
        {"layer_dist": 8.5, "target_pitch": 4.8},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    device = torch.device("cpu")
    dtype = torch.float32

    # Layer 0 (chain 0, X = -4.25) vs Layer 1 (chain 1, X = +4.25)
    flip_0 = double_tape_antiparallel_rotation(0, device=device, dtype=dtype)
    flip_1 = double_tape_antiparallel_rotation(1, device=device, dtype=dtype)

    chain_0 = torch.matmul(aligned, flip_0.T) + coms[0]
    chain_1 = torch.matmul(aligned, flip_1.T) + coms[1]

    min_dist = float(torch.min(torch.cdist(chain_0, chain_1)))
    assert min_dist >= 8.4, f"Expected clearance >= 8.4 A, got {min_dist:.2f} A"
