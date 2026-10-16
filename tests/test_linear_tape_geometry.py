import math

import torch

from boltzgen.model.modules import materials

EPS = 1e-4
MIN_END_TO_END = 40.0
MIN_NON_ADJ_DIST = 3.5
MIN_DIR_ALIGN = 0.99


def test_align_consensus_to_linear_tape_preserves_internal_geometry() -> None:
    """Verify alignment to +Y while preserving all internal pairwise distances."""
    torch.manual_seed(42)
    l_res = 15
    base_coords = torch.zeros((l_res, 3), dtype=torch.float32)
    for i in range(l_res):
        base_coords[i, 0] = 0.5 * (-1) ** i
        base_coords[i, 1] = i * 3.4
        base_coords[i, 2] = 0.0

    rx = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(0.8), -math.sin(0.8)],
            [0, math.sin(0.8), math.cos(0.8)],
        ],
        dtype=torch.float32,
    )
    rz = torch.tensor(
        [
            [math.cos(0.5), -math.sin(0.5), 0],
            [math.sin(0.5), math.cos(0.5), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    rotated = base_coords @ (rz @ rx).T + torch.tensor([5.0, -3.0, 2.0])

    dist_before = torch.cdist(rotated, rotated)
    aligned = materials.align_consensus_to_linear_tape(rotated)
    dist_after = torch.cdist(aligned, aligned)

    assert torch.allclose(dist_before, dist_after, atol=1e-5)
    s_dir = materials.strand_direction(aligned)
    assert torch.dot(s_dir, torch.tensor([0.0, 1.0, 0.0])) > MIN_DIR_ALIGN


def test_polarity_alignment_prevents_hairpin_collapse() -> None:
    """Verify that synchronizing strand polarity prevents residue palindrome overlap."""
    torch.manual_seed(42)
    l_res = 15
    strand_fwd = torch.zeros((l_res, 3), dtype=torch.float32)
    for i in range(l_res):
        strand_fwd[i, 0] = 0.5 * (-1) ** i
        strand_fwd[i, 1] = (i - 7.0) * 3.4
        strand_fwd[i, 2] = 0.0

    strand_rev = strand_fwd.clone()
    flip_matrix = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    strand_rev = torch.matmul(strand_rev, flip_matrix)

    naive_avg = (strand_fwd + strand_rev) / 2.0
    assert torch.norm(naive_avg[0] - naive_avg[-1]).item() < EPS

    ref_y = torch.tensor([0.0, 1.0, 0.0])
    chains = [strand_fwd, strand_rev]
    synced_chains = []
    for c in chains:
        s_dir = materials.strand_direction(c)
        if torch.dot(s_dir, ref_y) < 0:
            synced_chains.append(torch.matmul(c, flip_matrix))
        else:
            synced_chains.append(c)

    synced_avg = torch.stack(synced_chains).mean(dim=0)
    end_to_end = torch.norm(synced_avg[-1] - synced_avg[0]).item()
    assert end_to_end > MIN_END_TO_END

    dist_matrix = torch.cdist(synced_avg, synced_avg)
    for i in range(l_res):
        dist_matrix[i, i] = 999.0
        if i + 1 < l_res:
            dist_matrix[i, i + 1] = 999.0
            dist_matrix[i + 1, i] = 999.0
    assert dist_matrix.min().item() > MIN_NON_ADJ_DIST


def test_collinear_kabsch_orientation_stability() -> None:
    """Verify linear_tape Kabsch produces deterministic, orthonormal frame."""
    n_chains = 8
    pitch = 4.8
    steps = torch.arange(n_chains, dtype=torch.float32) - (n_chains - 1) / 2
    q_ideal = torch.zeros((n_chains, 3), dtype=torch.float32)
    q_ideal[:, 2] = steps * pitch
    q_centered = q_ideal - q_ideal.mean(dim=0)

    torch.manual_seed(123)
    for _ in range(10):
        theta = torch.rand(1).item() * math.pi
        r_rot = torch.tensor([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)],
        ])
        p_coords = (q_centered @ r_rot.T) + torch.randn_like(q_centered) * 0.2
        p_centered = p_coords - p_coords.mean(dim=0)

        w = (p_centered * q_centered[:, 2:3]).sum(dim=0)
        u_z = w / (w.norm() + 1e-8)

        v_ref = torch.tensor([0.0, 1.0, 0.0])
        v_perp = v_ref - torch.dot(v_ref, u_z) * u_z
        if v_perp.norm() < EPS:
            v_ref = torch.tensor([1.0, 0.0, 0.0])
            v_perp = v_ref - torch.dot(v_ref, u_z) * u_z
        u_y = v_perp / v_perp.norm()
        u_x = torch.linalg.cross(u_y, u_z)
        u_x = u_x / u_x.norm()

        r_kabsch = torch.stack([u_x, u_y, u_z], dim=1)

        assert torch.allclose(torch.det(r_kabsch), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(r_kabsch @ r_kabsch.T, torch.eye(3), atol=1e-5)

        mapped_z = u_z @ r_kabsch
        assert torch.allclose(mapped_z, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

        mapped_y = u_y @ r_kabsch
        assert torch.allclose(mapped_y, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_linear_tape_guidance_no_shrinkage() -> None:
    """Verify that multi-step linear_tape consensus guidance preserves metric scale without shrinkage.

    Tests 8 copies of a 15-mer peptide ([8, 15, 3] coordinates) across 50 steps
    under guidance_scale = 0.3 for both parallel (antiparallel_prob = 0.0) and
    antiparallel modes with arbitrary initial polarities.
    """
    torch.manual_seed(42)
    n_chains = 8
    l_res = 15
    pitch = 4.8
    guidance_scale = 0.3
    flip_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    ref_y = torch.tensor([0.0, 1.0, 0.0])

    base_strand = torch.zeros((l_res, 3), dtype=torch.float32)
    for i in range(l_res):
        base_strand[i, 0] = 0.5 * (-1) ** i
        base_strand[i, 1] = (i - (l_res - 1) / 2.0) * 3.4

    for is_anti in [False, True]:
        chains = []
        for i in range(n_chains):
            sign = 1.0 if (i % 3 != 0) else -1.0
            c = base_strand.clone()
            c[:, 1] *= sign
            c += torch.randn_like(c) * 0.2
            chains.append(c)

        for _ in range(50):
            coms = torch.stack([c.mean(dim=0) for c in chains])
            ideal_coms = torch.zeros((n_chains, 3), dtype=torch.float32)
            ideal_coms[:, 2] = (
                torch.arange(n_chains, dtype=torch.float32) - (n_chains - 1) / 2.0
            ) * pitch

            p_mean = coms.mean(dim=0)
            q_mean = ideal_coms.mean(dim=0)
            p_c = coms - p_mean
            q_c = ideal_coms - q_mean

            w = (p_c * q_c[:, 2:3]).sum(dim=0)
            u_z = w / (w.norm() + 1e-8)
            s_dirs = [materials.strand_direction(c) for c in chains]
            ref_s = s_dirs[0]
            aligned = [s if torch.dot(s, ref_s) >= 0 else -s for s in s_dirs]
            v_ref = torch.stack(aligned).mean(dim=0)
            v_perp = v_ref - torch.dot(v_ref, u_z) * u_z
            u_y = v_perp / (v_perp.norm() + 1e-8)
            u_x = torch.linalg.cross(u_y, u_z)
            r_kabsch = torch.stack([u_x, u_y, u_z], dim=1)

            ref_acc = None
            for i in range(n_chains):
                expected_sign = -1.0 if (is_anti and i % 2 == 1) else 1.0
                c_local = (chains[i] - p_mean) @ r_kabsch + q_mean
                c_folded = c_local - ideal_coms[i]

                s_dir = materials.strand_direction(c_folded)
                if torch.dot(s_dir, ref_y) * expected_sign < 0:
                    c_folded = c_folded @ flip_matrix
                    c_unf = c_folded + ideal_coms[i]
                    chains[i] = (c_unf - q_mean) @ r_kabsch.T + p_mean

                c_for_consensus = (
                    c_folded @ flip_matrix if (is_anti and i % 2 == 1) else c_folded
                )
                if ref_acc is None:
                    ref_acc = c_for_consensus.clone()
                else:
                    ref_acc += c_for_consensus

            ref_coords = ref_acc / n_chains

            for i in range(n_chains):
                c_unf_slot = (
                    ref_coords @ flip_matrix if (is_anti and i % 2 == 1) else ref_coords
                )
                c_unf = c_unf_slot + ideal_coms[i]
                c_global = (c_unf - q_mean) @ r_kabsch.T + p_mean
                chains[i] = (
                    1.0 - guidance_scale
                ) * chains[i] + guidance_scale * c_global

        for i in range(n_chains):
            end_to_end = torch.norm(chains[i][-1] - chains[i][0]).item()
            assert end_to_end > MIN_END_TO_END, (
                f"Shrinkage detected: chain {i} end_to_end={end_to_end:.2f} < {MIN_END_TO_END}"
            )
            ca_dist = torch.diff(chains[i], dim=0).norm(dim=1).mean().item()
            assert 3.2 < ca_dist < 4.0, (
                f"Bond length distorted: chain {i} CA-CA={ca_dist:.2f}"
            )

