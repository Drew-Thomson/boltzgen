"""Unit tests for material lattice geometry and point-group rotations."""

# ruff: noqa: INP001

from __future__ import annotations

import math

import pytest
import torch

from boltzgen.model.modules.materials import (
    OCTAHEDRAL_ROTATIONS,
    TETRAHEDRAL_ROTATIONS,
    fold_to_consensus_frame,
    generate_ideal_lattice,
    unfold_from_consensus_frame,
)

TETRAHEDRAL_COPY_COUNT = 12
OCTAHEDRAL_COPY_COUNT = 24


def _assert_group(rotations: tuple[tuple[tuple[int, ...], ...], ...]) -> None:
    matrices = [torch.tensor(rotation, dtype=torch.float64) for rotation in rotations]
    identity = torch.eye(3, dtype=torch.float64)
    for matrix in matrices:
        torch.testing.assert_close(matrix @ matrix.T, identity, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            torch.linalg.det(matrix), torch.tensor(1.0, dtype=torch.float64)
        )
    for left in matrices:
        for right in matrices:
            product = left @ right
            assert any(torch.allclose(product, member) for member in matrices)


def test_tetrahedral_rotations_are_valid_so3() -> None:
    assert len(TETRAHEDRAL_ROTATIONS) == TETRAHEDRAL_COPY_COUNT
    _assert_group(TETRAHEDRAL_ROTATIONS)


def test_octahedral_rotations_are_valid_so3() -> None:
    assert len(OCTAHEDRAL_ROTATIONS) == OCTAHEDRAL_COPY_COUNT
    _assert_group(OCTAHEDRAL_ROTATIONS)


def test_generate_ideal_lattice_cyclic_matches_legacy() -> None:
    centers, rotations = generate_ideal_lattice(
        "cyclic", 4, 1, {"target_radius": 15.0}, torch.device("cpu"), torch.float64
    )
    angles = torch.arange(4, dtype=torch.float64) * (2 * torch.pi / 4)
    expected = torch.stack(
        (15 * torch.cos(angles), 15 * torch.sin(angles), torch.zeros(4)), dim=1
    )
    torch.testing.assert_close(centers, expected)
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(4, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("topology", "params", "expected"),
    [
        ("linear_tape", {"target_pitch": 5.0}, [[0, 0, -5], [0, 0, 0], [0, 0, 5]]),
        (
            "double_tape",
            {"target_pitch": 4.8, "layer_dist": 10.0},
            [[-5, 0, -1.2], [5, 0, -1.2], [5, 0, 3.6]],
        ),
        (
            "helical",
            {"target_radius": 2.0, "target_angle": 90.0, "target_dz": 5.0},
            [[2, 0, -5], [0, 2, 0], [-2, 0, 5]],
        ),
    ],
)
def test_existing_linear_and_helical_lattices(
    topology: str, params: dict[str, float], expected: list[list[float]]
) -> None:
    centers, _ = generate_ideal_lattice(
        topology, 3, 1, params, torch.device("cpu"), torch.float64
    )
    torch.testing.assert_close(centers, torch.tensor(expected, dtype=torch.float64))


def test_generate_ideal_lattice_cage_tetrahedral_shape_and_orbit() -> None:
    centers, rotations = generate_ideal_lattice(
        "cage_tetrahedral", 12, 1, {"target_cage_radius": 20.0}, torch.device("cpu")
    )
    assert centers.shape == (TETRAHEDRAL_COPY_COUNT, 3)
    assert rotations.shape == (TETRAHEDRAL_COPY_COUNT, 3, 3)
    norms = torch.linalg.vector_norm(centers, dim=1)
    torch.testing.assert_close(norms, torch.full((TETRAHEDRAL_COPY_COUNT,), 20.0))
    matrix = torch.tensor(TETRAHEDRAL_ROTATIONS[3], dtype=centers.dtype)
    transformed = torch.einsum("ij,nj->ni", matrix, centers)
    expected_distances = torch.cdist(centers, centers).sort(dim=1)[0]
    transformed_distances = torch.cdist(transformed, transformed).sort(dim=1)[0]
    torch.testing.assert_close(transformed_distances, expected_distances)


def test_generate_ideal_lattice_cage_octahedral_shape() -> None:
    centers, rotations = generate_ideal_lattice(
        "cage_octahedral", 24, 1, {"target_cage_radius": 35.0}, torch.device("cpu")
    )
    assert centers.shape == (OCTAHEDRAL_COPY_COUNT, 3)
    assert rotations.shape == (OCTAHEDRAL_COPY_COUNT, 3, 3)
    norms = torch.linalg.vector_norm(centers, dim=1)
    torch.testing.assert_close(norms, torch.full((OCTAHEDRAL_COPY_COUNT,), 35.0))


def test_generate_ideal_lattice_bilayer_sheet_matches_grid() -> None:
    centers, rotations = generate_ideal_lattice(
        "bilayer_sheet",
        8,
        1,
        {
            "grid_dim_x": 2,
            "grid_dim_y": 2,
            "target_pitch": 5.0,
            "row_pitch": 8.0,
            "layer_dist": 12.0,
        },
        torch.device("cpu"),
        torch.float64,
    )
    expected = torch.tensor(
        [
            [-2.5, -4.0, -6.0], [-2.5, 4.0, -6.0],
            [2.5, -4.0, -6.0], [2.5, 4.0, -6.0],
            [-2.5, -4.0, 6.0], [-2.5, 4.0, 6.0],
            [2.5, -4.0, 6.0], [2.5, 4.0, 6.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(centers, expected)
    identity = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
    torch.testing.assert_close(rotations, identity)


def test_generate_ideal_lattice_bilayer_sheet_requires_matching_copy_count() -> None:
    with pytest.raises(ValueError, match="bilayer_sheet requires 8 chains"):
        generate_ideal_lattice(
            "bilayer_sheet",
            7,
            1,
            {"grid_dim_x": 2, "grid_dim_y": 2},
            torch.device("cpu"),
        )


def test_generate_ideal_lattice_hexagonal_mesh_shape_and_rotation() -> None:
    centers, rotations = generate_ideal_lattice(
        "hexagonal_mesh",
        12,
        1,
        {"grid_dim_x": 2, "grid_dim_y": 2, "lattice_constant": 10.0},
        torch.device("cpu"),
        torch.float64,
    )
    assert centers.shape == (12, 3)
    assert rotations.shape == (12, 3, 3)
    torch.testing.assert_close(centers[:, 2], torch.zeros(12, dtype=torch.float64))
    expected_node_centers = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 5 * 3**0.5, 0],
         [5, 5 * 3**0.5, 0], [5, 5 * 3**0.5, 0], [10, 0, 0], [10, 0, 0],
         [10, 0, 0], [15, 5 * 3**0.5, 0], [15, 5 * 3**0.5, 0],
         [15, 5 * 3**0.5, 0]], dtype=torch.float64
    )
    torch.testing.assert_close(centers, expected_node_centers)
    sqrt_three = 3**0.5
    expected_rotations = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [
                [-0.5, sqrt_three / 2, 0.0],
                [-sqrt_three / 2, -0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [-0.5, -sqrt_three / 2, 0.0],
                [sqrt_three / 2, -0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rotations[:3], expected_rotations)
    for node in range(4):
        torch.testing.assert_close(
            rotations[node * 3 : (node + 1) * 3], expected_rotations
        )


def test_generate_ideal_lattice_hexagonal_mesh_requires_matching_copy_count() -> None:
    with pytest.raises(ValueError, match="hexagonal_mesh requires 12 chains"):
        generate_ideal_lattice(
            "hexagonal_mesh",
            10,
            1,
            {"grid_dim_x": 2, "grid_dim_y": 2},
            torch.device("cpu"),
        )


def test_generate_ideal_lattice_nanotube_one_tier_matches_cyclic() -> None:
    params = {"ring_size": 4, "num_tiers": 1, "target_radius": 15.0}
    centers, rotations = generate_ideal_lattice(
        "nanotube", 4, 1, params, torch.device("cpu"), torch.float64
    )
    cyclic_centers, cyclic_rotations = generate_ideal_lattice(
        "cyclic", 4, 1, {"target_radius": 15.0}, torch.device("cpu"), torch.float64
    )
    torch.testing.assert_close(centers, cyclic_centers)
    torch.testing.assert_close(rotations, cyclic_rotations)


def test_generate_ideal_lattice_nanotube_stacked_tiers() -> None:
    centers, _ = generate_ideal_lattice(
        "nanotube",
        4,
        1,
        {"ring_size": 2, "num_tiers": 2, "target_radius": 3.0, "target_dz": 4.8},
        torch.device("cpu"),
        torch.float64,
    )
    expected = torch.tensor(
        [[3.0, 0.0, -2.4], [-3.0, 0.0, -2.4], [3.0, 0.0, 2.4], [-3.0, 0.0, 2.4]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(centers, expected)


def test_generate_ideal_lattice_nanotube_chiral_stagger_rotates_tiers() -> None:
    centers, rotations = generate_ideal_lattice(
        "nanotube",
        4,
        1,
        {
            "ring_size": 2,
            "num_tiers": 2,
            "target_radius": 3.0,
            "target_dz": 4.8,
            "chiral_stagger": 90.0,
        },
        torch.device("cpu"),
        torch.float64,
    )
    torch.testing.assert_close(
        centers[2], torch.tensor([0.0, 3.0, 2.4], dtype=torch.float64),
        atol=1e-12, rtol=0,
    )
    torch.testing.assert_close(
        centers[3], torch.tensor([0.0, -3.0, 2.4], dtype=torch.float64),
        atol=1e-12, rtol=0,
    )
    expected_rotation = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rotations[2], expected_rotation, atol=1e-12, rtol=0)


def test_generate_ideal_lattice_nanotube_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="nanotube requires 8 chains"):
        generate_ideal_lattice(
            "nanotube", 7, 1, {"ring_size": 2, "num_tiers": 4}, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="must be positive"):
        generate_ideal_lattice(
            "nanotube", 4, 1, {"ring_size": 0, "num_tiers": 2}, torch.device("cpu")
        )


def test_generate_ideal_lattice_multi_helical_one_start_matches_helical() -> None:
    params = {
        "num_starts": 1,
        "target_radius": 2.0,
        "target_angle": 90.0,
        "target_dz": 5.0,
    }
    centers, _ = generate_ideal_lattice(
        "multi_helical", 3, 1, params, torch.device("cpu"), torch.float64
    )
    helical_centers, _ = generate_ideal_lattice(
        "helical",
        3,
        1,
        {"target_radius": 2.0, "target_angle": 90.0, "target_dz": 5.0},
        torch.device("cpu"),
        torch.float64,
    )
    torch.testing.assert_close(centers, helical_centers)


def test_generate_ideal_lattice_multi_helical_strand_phase_and_tilt() -> None:
    centers, rotations = generate_ideal_lattice(
        "multi_helical",
        6,
        1,
        {"num_starts": 3, "target_radius": 2.0, "target_angle": 30.0, "target_dz": 4.0},
        torch.device("cpu"),
        torch.float64,
    )
    at_first_position = centers[:3]
    torch.testing.assert_close(
        torch.linalg.vector_norm(at_first_position[:, :2], dim=1),
        torch.full((3,), 2.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        at_first_position[:, 2], torch.full((3,), -2.0, dtype=torch.float64)
    )
    angles = torch.atan2(at_first_position[:, 1], at_first_position[:, 0])
    angle_differences = torch.remainder(angles[1:] - angles[:-1], 2 * torch.pi)
    expected_angle_differences = torch.full(
        (2,), 2 * torch.pi / 3, dtype=torch.float64
    )
    torch.testing.assert_close(angle_differences, expected_angle_differences)

    tilt = math.atan2(4.0, 2.0 * math.radians(30.0))
    expected_rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), math.sin(tilt)],
            [0.0, -math.sin(tilt), math.cos(tilt)],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rotations[0], expected_rotation)


def test_generate_ideal_lattice_multi_helical_zero_angle_uses_axial_tilt() -> None:
    centers, rotations = generate_ideal_lattice(
        "multi_helical",
        3,
        1,
        {"num_starts": 1, "target_radius": 2.0, "target_angle": 0.0, "target_dz": 5.0},
        torch.device("cpu"),
        torch.float64,
    )
    assert torch.isfinite(centers).all()
    torch.testing.assert_close(
        rotations[0],
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=0,
    )


def test_generate_ideal_lattice_multi_helical_requires_divisible_copy_count() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        generate_ideal_lattice(
            "multi_helical", 8, 1, {"num_starts": 3}, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="num_starts must be positive"):
        generate_ideal_lattice(
            "multi_helical", 4, 1, {"num_starts": 0}, torch.device("cpu")
        )


def test_generate_ideal_lattice_requires_polyhedral_copy_count() -> None:
    with pytest.raises(ValueError, match="requires 12 copies"):
        generate_ideal_lattice("cage_tetrahedral", 11, 1, {}, torch.device("cpu"))


def test_fold_and_unfold_round_trip() -> None:
    torch.manual_seed(9)
    rotations = torch.stack(
        [
            torch.tensor(rotation, dtype=torch.float64)
            for rotation in TETRAHEDRAL_ROTATIONS[:4]
        ]
    )
    centers = torch.randn((4, 3), dtype=torch.float64)
    consensus = torch.randn((7, 3), dtype=torch.float64)
    coords = unfold_from_consensus_frame(consensus, centers, rotations)
    folded = fold_to_consensus_frame(coords, centers, rotations)
    torch.testing.assert_close(folded, consensus.expand_as(folded))
