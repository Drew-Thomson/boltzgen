import torch

from boltzgen.model.modules.materials import (
    double_tape_layer_position,
    generate_ideal_lattice,
)


def test_double_tape_chain_order_alternates_layers_at_each_axial_position() -> None:
    count = 8
    pitch = 4.8
    layer_dist = 8.0
    coms, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {"target_pitch": pitch, "layer_dist": layer_dist},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert [double_tape_layer_position(index) for index in range(count)] == [
        (index % 2, index // 2) for index in range(count)
    ]
    assert torch.allclose(coms[::2, 0], torch.full((4,), -layer_dist / 2))
    assert torch.allclose(coms[1::2, 0], torch.full((4,), layer_dist / 2))
    # Default longitudinal offset between sheets is 0.5 * pitch (2.4 A)
    assert torch.allclose(coms[1::2, 2] - coms[::2, 2], torch.full((4,), 0.5 * pitch))
    assert torch.allclose(torch.diff(coms[::2, 2]), torch.full((3,), pitch))
    assert torch.allclose(torch.diff(coms[1::2, 2]), torch.full((3,), pitch))


def test_double_tape_supercoiling_geometry() -> None:
    count = 8
    pitch = 4.8
    layer_dist = 8.0
    twist_deg = 10.0

    coms, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {
            "target_pitch": pitch,
            "layer_dist": layer_dist,
            "target_twist": twist_deg,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Transverse cross-layer distance at each step must remain equal to layer_dist
    for s in range(count // 2):
        pair_dist_xy = torch.norm((coms[2 * s + 1] - coms[2 * s])[:2])
        assert torch.isclose(pair_dist_xy, torch.tensor(layer_dist), atol=1e-5)
        # Longitudinal offset is 0.5 * pitch
        assert torch.isclose(
            coms[2 * s + 1, 2] - coms[2 * s, 2], torch.tensor(0.5 * pitch), atol=1e-5
        )

    # Twist offset adds linearly to target_twist
    coms_offset, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {
            "target_pitch": pitch,
            "layer_dist": layer_dist,
            "target_twist": twist_deg,
            "twist_offset": 5.0,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    coms_expected, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {
            "target_pitch": pitch,
            "layer_dist": layer_dist,
            "target_twist": twist_deg + 5.0,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(coms_offset, coms_expected, atol=1e-5)


def test_double_tape_strand_spacing_default_and_custom() -> None:
    count = 6
    # 1. Default strand spacing without target_pitch/strand_spacing specified is 4.8
    coms_default, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {"layer_dist": 10.0},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(torch.diff(coms_default[::2, 2]), torch.full((2,), 4.8))

    # 2. Custom strand_spacing parameter
    coms_custom, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {"layer_dist": 10.0, "strand_spacing": 5.2},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(torch.diff(coms_custom[::2, 2]), torch.full((2,), 5.2))


def test_double_tape_layer_offset_z_custom_and_zero() -> None:
    count = 6
    pitch = 4.8
    # Explicit 0.0 offset produces coplanar strands across layers
    coms_zero, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {"target_pitch": pitch, "layer_dist": 10.0, "layer_offset_z": 0.0},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(coms_zero[::2, 2], coms_zero[1::2, 2])

    # Explicit custom offset
    coms_custom, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {"target_pitch": pitch, "layer_dist": 10.0, "layer_offset_z": 3.0},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(coms_custom[1::2, 2] - coms_custom[::2, 2], torch.full((3,), 3.0))


def test_double_tape_spacing_noise_modulates_inter_sheet_geometry() -> None:
    count = 8
    pitch = 4.8
    layer_dist = 10.0
    dist_noise = 1.5
    offset_y = 0.8
    offset_z_noise = 0.4

    coms, _ = generate_ideal_lattice(
        "double_tape",
        count,
        1,
        {
            "strand_spacing": pitch,
            "layer_dist": layer_dist,
            "layer_dist_noise": dist_noise,
            "layer_offset_y_noise": offset_y,
            "layer_offset_z_noise": offset_z_noise,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Inter-sheet distance along X is perturbed to layer_dist + dist_noise
    expected_dist = layer_dist + dist_noise
    assert torch.allclose(coms[1::2, 0] - coms[::2, 0], torch.full((4,), expected_dist))

    # Lateral offset along Y is perturbed to offset_y
    assert torch.allclose(coms[1::2, 1] - coms[::2, 1], torch.full((4,), offset_y))

    # Longitudinal offset along Z is nominal (0.5 * pitch) + offset_z_noise
    expected_offset_z = 0.5 * pitch + offset_z_noise
    assert torch.allclose(coms[1::2, 2] - coms[::2, 2], torch.full((4,), expected_offset_z))

    # Intra-sheet strand pitch along Z is strictly preserved at pitch
    assert torch.allclose(torch.diff(coms[::2, 2]), torch.full((3,), pitch))
    assert torch.allclose(torch.diff(coms[1::2, 2]), torch.full((3,), pitch))



