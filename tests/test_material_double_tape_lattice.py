import torch

from boltzgen.model.modules.materials import (
    double_tape_layer_position,
    generate_ideal_lattice,
)


def test_double_tape_chain_order_alternates_layers_at_each_axial_position():
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
    assert torch.allclose(coms[::2, 1], torch.full((4,), -layer_dist / 2))
    assert torch.allclose(coms[1::2, 1], torch.full((4,), layer_dist / 2))
    assert torch.allclose(coms[::2, 2], coms[1::2, 2])
    assert torch.allclose(torch.diff(coms[::2, 2]), torch.full((3,), pitch))
    assert torch.allclose(torch.diff(coms[1::2, 2]), torch.full((3,), pitch))
