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
    
    M_A = torch.zeros((10, 3))
    M_A[:, 2] = torch.linspace(0, 10, 10)
    M_B = torch.zeros((10, 3))
    M_B[:, 2] = torch.linspace(0, 10, 10)
    
    slots = [M_A, M_B]
    locked_slots = lock_slot_orientations(slots)
    
    layer_chains = {0: {}, 1: {}}
    
    for i in range(n_chains):
        layer, step = double_tape_layer_position(i)
        # Heteromer slot logic: even step -> slot 0, odd step -> slot 1
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
