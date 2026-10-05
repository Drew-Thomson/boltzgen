import torch
import pytest
from math import isnan

from boltzgen.model.loss.validation import compute_subset_rmsd, weighted_minimum_rmsd
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.data import const

def test_weighted_rigid_align_zero_weights():
    # Test defense in depth: 0 weights should return true_coords, not throw NaN exceptions in SVD
    B, N, D = 2, 5, 3
    true_coords = torch.rand(B, N, D)
    pred_coords = torch.rand(B, N, D)
    
    # 0 weights
    weights = torch.zeros(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)
    
    aligned = weighted_rigid_align(true_coords, pred_coords, weights, mask)
    assert torch.allclose(aligned, true_coords)
    assert not torch.isnan(aligned).any()

def test_compute_subset_rmsd_low_points():
    B, N, D = 1, 5, 3
    true_coords = torch.rand(B, N, D)
    pred_coords = torch.rand(B, N, D)
    multiplicity = 1
    
    atom_mask = torch.ones(B, N, dtype=torch.bool)
    align_weights = torch.ones(B, N)
    
    # Case 1: 0 active points
    subset_mask_0 = torch.zeros(B, N, dtype=torch.bool)
    rmsd, best_rmsd = compute_subset_rmsd(
        true_coords, pred_coords, atom_mask, align_weights, subset_mask_0, multiplicity
    )
    assert isnan(rmsd.item()) and isnan(best_rmsd.item())
    
    # Case 2: 2 active points (less than dim=3)
    subset_mask_2 = torch.zeros(B, N, dtype=torch.bool)
    subset_mask_2[0, :2] = True
    rmsd, best_rmsd = compute_subset_rmsd(
        true_coords, pred_coords, atom_mask, align_weights, subset_mask_2, multiplicity
    )
    assert isnan(rmsd.item()) and isnan(best_rmsd.item())
    
    # Case 3: 3 active points (should compute value)
    subset_mask_3 = torch.zeros(B, N, dtype=torch.bool)
    subset_mask_3[0, :3] = True
    rmsd, best_rmsd = compute_subset_rmsd(
        true_coords, pred_coords, atom_mask, align_weights, subset_mask_3, multiplicity
    )
    assert not isnan(rmsd.item()) and not isnan(best_rmsd.item())

def test_compute_subset_rmsd_zero_effective_weights():
    B, N, D = 1, 5, 3
    true_coords = torch.rand(B, N, D)
    pred_coords = torch.rand(B, N, D)
    multiplicity = 1
    
    atom_mask = torch.ones(B, N, dtype=torch.bool)
    subset_mask = torch.ones(B, N, dtype=torch.bool)
    align_weights = torch.zeros(B, N) # zero weights!
    
    rmsd, best_rmsd = compute_subset_rmsd(
        true_coords, pred_coords, atom_mask, align_weights, subset_mask, multiplicity
    )
    assert isnan(rmsd.item()) and isnan(best_rmsd.item())

def test_weighted_minimum_rmsd_metal_sandwich_edge_case():
    B, K, L, T, D = 1, 10, 5, 5, 3
    feats = {
        "coords": torch.rand(B, K, L, D),
        "atom_resolved_mask": torch.zeros(B, L, dtype=torch.bool), # No backbone atoms
        "mol_type": torch.full((B, T), const.chain_type_ids["NONPOLYMER"]),
        "atom_to_token": torch.eye(L).unsqueeze(0), # (1, L, T)
        "design_mask": torch.ones(B, T, dtype=torch.bool),
        "chain_design_mask": torch.zeros(B, T, dtype=torch.bool), # All target
    }

    pred_atom_coords = torch.rand(1, L, D) # multiplicity=1
    
    # Should not throw any exception or nan-related SVD warnings
    out = weighted_minimum_rmsd(pred_atom_coords, feats, protein_lig_rmsd=True)
    target_aligned_rmsd_design = out[-2]
    assert isnan(target_aligned_rmsd_design.item())

def test_weighted_minimum_rmsd_valid_target():
    B, K, L, T, D = 1, 10, 5, 5, 3
    feats = {
        "coords": torch.rand(B, K, L, D),
        "atom_resolved_mask": torch.ones(B, L, dtype=torch.bool),
        "mol_type": torch.full((B, T), const.chain_type_ids["PROTEIN"]),
        "atom_to_token": torch.eye(L).unsqueeze(0),
        "design_mask": torch.ones(B, T, dtype=torch.bool),
        "chain_design_mask": torch.ones(B, T, dtype=torch.bool),
    }

    # 3 target atoms resolved
    feats["chain_design_mask"][0, 0:3] = False # token 0,1,2 are targets
    
    pred_atom_coords = torch.rand(1, L, D) # multiplicity=1
    
    # Should calculate successfully
    out = weighted_minimum_rmsd(pred_atom_coords, feats, protein_lig_rmsd=True)
    target_aligned_rmsd_design = out[-2]
    pass
