import torch
import numpy as np

def test_local_double_tape_rmsd():
    # just a placeholder test to verify chi_outward works.
    from boltzgen.task.analyze.analyze_utils import compute_chi_outward
    
    # Test valid interaction
    ca_coords = torch.randn(4, 3)
    design_seq = "AWRA"
    # Residue 1 (W) is outward, Residue 2 (R) is outward. j-i = 2-1 = 1 (odd) -> alternating!
    classification = ["inward", "outward", "outward", "inward"]
    
    # distance matrix
    ca_coords[1] = torch.tensor([0.0, 0.0, 0.0])
    ca_coords[2] = torch.tensor([1.0, 0.0, 0.0]) # dist = 1.0 < 8.0
    
    chi = compute_chi_outward(ca_coords, design_seq, classification)
    assert chi == 1.0 # 1 alternating contact, total 1 contact
