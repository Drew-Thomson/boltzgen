import pytest
import torch

from boltzgen.model.modules.materials import build_layout_guidance_masks


def _layout():
    return {
        "chains": [
            {"asym_index": 0, "role": "protein"},
            {"asym_index": 1, "role": "ligand"},
            {"asym_index": 2, "role": "protein"},
            {"asym_index": 3, "role": "ligand"},
        ],
        "placements": [
            {"placement_index": 0, "protein_asym_indices": [0], "ligand_asym_indices": [1]},
            {"placement_index": 1, "protein_asym_indices": [2], "ligand_asym_indices": [3]},
        ],
    }


def test_guidance_masks_exclude_ligands_from_placement_coms():
    atom_asym_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    masks, protein_masks, ligands = build_layout_guidance_masks(atom_asym_id, _layout(), "cyclic")
    assert [int(mask.sum()) for mask in masks] == [2, 2]
    assert list(protein_masks) == [0, 2]
    assert [entry[0] for entry in ligands] == [1, 3]
    assert not any(mask[2] for mask in masks)


def test_double_tape_guides_each_protein_slot_individually():
    layout = {
        "chains": [
            {"asym_index": 0, "role": "protein"},
            {"asym_index": 1, "role": "protein"},
            {"asym_index": 2, "role": "ligand"},
        ],
        "placements": [
            {"placement_index": 0, "protein_asym_indices": [0, 1], "ligand_asym_indices": [2]},
        ],
    }
    atom_asym_id = torch.tensor([0, 0, 1, 1, 2])
    masks, _, _ = build_layout_guidance_masks(atom_asym_id, layout, "double_tape")
    assert [int(mask.sum()) for mask in masks] == [2, 2]


def test_layout_missing_chain_fails_diagnostically():
    with pytest.raises(ValueError, match="missing asym_id"):
        build_layout_guidance_masks(torch.tensor([0, 1]), _layout(), "cyclic")
