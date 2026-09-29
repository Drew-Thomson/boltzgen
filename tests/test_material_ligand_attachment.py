import torch

from boltzgen.model.modules.materials import fit_rigid_transform


def test_parent_transform_carries_ligand_and_preserves_relative_geometry():
    protein = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    ligand = torch.tensor([[0.2, 0.3, 1.5], [0.8, 0.2, 1.8]])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([4.0, -2.0, 3.0])
    moved_protein = protein @ rotation + translation
    moved_ligand_expected = ligand @ rotation + translation

    fitted_rotation, fitted_translation = fit_rigid_transform(protein, moved_protein)
    moved_ligand = ligand @ fitted_rotation + fitted_translation

    assert torch.allclose(moved_ligand, moved_ligand_expected, atol=1e-6)
    assert torch.allclose(
        torch.cdist(moved_protein, moved_ligand),
        torch.cdist(protein, ligand),
        atol=1e-6,
    )
