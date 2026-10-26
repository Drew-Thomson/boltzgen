import torch

from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.model.modules.materials import (
    compute_anchored_ligand_position,
    fit_rigid_transform,
    resolve_anchor_residue_index,
    resolve_protein_metal_clashes,
)


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


def test_resolve_anchor_residue_index_specific_and_clamped():
    # 1-indexed to 0-indexed conversion
    assert resolve_anchor_residue_index(10, 20) == 9
    assert resolve_anchor_residue_index("5", 20) == 4
    # Clamping
    assert resolve_anchor_residue_index(99, 20) == 19
    assert resolve_anchor_residue_index(0, 20) == 0


def test_resolve_anchor_residue_index_random():
    gen = torch.Generator().manual_seed(42)
    # L = 20, random should sample in [2, 17] (1-indexed 3..18)
    samples = [
        resolve_anchor_residue_index("random", 20, generator=gen)
        for _ in range(50)
    ]
    assert all(2 <= s <= 17 for s in samples)
    # Ensure diverse sampling
    assert len(set(samples)) > 1


def test_compute_anchored_ligand_position_outward_radial():
    ca_coord = torch.tensor([10.0, 0.0, 6.0])
    assembly_com = torch.tensor([0.0, 0.0, 0.0])
    # Target position should be at r = 10.0 + 4.0 = 14.0, z = 6.0
    pos = compute_anchored_ligand_position(
        ca_coords=ca_coord,
        assembly_com=assembly_com,
        topology="cyclic",
        anchor_distance=4.0,
    )
    expected = torch.tensor([14.0, 0.0, 6.0])
    assert torch.allclose(pos, expected, atol=1e-5)


def test_compute_anchored_ligand_position_sidechain():
    ca_coord = torch.tensor([10.0, 0.0, 6.0])
    assembly_com = torch.tensor([0.0, 0.0, 0.0])
    sidechain_vec = torch.tensor([0.0, 1.0, 0.0])
    pos = compute_anchored_ligand_position(
        ca_coords=ca_coord,
        assembly_com=assembly_com,
        topology="cyclic",
        anchor_distance=3.5,
        sidechain_vec=sidechain_vec,
        direction_mode="sidechain",
    )
    expected = torch.tensor([10.0, 3.5, 6.0])
    assert torch.allclose(pos, expected, atol=1e-5)


def test_expand_material_spec_retains_anchor_metadata():
    entities, layout = expand_material_spec(
        2,
        [
            {"type": "protein", "name": "p", "length": 15},
            {
                "type": "ligand",
                "ccd": "ZN",
                "ligands_per_peptide": 1,
                "attach_to": "p",
                "anchor_residue": 8,
                "anchor_distance": 4.5,
                "anchor_direction": "outward",
            },
        ],
    )
    # Check that builder keys are not in entities
    for entity_dict in entities:
        for val in entity_dict.values():
            assert "anchor_residue" not in val
            assert "anchor_distance" not in val

    # Check that layout chains retain anchor metadata
    ligand_records = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligand_records) == 2
    for rec in ligand_records:
        assert rec["anchor_residue"] == 8
        assert rec["anchor_distance"] == 4.5
        assert rec["anchor_direction"] == "outward"


def test_expand_material_spec_defaults_to_random_anchor():
    _, layout = expand_material_spec(
        2,
        [
            {"type": "protein", "name": "p", "length": 15},
            {
                "type": "ligand",
                "ccd": "ZN",
                "ligands_per_peptide": 1,
                "attach_to": "p",
            },
        ],
    )
    ligand_records = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligand_records) == 2
    for rec in ligand_records:
        assert rec["anchor_residue"] == "random"
        assert rec["anchor_distance"] == 4.0
        assert rec["anchor_direction"] == "outward"


def test_boolean_token_mask_argmax_handling():
    # token_to_rep_atom and token_to_bb4_atoms are bool tensors in Boltz
    rep_mask = torch.tensor([False, False, True, False])
    # .argmax() directly on bool fails in PyTorch with RuntimeError
    idx = rep_mask.int().argmax()
    assert idx.item() == 2

    bb4_tok = torch.tensor([
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ])
    n_idx = bb4_tok[0].int().argmax()
    ca_idx = bb4_tok[1].int().argmax()
    c_idx = bb4_tok[2].int().argmax()
    assert n_idx.item() == 0
    assert ca_idx.item() == 1
    assert c_idx.item() == 2


def test_resolve_protein_metal_clashes_repels_embedded_metal():
    # Protein atom right next to metal (clashing: 0.5 A)
    protein = torch.tensor([[10.0, 0.0, 5.0], [11.0, 0.0, 5.0]])
    metal = torch.tensor([10.5, 0.0, 5.0])
    fixed_metal = resolve_protein_metal_clashes(
        metal, protein, assembly_com=torch.tensor([0.0, 0.0, 0.0]), min_distance=2.8
    )
    dists = torch.norm(protein - fixed_metal, dim=-1)
    assert dists.min().item() >= 2.8


def test_compute_anchored_ligand_position_inward_sidechain_avoids_backbone():
    ca_coord = torch.tensor([8.0, 0.0, 5.0]) # inner face of helix at r=10
    assembly_com = torch.tensor([0.0, 0.0, 0.0])
    # Sidechain points inward toward (0, 0, 0)
    sidechain_vec = torch.tensor([-1.0, 0.2, 0.0])
    pos = compute_anchored_ligand_position(
        ca_coords=ca_coord,
        assembly_com=assembly_com,
        topology="cyclic",
        anchor_distance=4.0,
        sidechain_vec=sidechain_vec,
        direction_mode="outward",
    )
    # Target should project along tangent ([0, 1, 0]) instead of cutting through backbone
    disp = pos - ca_coord
    assert abs(disp[0].item()) < 0.1 # does NOT penetrate radially through helix
    assert torch.isclose(torch.norm(disp), torch.tensor(4.0), atol=1e-5)
