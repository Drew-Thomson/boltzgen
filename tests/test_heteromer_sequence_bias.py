import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch

from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.data.parse.schema import parse_residue_constraints
from boltzgen.data import const

# For filter task
from boltzgen.task.predict.filter_identity_pairs import FilterHighIdentityPairs

# For anti-correlation
from boltzgen.model.modules.inverse_fold import (
    InverseFoldingDecoder,
    get_anti_correlation_weights,
)


def test_charge_bias_generation():
    entities, layout = expand_material_spec(
        copies=2,
        asym_unit=[
            {"type": "protein", "name": "A", "length": 5},
            {"type": "protein", "name": "B", "length": 5},
        ],
        heteromer_screening=True,
        topology="cyclic",
        charge_bias_strength=1.5
    )
    
    # Check that A has positive bias (R, K boosted, H excluded)
    ent_A = entities[0]["protein"]
    assert "residue_constraints" in ent_A
    weights_A = ent_A["residue_constraints"][0]["weights"]
    assert weights_A["R"] == pytest.approx(1.5 * 1.15)
    assert weights_A["K"] == pytest.approx(1.5 * 1.15)
    assert "H" not in weights_A
    assert weights_A["D"] < 0
    
    ent_B = entities[1]["protein"]
    weights_B = ent_B["residue_constraints"][0]["weights"]
    assert weights_B["D"] == 1.5
    assert weights_B["E"] == 1.5
    assert "H" not in weights_B
    assert weights_B["R"] < 0


def test_double_tape_charge_bias_surface_positions():
    entities, _ = expand_material_spec(
        copies=4,
        asym_unit=[
            {"type": "protein", "name": "A", "length": 9},
            {"type": "protein", "name": "B", "length": 9},
        ],
        heteromer_screening=False,
        topology="double_tape",
        charge_bias_strength=1.0,
    )
    ent_A = entities[0]["protein"]
    assert "residue_constraints" in ent_A
    assert ent_A["residue_constraints"][0]["position"] == "1,3,5,7,9"
    ent_B = entities[2]["protein"]
    assert "residue_constraints" in ent_B
    assert ent_B["residue_constraints"][0]["position"] == "1,3,5,7,9"


def test_parse_residue_constraints_with_bias():
    constraints_spec = [
        {
            "position": "1..5",
            "weights": {"R": 1.5, "D": -0.5}
        }
    ]
    mask = parse_residue_constraints(
        constraints_spec,
        chain_length=5,
        canonical_tokens=const.canonical_tokens,
        prot_letter_to_token=const.prot_letter_to_token,
    )
    
    idx_R = const.canonical_tokens.index("ARG")
    idx_D = const.canonical_tokens.index("ASP")
    
    assert mask[0, idx_R] == 1.5
    assert mask[0, idx_D] == -0.5
    assert mask[0, 0] == 0.0 # ALA


def test_anti_correlation_decoder():
    decoder_default = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
    )
    assert decoder_default.heteromer_anti_correlation is False
    assert decoder_default.anti_correlation_strength == 0.0

    decoder_auto = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
        anti_correlation_strength=2.0,
    )
    assert decoder_auto.heteromer_anti_correlation is True
    assert decoder_auto.anti_correlation_strength == 2.0

    decoder_explicit = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
        heteromer_anti_correlation=True,
        anti_correlation_strength=2.0,
    )
    assert decoder_explicit.heteromer_anti_correlation is True
    assert decoder_explicit.anti_correlation_strength == 2.0

    decoder_default_strength = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
        heteromer_anti_correlation=True,
    )
    assert decoder_default_strength.heteromer_anti_correlation is True
    assert decoder_default_strength.anti_correlation_strength == 20.0


def test_filter_identity_pairs_missing_dir():
    task = FilterHighIdentityPairs("does_not_exist_dir", "dummy_layout.json")
    # Should not raise
    task.run()


def test_get_anti_correlation_weights() -> None:
    """Test get_anti_correlation_weights for exact and blosum62 modes."""
    w_exact = get_anti_correlation_weights("exact")
    assert isinstance(w_exact, torch.Tensor)
    assert w_exact.shape == (20, 20)
    assert torch.allclose(w_exact, torch.eye(20, dtype=torch.float32))

    w_blosum = get_anti_correlation_weights("blosum62")
    assert isinstance(w_blosum, torch.Tensor)
    assert w_blosum.shape == (20, 20)
    letters = [const.prot_token_to_letter[t] for t in const.canonical_tokens]
    l_idx = letters.index("L")
    i_idx = letters.index("I")
    d_idx = letters.index("D")
    assert w_blosum[l_idx, l_idx] == 1.0
    assert w_blosum[l_idx, i_idx] > 0.0
    assert w_blosum[l_idx, d_idx] == 0.0


def test_anti_correlation_mode_decoder() -> None:
    """Test InverseFoldingDecoder anti_correlation_mode attribute and buffer."""
    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
        anti_correlation_strength=2.0,
        anti_correlation_mode="blosum62",
    )
    assert decoder.anti_correlation_mode == "blosum62"
    assert hasattr(decoder, "anti_correlation_weights")
    assert decoder.anti_correlation_weights.shape == (20, 20)


def test_1copy_heterodimer_anti_correlation() -> None:
    """Test anti-correlation penalty applies in 1-copy heterodimers (copies=1)."""
    torch.manual_seed(42)
    n = 6
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    chain_a = res_types[:3]
    chain_b = res_types[3:]
    for a, b in zip(chain_a, chain_b):
        assert a != b, f"Positions at same residue index should not match: {a} vs {b}"


def test_multicomponent_heteromer_anti_correlation() -> None:
    """Test anti-correlation applies across >2 symmetric groups (heterotrimer)."""
    torch.manual_seed(42)
    n = 3
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    msg = f"Expected 3 distinct residues across 3 partners, got {res_types}"
    assert len(set(res_types)) == 3, msg


def test_anti_correlation_fallback_feature_asym_id() -> None:
    """Test fallback to feature_asym_id when symmetric_group is 0."""
    torch.manual_seed(42)
    n = 4
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.zeros(1, n, dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    assert res_types[0] != res_types[2]
    assert res_types[1] != res_types[3]


def test_inverse_fold_restriction_pro_met() -> None:
    """Test that inverse_fold_restriction completely excludes PRO and MET."""
    torch.manual_seed(42)
    n = 4
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        inverse_fold_restriction=["CYS", "PRO", "MET"],
        double_tape_core_bias=5.0,
        double_tape_polar_bias=5.0,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    pro_idx = const.canonical_tokens.index("PRO") + const.canonicals_offset
    met_idx = const.canonical_tokens.index("MET") + const.canonicals_offset
    cys_idx = const.canonical_tokens.index("CYS") + const.canonicals_offset
    for r in res_types:
        assert r != pro_idx, "Proline was sampled despite being restricted!"
        assert r != met_idx, "Methionine was sampled despite being restricted!"
        assert r != cys_idx, "Cysteine was sampled despite being restricted!"


def test_double_tape_amphipathic_alternation() -> None:
    """Test that double_tape odd positions are outward/polar and even positions are inward/core."""
    torch.manual_seed(42)
    n = 8
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    # 4 residues for chain 0, 4 residues for chain 1
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        inverse_fold_restriction=["CYS", "PRO", "MET"],
        double_tape_core_bias=10.0,
        double_tape_polar_bias=10.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()

    core_tokens = {"ALA", "VAL", "ILE", "LEU", "PHE", "TRP"}
    polar_tokens = {
        "ARG", "LYS", "ASP", "GLU", "ASN", "GLN", "HIS", "THR", "SER"
    }

    for i, r in enumerate(res_types):
        tok = const.canonical_tokens[r - const.canonicals_offset]
        res_idx = feats["feature_residue_index"][0, i].item()
        if res_idx % 2 == 0:  # 1-indexed odd: outward
            assert tok in polar_tokens, f"Position {i} (res {res_idx+1}) expected polar, got {tok}"
        else:  # 1-indexed even: inward
            assert tok in core_tokens, f"Position {i} (res {res_idx+1}) expected core hydrophobic, got {tok}"


def test_antiparallel_heteromer_anti_correlation() -> None:
    """Test cross-partner anti-correlation mapping under antiparallel strand orientation."""
    torch.manual_seed(42)
    n = 6
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
        "is_antiparallel": torch.tensor([True]),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    chain_a = res_types[:3]
    chain_b = res_types[3:]
    # Under antiparallel alignment, index 0 of A contacts index 2 of B, etc.
    assert chain_a[0] != chain_b[2], f"Match at ends: {chain_a[0]} == {chain_b[2]}"
    assert chain_a[1] != chain_b[1], f"Match at centre: {chain_a[1]} == {chain_b[1]}"
    assert chain_a[2] != chain_b[0], f"Match at ends: {chain_a[2]} == {chain_b[0]}"


def test_surface_only_anti_correlation() -> None:
    """Test surface-only anti-correlation leaves core positions unpenalized."""
    torch.manual_seed(42)
    n = 8
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long),
        "feature_residue_index": torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long
        ),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        double_tape_core_bias=10.0,
        anti_correlation_strength=100.0,
        anti_correlation_surface_only=True,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    chain_a = res_types[:4]
    chain_b = res_types[4:]

    # Surface positions (even 0-indexed: 0, 2) must differ
    assert chain_a[0] != chain_b[0], f"Surface position 0 matched: {chain_a[0]}"
    assert chain_a[2] != chain_b[2], f"Surface position 2 matched: {chain_a[2]}"


def test_filter_identity_pairs_antiparallel_and_zoned(tmp_path: Path) -> None:
    """Test FilterHighIdentityPairs with antiparallel 4a+4b design from Futty et al."""
    layout_file = tmp_path / "material_layout.json"
    layout = {
        "topology": "double_tape",
        "chains": [
            {"chain_id": "A", "role": "protein", "partner_label": "A"},
            {"chain_id": "B", "role": "protein", "partner_label": "B"},
        ],
    }
    layout_file.write_text(json.dumps(layout))

    # Peptide 4a: EVWAKVKAT, Peptide 4b: TATVEAWVK
    st = gemmi.Structure()
    m = gemmi.Model("1")
    ch_a = gemmi.Chain("A")
    for idx, char in enumerate("EVWAKVKAT"):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, float(idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_a.add_residue(res)
    m.add_chain(ch_a)

    ch_b = gemmi.Chain("B")
    for idx, char in enumerate("TATVEAWVK"):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        # Reverse coordinate vector along Y for antiparallel orientation
        atom.pos = gemmi.Position(10.0, float(8 - idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_b.add_residue(res)
    m.add_chain(ch_b)
    st.add_model(m)

    cif_path = tmp_path / "design_4a_4b.cif"
    st.make_mmcif_document().write_file(str(cif_path))

    # Test zoned filtering: surface identity is 40% <= 50%, should survive!
    task_zoned = FilterHighIdentityPairs(
        str(tmp_path),
        str(layout_file),
        max_partner_identity=0.5,
        zoned=True,
        weight_core=0.0,
        weight_surface=1.0,
        orientation="auto",
    )
    task_zoned.run()
    assert cif_path.exists(), "4a+4b should survive zoned identity filtering!"


def test_filter_identity_pairs_parallel_zoned(tmp_path: Path) -> None:
    """Test FilterHighIdentityPairs with parallel peptide pair."""
    layout_file = tmp_path / "material_layout.json"
    layout = {
        "topology": "double_tape",
        "chains": [
            {"chain_id": "A", "role": "protein", "partner_label": "A"},
            {"chain_id": "B", "role": "protein", "partner_label": "B"},
        ],
    }
    layout_file.write_text(json.dumps(layout))

    # Parallel pair with identical core (V, A, V, A) but diverse surface (40% match)
    seq_a = "EVWAKVKAT"
    seq_b = "KVWAEVTAT"
    st = gemmi.Structure()
    m = gemmi.Model("1")
    ch_a = gemmi.Chain("A")
    for idx, char in enumerate(seq_a):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, float(idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_a.add_residue(res)
    m.add_chain(ch_a)

    ch_b = gemmi.Chain("B")
    for idx, char in enumerate(seq_b):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        # Parallel orientation along Y
        atom.pos = gemmi.Position(10.0, float(idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_b.add_residue(res)
    m.add_chain(ch_b)
    st.add_model(m)

    cif_path = tmp_path / "design_parallel.cif"
    st.make_mmcif_document().write_file(str(cif_path))

    # Zoned filtering: surface identity = 40% <= 50%, should survive
    task_zoned = FilterHighIdentityPairs(
        str(tmp_path),
        str(layout_file),
        max_partner_identity=0.5,
        zoned=True,
        weight_core=0.0,
        weight_surface=1.0,
        orientation="parallel",
    )
    task_zoned.run()
    assert cif_path.exists(), "Parallel pair with diverse surface should survive!"


def test_different_length_antiparallel_anti_correlation() -> None:
    """Test anti-correlation when partner chains have different lengths."""
    torch.manual_seed(42)
    n_a = 6
    n_b = 10
    n = n_a + n_b
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)
    sym_group = torch.cat(
        [torch.full((n_a,), 1, dtype=torch.long), torch.full((n_b,), 2, dtype=torch.long)]
    ).unsqueeze(0)
    asym_id = torch.cat(
        [torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)]
    ).unsqueeze(0)
    res_idx = torch.cat(
        [torch.arange(n_a, dtype=torch.long), torch.arange(n_b, dtype=torch.long)]
    ).unsqueeze(0)
    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": sym_group,
        "feature_asym_id": asym_id,
        "feature_residue_index": res_idx,
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
        "is_antiparallel": torch.tensor([True]),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=8, pair_dim=8, hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    chain_a = res_types[:n_a]
    chain_b = res_types[n_a:]

    # A has mid=2.5, B has mid=4.5
    # Antiparallel: A[0] (0) maps to round(4.5 - (0 - 2.5)) = round(7.0) = 7
    # A[5] (5) maps to round(4.5 - (5 - 2.5)) = round(2.0) = 2
    assert chain_a[0] != chain_b[7], f"Mapped pair matched: {chain_a[0]} == {chain_b[7]}"
    assert chain_a[5] != chain_b[2], f"Mapped pair matched: {chain_a[5]} == {chain_b[2]}"


def test_different_length_zoned_filter_identity_pairs(tmp_path: Path) -> None:
    """Test FilterHighIdentityPairs with different strand lengths (len 9 and len 13)."""
    layout_file = tmp_path / "material_layout.json"
    layout = {
        "topology": "double_tape",
        "chains": [
            {"chain_id": "A", "role": "protein", "partner_label": "A"},
            {"chain_id": "B", "role": "protein", "partner_label": "B"},
        ],
    }
    layout_file.write_text(json.dumps(layout))

    # A is len 9 (EVWAKVKAT), B is len 13 with overhangs (GSTATVEAWVKSK)
    seq_a = "EVWAKVKAT"
    seq_b = "GSTATVEAWVKSK"
    st = gemmi.Structure()
    m = gemmi.Model("1")
    ch_a = gemmi.Chain("A")
    for idx, char in enumerate(seq_a):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, float(idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_a.add_residue(res)
    m.add_chain(ch_a)

    ch_b = gemmi.Chain("B")
    for idx, char in enumerate(seq_b):
        res = gemmi.Residue()
        res.name = const.prot_letter_to_token[char]
        res.seqid = gemmi.SeqId(str(idx + 1))
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        # Reverse along Y for antiparallel orientation
        atom.pos = gemmi.Position(10.0, float(len(seq_b) - 1 - idx) * 3.8, 0.0)
        res.add_atom(atom)
        ch_b.add_residue(res)
    m.add_chain(ch_b)
    st.add_model(m)

    cif_path = tmp_path / "design_diff_lengths.cif"
    st.make_mmcif_document().write_file(str(cif_path))

    # Zoned filtering across unequal lengths: surface identity = 40% <= 50%, survives!
    task = FilterHighIdentityPairs(
        str(tmp_path),
        str(layout_file),
        max_partner_identity=0.5,
        zoned=True,
        weight_core=0.0,
        weight_surface=1.0,
        orientation="antiparallel",
    )
    task.run()
    assert cif_path.exists(), "Unequal length pair should survive zoned filtering!"


