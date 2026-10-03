import pytest
import numpy as np
import torch
from pathlib import Path

from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.data.parse.schema import parse_residue_constraints
from boltzgen.data import const

# For filter task
from boltzgen.task.predict.filter_identity_pairs import FilterHighIdentityPairs

# For anti-correlation
from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder


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
    
    # Check that A has positive bias (R, K, H boosted)
    ent_A = entities[0]["protein"]
    assert "residue_constraints" in ent_A
    weights_A = ent_A["residue_constraints"][0]["weights"]
    assert weights_A["R"] == 1.5
    assert weights_A["D"] < 0
    
    ent_B = entities[1]["protein"]
    weights_B = ent_B["residue_constraints"][0]["weights"]
    assert weights_B["D"] == 1.5
    assert weights_B["R"] < 0


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
    decoder = InverseFoldingDecoder(
        atom_s=1, atom_z=1, token_s=1, token_z=1,
        node_dim=1, pair_dim=1, hidden_dim=1,
        heteromer_anti_correlation=True,
        anti_correlation_strength=2.0
    )
    assert decoder.heteromer_anti_correlation is True
    assert decoder.anti_correlation_strength == 2.0


def test_filter_identity_pairs_missing_dir():
    task = FilterHighIdentityPairs("does_not_exist_dir", "dummy_layout.json")
    # Should not raise
    task.run()
