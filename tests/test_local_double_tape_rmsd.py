"""Tests for local double tape metrics and two-chain partner mapping."""

# ruff: noqa: INP001, ANN201, PLR2004

import torch

from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.task.analyze.analyze_utils import compute_chi_outward
from boltzgen.task.analyze.material_metrics import (
    compute_two_chain_alternating_metrics,
)


def test_local_double_tape_rmsd() -> None:
    ca_coords = torch.randn(4, 3)
    design_seq = "AWRA"
    # Residue 1 (W) is outward, Residue 2 (R) is outward.
    # j-i = 2-1 = 1 (odd) -> alternating!
    classification = ["inward", "outward", "outward", "inward"]

    # distance matrix
    ca_coords[1] = torch.tensor([0.0, 0.0, 0.0])
    ca_coords[2] = torch.tensor([1.0, 0.0, 0.0])  # dist = 1.0 < 8.0

    chi = compute_chi_outward(ca_coords, design_seq, classification)
    assert chi == 1.0  # 1 alternating contact, total 1 contact


def test_two_chain_double_tape_partner_mapping() -> None:
    # Design spec with 2 chains in asym_unit without counter screen
    asym_unit = [
        {"type": "protein", "length": 9, "symmetric_group": 1},
        {"type": "protein", "length": 9, "symmetric_group": 2},
    ]
    entities, layout = expand_material_spec(4, asym_unit, topology="double_tape")
    # 4 placements * 2 chains = 8 chains
    assert len(entities) == 8
    # Partners alternate 0, 1, 0, 1, 0, 1, 0, 1
    partner_slots = {
        idx: record["protein_slot_index"]
        for idx, record in enumerate(layout["chains"])
        if record["role"] == "protein"
    }
    assert partner_slots == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}

    # Verify that identical ideal/refold yields zero RMSD and 1.0 alternation ratio
    num_atoms = 8 * 9 * 4
    coords = torch.zeros((num_atoms, 3))
    chain_ids = torch.zeros(num_atoms, dtype=torch.long)
    for c in range(8):
        start = c * 36
        end = (c + 1) * 36
        chain_ids[start:end] = c
        coords[start:end, 2] = (c // 2) * 4.8
        coords[start:end, 0] = (c % 2) * 8.5
    bb_mask = torch.ones(num_atoms, dtype=torch.bool)

    metrics = compute_two_chain_alternating_metrics(
        coords_refold=coords,
        coords_ideal=coords,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert metrics["lattice_rmsd_refolded"] == 0.0
    assert metrics["two_chain_alternation_ratio"] == 1.0
