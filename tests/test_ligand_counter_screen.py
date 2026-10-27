"""Tests for ligand counter-screen preparation and structure extraction."""

import json
from pathlib import Path
import numpy as np
import pytest

from boltzgen.data import const
from boltzgen.data.data import Atom, Bond, Chain, Coords, Ensemble, Interface, Residue, Structure
from boltzgen.task.predict.ligand_counter_screen import (
    PrepareLigandCounterScreens,
    extract_protein_only_structure,
)


class MockParseResult:
    """Mock parsed structure containing protein and ligand chains."""

    def __init__(self, data: Structure, sequences: dict[str, str]):
        self.data = data
        self.sequences = sequences


def _create_mock_protein_ligand_structure() -> tuple[Structure, dict[str, str]]:
    """Create a minimal Structure with 2 protein chains and 1 ligand chain."""
    # Chain A: 5 residues (20 atoms)
    # Chain B: 5 residues (20 atoms)
    # Chain L (ligand): 1 residue (1 Cu atom)
    total_atoms = 20 + 20 + 1
    total_residues = 5 + 5 + 1
    total_chains = 3

    coords = np.zeros(total_atoms, dtype=Coords)
    coords["coords"] = np.random.randn(total_atoms, 3).astype(np.float32)

    atoms = np.zeros(total_atoms, dtype=Atom)
    atoms["coords"] = coords["coords"]

    residues = np.zeros(total_residues, dtype=Residue)
    chains = np.zeros(total_chains, dtype=Chain)

    # Chain A (protein)
    chains[0]["name"] = "A"
    chains[0]["asym_id"] = 0
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 1
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 5
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = 20
    chains[0]["mol_type"] = const.chain_type_ids["PROTEIN"]

    for r in range(5):
        residues[r]["name"] = "ALA"
        residues[r]["res_idx"] = r
        residues[r]["atom_idx"] = r * 4
        residues[r]["atom_num"] = 4
        residues[r]["atom_center"] = r * 4 + 1
        atoms["name"][r * 4] = "N"
        atoms["name"][r * 4 + 1] = "CA"
        atoms["name"][r * 4 + 2] = "C"
        atoms["name"][r * 4 + 3] = "O"

    # Chain B (protein)
    chains[1]["name"] = "B"
    chains[1]["asym_id"] = 1
    chains[1]["entity_id"] = 0
    chains[1]["sym_id"] = 1
    chains[1]["res_idx"] = 5
    chains[1]["res_num"] = 5
    chains[1]["atom_idx"] = 20
    chains[1]["atom_num"] = 20
    chains[1]["mol_type"] = const.chain_type_ids["PROTEIN"]

    for r in range(5):
        res_idx = 5 + r
        residues[res_idx]["name"] = "ALA"
        residues[res_idx]["res_idx"] = res_idx
        residues[res_idx]["atom_idx"] = 20 + r * 4
        residues[res_idx]["atom_num"] = 4
        residues[res_idx]["atom_center"] = 20 + r * 4 + 1
        atoms["name"][20 + r * 4] = "N"
        atoms["name"][20 + r * 4 + 1] = "CA"
        atoms["name"][20 + r * 4 + 2] = "C"
        atoms["name"][20 + r * 4 + 3] = "O"

    # Chain L (ligand Cu)
    chains[2]["name"] = "L"
    chains[2]["asym_id"] = 2
    chains[2]["entity_id"] = 1
    chains[2]["sym_id"] = 1
    chains[2]["res_idx"] = 10
    chains[2]["res_num"] = 1
    chains[2]["atom_idx"] = 40
    chains[2]["atom_num"] = 1
    chains[2]["mol_type"] = const.chain_type_ids["NONPOLYMER"]

    residues[10]["name"] = "CU"
    residues[10]["res_idx"] = 10
    residues[10]["atom_idx"] = 40
    residues[10]["atom_num"] = 1
    residues[10]["atom_center"] = 40
    atoms["name"][40] = "CU"

    bonds = np.zeros(0, dtype=Bond)
    interfaces = np.zeros(0, dtype=Interface)
    ensemble = np.zeros(1, dtype=Ensemble)
    mask = np.ones(total_atoms, dtype=bool)

    structure = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
    )
    sequences = {"A": "AAAAA", "B": "AAAAA"}
    return structure, sequences


def test_extract_protein_only_structure():
    data, sequences = _create_mock_protein_ligand_structure()
    parsed = MockParseResult(data, sequences)

    apo = extract_protein_only_structure(parsed)
    # Check that ligand chain L was removed and only chains A and B remain
    assert len(apo.chains) == 2
    assert list(apo.chains["name"]) == ["A", "B"]
    assert len(apo.residues) == 10
    assert len(apo.atoms) == 40
    assert all(apo.atoms["name"] != "CU")


def test_prepare_ligand_counter_screens_task(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "apo_screens"
    input_dir.mkdir()

    # Create mock CIF input files
    (input_dir / "candidate_01.cif").write_text("dummy cif 1")
    (input_dir / "candidate_02.cif").write_text("dummy cif 2")

    mock_struct, mock_seqs = _create_mock_protein_ligand_structure()
    mock_parsed = MockParseResult(mock_struct, mock_seqs)

    # Monkeypatch parse_mmcif to return mock parsed structure
    monkeypatch.setattr(
        "boltzgen.task.predict.ligand_counter_screen.parse_mmcif",
        lambda *args, **kwargs: mock_parsed,
    )

    task = PrepareLigandCounterScreens(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )
    task.run()

    # Verify output files
    assert (output_dir / "candidate_01_apo.cif").exists()
    assert (output_dir / "candidate_01_apo.npz").exists()
    assert (output_dir / "candidate_02_apo.cif").exists()
    assert (output_dir / "candidate_02_apo.npz").exists()

    mapping_file = output_dir / "counter_screen_mapping.json"
    assert mapping_file.exists()
    mapping = json.loads(mapping_file.read_text())
    assert mapping == {
        "candidate_01": {"apo": "candidate_01_apo"},
        "candidate_02": {"apo": "candidate_02_apo"},
    }
