"""Tests for ligand confidence, spatial clash, and coordination metrics."""

import json
from pathlib import Path
import gemmi
import numpy as np
import pandas as pd
import pytest

from boltzgen.task.analyze.analyze_utils import compute_ligand_metrics
from boltzgen.task.analyze.filter_target_candidates import FilterTargetCandidates
from boltzgen.task.filter.material_presets import ligand_counter_screen_filtering_overrides


def _create_synthetic_cif(
    path: Path,
    cu_positions: list[tuple[float, float, float]],
    cu_b_isos: list[float] | None = None,
    his_n_pos: tuple[float, float, float] | None = None,
) -> Path:
    """Create a minimal MMCIF file containing protein and CU ligands."""
    doc = gemmi.cif.Document()
    block = doc.add_new_block("structure")

    # Construct gemmi structure
    st = gemmi.Structure()
    model = gemmi.Model("1")

    # Add protein chain A
    chain_a = gemmi.Chain("A")
    res1 = gemmi.Residue()
    res1.name = "HIS"
    res1.seqid = gemmi.SeqId("1")

    ca = gemmi.Atom()
    ca.name = "CA"
    ca.element = gemmi.Element("C")
    ca.pos = gemmi.Position(0.0, 0.0, 0.0)
    ca.b_iso = 0.8
    res1.add_atom(ca)

    if his_n_pos is not None:
        nd1 = gemmi.Atom()
        nd1.name = "ND1"
        nd1.element = gemmi.Element("N")
        nd1.pos = gemmi.Position(*his_n_pos)
        nd1.b_iso = 0.8
        res1.add_atom(nd1)

    chain_a.add_residue(res1)
    model.add_chain(chain_a)

    # Add CU ligands as separate chains
    if cu_b_isos is None:
        cu_b_isos = [0.5] * len(cu_positions)

    for idx, (pos, b_iso) in enumerate(zip(cu_positions, cu_b_isos)):
        chain_name = chr(ord("B") + idx)
        cu_chain = gemmi.Chain(chain_name)
        cu_res = gemmi.Residue()
        cu_res.name = "CU"
        cu_res.seqid = gemmi.SeqId("1")

        cu_atom = gemmi.Atom()
        cu_atom.name = "CU"
        cu_atom.element = gemmi.Element("Cu")
        cu_atom.pos = gemmi.Position(*pos)
        cu_atom.b_iso = b_iso
        cu_res.add_atom(cu_atom)

        cu_chain.add_residue(cu_res)
        model.add_chain(cu_chain)

    st.add_model(model)
    st.setup_entities()
    doc = st.make_mmcif_document()
    doc.write_file(str(path))
    return path


def test_compute_ligand_metrics_well_spaced(tmp_path):
    cif_file = tmp_path / "well_spaced.cif"
    _create_synthetic_cif(
        cif_file,
        cu_positions=[(0.0, 5.0, 0.0), (0.0, -5.0, 0.0)],
        cu_b_isos=[0.40, 0.60],
        his_n_pos=(0.0, 3.0, 0.0),  # 2.0 A from first Cu, 8.0 A from second Cu
    )

    metrics = compute_ligand_metrics(cif_file, coord_shell_radius=2.8, clash_threshold=3.5)
    assert metrics["min_ligand_distance"] == pytest.approx(10.0)
    assert metrics["num_ligand_clashes"] == 0
    assert metrics["neg_num_ligand_clashes"] == 0
    assert metrics["ligand_clash_penalty"] == pytest.approx(0.0)
    assert metrics["mean_ligand_plddt"] == pytest.approx(0.50)
    assert metrics["min_ligand_plddt"] == pytest.approx(0.40)
    assert metrics["mean_ligand_coordination"] == pytest.approx(0.5)  # one has 1, other has 0
    assert metrics["min_ligand_coordination"] == 0


def test_compute_ligand_metrics_clashing(tmp_path):
    cif_file = tmp_path / "clashing.cif"
    _create_synthetic_cif(
        cif_file,
        cu_positions=[(0.0, 1.0, 0.0), (0.0, 3.2, 0.0)],  # dist = 2.2 A
        cu_b_isos=[0.25, 0.35],
    )

    metrics = compute_ligand_metrics(cif_file, coord_shell_radius=2.8, clash_threshold=3.5)
    assert metrics["min_ligand_distance"] == pytest.approx(2.2)
    assert metrics["num_ligand_clashes"] == 1
    assert metrics["neg_num_ligand_clashes"] == -1
    assert metrics["ligand_clash_penalty"] == pytest.approx(1.3)  # 3.5 - 2.2
    assert metrics["mean_ligand_plddt"] == pytest.approx(0.30)
    assert metrics["min_ligand_plddt"] == pytest.approx(0.25)


def test_compute_ligand_metrics_no_ligands(tmp_path):
    cif_file = tmp_path / "no_ligands.cif"
    _create_synthetic_cif(cif_file, cu_positions=[])

    metrics = compute_ligand_metrics(cif_file)
    assert metrics == {}


def test_filter_target_candidates_with_min_ligand_distance(tmp_path):
    design_dir = tmp_path / "designs"
    design_dir.mkdir(parents=True)

    csv_path = design_dir / "aggregate_metrics_analyze.csv"
    pd.DataFrame([
        {
            "id": "spec_clean",
            "lattice_rmsd_refolded": 2.5,
            "neg_lattice_rmsd_refolded": -2.5,
            "complex_plddt": 0.60,
            "min_ligand_distance": 8.5,
        },
        {
            "id": "spec_clash",
            "lattice_rmsd_refolded": 2.0,
            "neg_lattice_rmsd_refolded": -2.0,
            "complex_plddt": 0.65,
            "min_ligand_distance": 2.3,
        },
        {
            "id": "spec_single_ligand",
            "lattice_rmsd_refolded": 2.2,
            "neg_lattice_rmsd_refolded": -2.2,
            "complex_plddt": 0.62,
            "min_ligand_distance": float("nan"),
        },
    ]).to_csv(csv_path, index=False)

    task = FilterTargetCandidates(
        design_dir=str(design_dir),
        max_rmsd=4.0,
        min_ligand_distance=3.5,
    )
    task.run()

    candidates_path = design_dir / "counter_screen_candidates.json"
    assert candidates_path.exists()
    data = json.loads(candidates_path.read_text())
    assert data["min_ligand_distance"] == 3.5
    # spec_clean and spec_single_ligand pass, spec_clash is culled
    assert "spec_clean" in data["candidate_ids"]
    assert "spec_single_ligand" in data["candidate_ids"]
    assert "spec_clash" not in data["candidate_ids"]


def test_ligand_counter_screen_filtering_overrides_includes_ligand_metrics():
    overrides = ligand_counter_screen_filtering_overrides()
    assert "min_ligand_distance" in overrides
    assert "neg_num_ligand_clashes" in overrides
    assert "ligand_iptm" in overrides
    assert "mean_ligand_coordination" in overrides
    assert "mean_ligand_plddt" in overrides
