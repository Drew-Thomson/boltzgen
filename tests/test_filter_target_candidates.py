"""Tests for pre-counter-screen target candidate culling."""

from __future__ import annotations

import json

import pandas as pd

from boltzgen.task.analyze.filter_target_candidates import (
    FilterTargetCandidates,
    target_rmsd,
)


def test_double_tape_candidate_rmsd_uses_local_metric() -> None:
    frame = pd.DataFrame(
        {
            "local_double_tape_rmsd": [1.0, 6.0],
            "neg_lattice_rmsd_refolded": [-100.0, -0.0],
        }
    )

    assert target_rmsd(frame, "double_tape").tolist() == [1.0, 6.0]


def test_double_tape_candidate_rmsd_prefers_local_step_and_geom() -> None:
    frame_step = pd.DataFrame(
        {
            "local_step_rmsd": [1.5, 3.0],
            "lattice_rmsd_geom": [2.0, 4.0],
            "local_double_tape_rmsd": [16.0, 18.0],
        }
    )
    assert target_rmsd(frame_step, "double_tape").tolist() == [1.5, 3.0]

    frame_geom = pd.DataFrame(
        {
            "lattice_rmsd_geom": [2.0, 4.0],
            "local_double_tape_rmsd": [16.0, 18.0],
        }
    )
    assert target_rmsd(frame_geom, "double_tape").tolist() == [2.0, 4.0]


def test_double_tape_culling_gates_counter_screens_by_local_rmsd(tmp_path) -> None:
    pd.DataFrame(
        {
            "id": ["passes_local_rmsd", "fails_local_rmsd", "missing_local_rmsd"],
            "local_double_tape_rmsd": [2.0, 6.0, float("nan")],
            "neg_lattice_rmsd_refolded": [-2.0, -6.0, -1.0],
        }
    ).to_csv(tmp_path / "aggregate_metrics_analyze.csv", index=False)

    FilterTargetCandidates(
        design_dir=str(tmp_path), topology="double_tape", max_rmsd=2.5
    ).run()

    output = json.loads((tmp_path / "counter_screen_candidates.json").read_text())
    assert output["max_rmsd"] == 2.5
    assert output["candidate_ids"] == ["passes_local_rmsd"]


def test_cyclic_ligand_culling_drops_template_rmsd_and_gates_symmetry_and_coord(tmp_path) -> None:
    pd.DataFrame(
        {
            "id": ["relaxed_ring", "broken_ring"],
            "bb_rmsd": [25.0, 2.0],
            "neg_lattice_rmsd_refolded": [-25.0, -2.0],
            "neg_sym_chain_rmsd": [-0.6, -3.0],
            "neg_sym_ligand_rmsd": [-0.3, -2.0],
            "neg_ligand_coord_geom_dev": [-6.0, -25.0],
            "neg_ligand_coord_bond_dev": [-0.1, -0.8],
            "ligand_min_coord_number": [4.0, 1.0],
        }
    ).to_csv(tmp_path / "aggregate_metrics_analyze.csv", index=False)

    FilterTargetCandidates(design_dir=str(tmp_path), topology="cyclic", max_rmsd=None).run()

    output = json.loads((tmp_path / "counter_screen_candidates.json").read_text())
    assert output["max_rmsd"] is None
    assert output["candidate_ids"] == ["relaxed_ring"]


def test_initial_screen_gates_ss_orientation_coord_and_caps_max_candidates(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "id": "best_design",
                "bb_rmsd": 22.0,
                "complex_plddt": 0.90,
                "ligand_iptm": 0.85,
                "mean_ligand_plddt": 0.88,
                "ss_accuracy_refolded": 0.95,
                "sym_chain_rmsd": 0.4,
                "sym_ligand_rmsd": 0.2,
                "sym_ligand_orbit_ok": 1.0,
                "sym_orientation_match": 1.0,
                "ligand_min_coord_number": 4.0,
                "ligand_coord_bond_dev": 0.1,
                "ligand_coord_geom_dev": 5.0,
            },
            {
                "id": "second_design",
                "bb_rmsd": 19.0,
                "complex_plddt": 0.82,
                "ligand_iptm": 0.78,
                "mean_ligand_plddt": 0.80,
                "ss_accuracy_refolded": 0.85,
                "sym_chain_rmsd": 0.8,
                "sym_ligand_rmsd": 0.4,
                "sym_ligand_orbit_ok": 1.0,
                "sym_orientation_match": 1.0,
                "ligand_min_coord_number": 4.0,
                "ligand_coord_bond_dev": 0.2,
                "ligand_coord_geom_dev": 10.0,
            },
            {
                "id": "third_design",
                "bb_rmsd": 1.5,
                "complex_plddt": 0.75,
                "ligand_iptm": 0.72,
                "mean_ligand_plddt": 0.74,
                "ss_accuracy_refolded": 0.75,
                "sym_chain_rmsd": 1.2,
                "sym_ligand_rmsd": 0.6,
                "sym_ligand_orbit_ok": 1.0,
                "sym_orientation_match": 1.0,
                "ligand_min_coord_number": 3.0,
                "ligand_coord_bond_dev": 0.3,
                "ligand_coord_geom_dev": 15.0,
            },
            {
                "id": "unfolded_helix",
                "bb_rmsd": 1.0,
                "complex_plddt": 0.92,
                "ligand_iptm": 0.90,
                "mean_ligand_plddt": 0.90,
                "ss_accuracy_refolded": 0.30,
                "sym_chain_rmsd": 0.3,
                "sym_ligand_rmsd": 0.1,
                "sym_ligand_orbit_ok": 1.0,
                "sym_orientation_match": 1.0,
                "ligand_min_coord_number": 4.0,
                "ligand_coord_bond_dev": 0.1,
                "ligand_coord_geom_dev": 4.0,
            },
            {
                "id": "wrong_orientation",
                "bb_rmsd": 1.0,
                "complex_plddt": 0.92,
                "ligand_iptm": 0.90,
                "mean_ligand_plddt": 0.90,
                "ss_accuracy_refolded": 0.95,
                "sym_chain_rmsd": 0.3,
                "sym_ligand_rmsd": 0.1,
                "sym_ligand_orbit_ok": 1.0,
                "sym_orientation_match": 0.0,
                "ligand_min_coord_number": 4.0,
                "ligand_coord_bond_dev": 0.1,
                "ligand_coord_geom_dev": 4.0,
            },
        ]
    ).to_csv(tmp_path / "aggregate_metrics_analyze.csv", index=False)

    FilterTargetCandidates(
        design_dir=str(tmp_path),
        topology="cyclic",
        use_template_rmsd=False,
        rank_initial_candidates=True,
        max_candidates=2,
    ).run()

    output = json.loads((tmp_path / "counter_screen_candidates.json").read_text())
    assert output["num_passed_gates"] == 3
    assert output["candidate_ids"] == ["best_design", "second_design"]
    ranked_df = pd.read_csv(tmp_path / "initial_screen_ranked.csv")
    assert "bb_rmsd" in ranked_df.columns
    assert ranked_df.iloc[0]["id"] == "best_design"
