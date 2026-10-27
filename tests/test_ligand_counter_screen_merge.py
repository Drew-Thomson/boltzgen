"""Tests for ligand-free counter-screen metrics merging and contrast calculations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from boltzgen.task.analyze.merge_counter_screen_metrics import MergeCounterScreenMetrics


def test_merge_counter_screen_metrics_apo(tmp_path, monkeypatch):
    design_dir = tmp_path / "design"
    screen_dir = tmp_path / "screens"
    design_dir.mkdir(parents=True)
    screen_dir.mkdir(parents=True)

    # Set up mapping with apo
    mapping = {
        "candidate_01": {"apo": "candidate_01_apo"},
        "candidate_02": {"apo": "candidate_02_apo"},
    }
    (screen_dir / "counter_screen_mapping.json").write_text(json.dumps(mapping))

    # Set up apo folding prediction NPZs
    # Candidate 01: good holo (plddt=85, iptm=0.8, rmsd=1.5), poor apo (plddt=45, iptm=0.2)
    # Candidate 02: folds equally well in apo (plddt=80, iptm=0.75)
    np.savez_compressed(
        screen_dir / "candidate_01_apo.npz",
        complex_plddt=np.asarray([45.0]),
        protein_iptm=np.asarray([0.20]),
        design_iptm=np.asarray([0.18]),
        min_interaction_pae=np.asarray([18.0]),
    )
    np.savez_compressed(
        screen_dir / "candidate_02_apo.npz",
        complex_plddt=np.asarray([80.0]),
        protein_iptm=np.asarray([0.75]),
        design_iptm=np.asarray([0.70]),
        min_interaction_pae=np.asarray([5.0]),
    )

    # Candidate 01 has apo_rmsd = 12.0, Candidate 02 has apo_rmsd = 1.6
    apo_rmsds = {
        ("candidate_01_apo", "candidate_01"): 12.0,
        ("candidate_02_apo", "candidate_02"): 1.6,
        ("candidate_01", "candidate_01_apo"): 11.5,
        ("candidate_02", "candidate_02_apo"): 0.5,
    }
    monkeypatch.setattr(
        "boltzgen.task.analyze.merge_counter_screen_metrics._compute_cif_backbone_rmsd",
        lambda refold, design, *args, **kwargs: apo_rmsds.get(
            (refold.stem, design.stem), 99.0
        ),
    )

    # Holo metrics in CSV
    csv_path = design_dir / "aggregate_metrics_analyze.csv"
    pd.DataFrame([
        {
            "id": "candidate_01",
            "complex_plddt": 85.0,
            "protein_iptm": 0.80,
            "design_iptm": 0.78,
            "min_interaction_pae": 4.0,
            "rmsd": 1.5,
            "neg_lattice_rmsd_refolded": -1.5,
        },
        {
            "id": "candidate_02",
            "complex_plddt": 82.0,
            "protein_iptm": 0.76,
            "design_iptm": 0.72,
            "min_interaction_pae": 4.5,
            "rmsd": 1.6,
            "neg_lattice_rmsd_refolded": -1.6,
        },
    ]).to_csv(csv_path, index=False)

    task = MergeCounterScreenMetrics(str(design_dir), str(screen_dir))
    task.run()

    df = pd.read_csv(csv_path)
    assert len(df) == 2

    row1 = df[df["id"] == "candidate_01"].iloc[0]
    assert row1["apo_complex_plddt"] == pytest.approx(45.0)
    assert row1["apo_protein_iptm"] == pytest.approx(0.20)
    assert row1["apo_rmsd"] == pytest.approx(12.0)
    assert row1["delta_complex_plddt_vs_apo"] == pytest.approx(40.0)  # 85 - 45
    assert row1["delta_protein_iptm_vs_apo"] == pytest.approx(0.60)   # 0.80 - 0.20
    assert row1["delta_rmsd_vs_apo"] == pytest.approx(10.5)           # 12.0 - 1.5
    assert row1["delta_neg_min_interaction_pae_vs_apo"] == pytest.approx(14.0)  # 18 - 4

    row2 = df[df["id"] == "candidate_02"].iloc[0]
    assert row2["apo_complex_plddt"] == pytest.approx(80.0)
    assert row2["apo_protein_iptm"] == pytest.approx(0.75)
    assert row2["apo_rmsd"] == pytest.approx(1.6)
    assert row2["delta_complex_plddt_vs_apo"] == pytest.approx(2.0)   # 82 - 80
    assert row2["delta_protein_iptm_vs_apo"] == pytest.approx(0.01)   # 0.76 - 0.75
    assert row2["delta_rmsd_vs_apo"] == pytest.approx(0.0)            # 1.6 - 1.6


def test_ligand_counter_screen_ranking_priority():
    """Verify that ligand counter screen rankings prioritize apo divergence over copper-independent folding."""
    from boltzgen.task.filter.filter import Filter
    from boltzgen.task.filter.material_presets import ligand_counter_screen_filtering_overrides

    overrides = ligand_counter_screen_filtering_overrides()

    filter_task = Filter(
        design_dir=".",
        budget=10,
        metrics_override=overrides,
        refolding_rmsd_threshold=5.0,
    )

    # candidate_divergent: folds well with metal (rmsd=1.5), completely falls apart without (apo_rmsd=12.0)
    # candidate_independent: folds equally well without metal (rmsd=1.2, apo_rmsd=1.0, delta_rmsd=-0.2)
    df = pd.DataFrame([
        {
            "id": "candidate_divergent",
            "delta_rmsd_vs_apo": 10.5,
            "holo_apo_rmsd": 11.5,
            "apo_rmsd": 12.0,
            "delta_protein_iptm_vs_apo": 0.60,
            "delta_complex_plddt_vs_apo": 40.0,
            "neg_lattice_rmsd_refolded": -1.5,
            "lattice_rmsd_refolded": 1.5,
            "local_rmsd": 1.5,
            "global_rmsd": 1.5,
            "complex_plddt": 85.0,
            "num_filters_passed": 2,
            "pass_local_rmsd_filter": True,
            "pass_global_rmsd_filter": True,
        },
        {
            "id": "candidate_independent",
            "delta_rmsd_vs_apo": -0.2,
            "holo_apo_rmsd": 0.5,
            "apo_rmsd": 1.0,
            "delta_protein_iptm_vs_apo": 0.01,
            "delta_complex_plddt_vs_apo": 1.0,
            "neg_lattice_rmsd_refolded": -1.2,
            "lattice_rmsd_refolded": 1.2,
            "local_rmsd": 1.2,
            "global_rmsd": 1.2,
            "complex_plddt": 88.0,
            "num_filters_passed": 2,
            "pass_local_rmsd_filter": True,
            "pass_global_rmsd_filter": True,
        },
    ])

    filter_task.df = df
    filter_task.sort_df()

    sorted_ids = filter_task.df.sort_values("max_rank")["id"].tolist()
    assert sorted_ids[0] == "candidate_divergent"
    assert sorted_ids[1] == "candidate_independent"
