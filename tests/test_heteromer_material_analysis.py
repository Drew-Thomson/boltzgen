"""Tests for heteromer/homomer contrast calculations and CSV aggregation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from boltzgen.task.analyze.analyze_utils import heteromer_counter_screen_contrasts
from boltzgen.task.analyze.merge_counter_screen_metrics import MergeCounterScreenMetrics


def test_contrast_metrics_use_maximum_homomer_score():
    values = heteromer_counter_screen_contrasts(
        {"complex_plddt": 80, "design_iptm": 0.8, "protein_iptm": 0.75, "min_interaction_pae": 4},
        {"complex_plddt": 60, "design_iptm": 0.3, "protein_iptm": 0.4, "min_interaction_pae": 10},
        {"complex_plddt": 65, "design_iptm": 0.35, "protein_iptm": 0.45, "min_interaction_pae": 8},
    )
    assert values["delta_complex_plddt_vs_homomer_max"] == pytest.approx(15)
    assert values["delta_design_iptm_vs_homomer_max"] == pytest.approx(0.45)
    assert values["delta_protein_iptm_vs_homomer_max"] == pytest.approx(0.3)
    assert values["delta_neg_min_interaction_pae_vs_homomer_max"] == pytest.approx(4)


def test_merge_task_reports_missing_prediction(tmp_path):
    counter_dir = tmp_path / "screens"
    counter_dir.mkdir()
    design_dir = tmp_path / "designs"
    design_dir.mkdir()
    (counter_dir / "counter_screen_mapping.json").write_text(
        json.dumps({"parent": {"A": "parent__homomer_a", "B": "parent__homomer_b"}})
    )
    pd.DataFrame([{"id": "parent"}]).to_csv(
        design_dir / "aggregate_metrics_analyze.csv", index=False
    )
    for partner in ("a", "b"):
        (counter_dir / "fold_out_npz").mkdir(exist_ok=True)
        np.savez_compressed(
            counter_dir / "fold_out_npz" / f"parent__homomer_{partner}.npz",
            complex_plddt=np.asarray([50.0]),
            design_iptm=np.asarray([0.2]),
            protein_iptm=np.asarray([0.2]),
            min_interaction_pae=np.asarray([20.0]),
        )
    task = MergeCounterScreenMetrics(str(design_dir), str(counter_dir))
    with pytest.raises(FileNotFoundError, match="Heteromer"):
        task.run()


def test_merge_task_adds_raw_and_contrast_metrics(tmp_path):
    design_dir = tmp_path / "design"
    screen_dir = tmp_path / "screens"
    (design_dir / "fold_out_npz").mkdir(parents=True)
    (screen_dir / "fold_out_npz").mkdir(parents=True)
    (screen_dir / "counter_screen_mapping.json").write_text(
        json.dumps({"parent": {"A": "parent__homomer_a", "B": "parent__homomer_b"}})
    )
    np.savez_compressed(
        design_dir / "fold_out_npz" / "parent.npz",
        complex_plddt=np.asarray([80.0]),
        design_iptm=np.asarray([0.8]),
        protein_iptm=np.asarray([0.75]),
        min_interaction_pae=np.asarray([4.0]),
    )
    for partner, plddt, iptm in (("a", 55.0, 0.3), ("b", 60.0, 0.35)):
        np.savez_compressed(
            screen_dir / "fold_out_npz" / f"parent__homomer_{partner}.npz",
            complex_plddt=np.asarray([plddt]),
            design_iptm=np.asarray([iptm]),
            protein_iptm=np.asarray([iptm]),
            min_interaction_pae=np.asarray([10.0]),
        )
    csv_path = design_dir / "aggregate_metrics_analyze.csv"
    pd.DataFrame([{"id": "parent", "heteromer_design_iptm": 0.8}]).to_csv(
        csv_path, index=False
    )

    MergeCounterScreenMetrics(str(design_dir), str(screen_dir)).run()
    output = pd.read_csv(csv_path).iloc[0]
    assert output["homomer_a_complex_plddt"] == pytest.approx(55.0)
    assert output["homomer_b_complex_plddt"] == pytest.approx(60.0)
    assert output["delta_complex_plddt_vs_homomer_max"] == pytest.approx(20.0)
    assert output["delta_design_iptm_vs_homomer_max"] == pytest.approx(0.45)


def test_load_prediction_metrics_handles_vector_outputs(tmp_path):
    from boltzgen.task.analyze.merge_counter_screen_metrics import _load_prediction_metrics

    path = tmp_path / "prediction.npz"
    np.savez_compressed(
        path,
        complex_plddt=np.asarray([70.0, 80.0]),
        design_iptm=np.asarray([0.6, 0.8]),
    )
    assert _load_prediction_metrics(path) == {
        "complex_plddt": 75.0,
        "design_iptm": 0.7,
    }


def test_best_folding_sample_uses_available_modern_confidence_keys():
    from boltzgen.task.analyze.analyze_utils import get_best_folding_sample

    folded = {
        "coords": np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]]),
        "design_iptm": np.asarray([0.4, 0.8]),
        "ptm": np.asarray([0.9, 0.9]),
        "complex_plddt": np.asarray([60.0, 80.0]),
    }

    best = get_best_folding_sample(folded)
    assert np.array_equal(best["coords"], folded["coords"][1])
    assert best["design_iptm"] == pytest.approx(0.8)
    assert best["complex_plddt"] == pytest.approx(80.0)


def test_best_folding_sample_falls_back_to_coords_without_confidence():
    from boltzgen.task.analyze.analyze_utils import get_best_folding_sample

    folded = {"coords": np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]])}
    best = get_best_folding_sample(folded)
    assert np.array_equal(best["coords"], folded["coords"][0])
