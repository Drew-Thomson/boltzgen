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
