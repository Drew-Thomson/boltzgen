"""Unit tests for piezo tri-state counter screen merging."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from boltzgen.data import const
from boltzgen.task.analyze.merge_counter_screen_metrics import MergeCounterScreenMetrics


def test_merge_piezo_counter_screen_metrics(tmp_path: Path):
    """Verify merging of tri-state tension and compression metrics into aggregate CSV."""
    design_dir = tmp_path / "design_dir"
    counter_dir = tmp_path / "counter_dir"
    folding_dir = counter_dir / const.folding_dirname

    design_dir.mkdir(parents=True)
    folding_dir.mkdir(parents=True)

    # 1. Create aggregate metrics CSV in design_dir
    df = pd.DataFrame({
        "id": ["sample_001", "sample_002"],
        "sequence": ["KKKKAAAAAADDDDD", "AAAAAAAAAAAAAAA"],
        "dipole_moment_z_debye": [1000.0, 500.0],
        "assembly_volume_angstrom3": [10000.0, 10000.0],
        "design_iptm": [0.85, 0.78],
        "complex_plddt": [88.0, 82.0],
    })
    df.to_csv(design_dir / "aggregate_metrics_0.csv", index=False)

    # 2. Create piezo mapping JSON
    mapping = {
        "sample_001": {
            "tension": "sample_001_tension",
            "compression": "sample_001_compression",
            "strain_tension": 0.10,
            "strain_compression": 0.10,
        },
        "sample_002": {
            "tension": "sample_002_tension",
            "compression": "sample_002_compression",
            "strain_tension": 0.10,
            "strain_compression": 0.10,
        },
    }
    (counter_dir / "piezo_counter_screen_mapping.json").write_text(json.dumps(mapping, indent=2))

    # 3. Create mock prediction .npz files
    # Sample 1: under tension dipole rises by 100 D; under compression dipole drops by 100 D
    np.savez_compressed(
        folding_dir / "sample_001_tension.npz",
        dipole_moment_z_debye=np.array([1100.0]),
        design_iptm=np.array([0.80]),
        complex_plddt=np.array([86.0]),
    )
    np.savez_compressed(
        folding_dir / "sample_001_compression.npz",
        dipole_moment_z_debye=np.array([900.0]),
        design_iptm=np.array([0.82]),
        complex_plddt=np.array([87.0]),
    )

    # Sample 2:
    np.savez_compressed(
        folding_dir / "sample_002_tension.npz",
        dipole_moment_z_debye=np.array([550.0]),
        design_iptm=np.array([0.70]),
        complex_plddt=np.array([80.0]),
    )
    np.savez_compressed(
        folding_dir / "sample_002_compression.npz",
        dipole_moment_z_debye=np.array([450.0]),
        design_iptm=np.array([0.72]),
        complex_plddt=np.array([81.0]),
    )

    # Run merger
    merger = MergeCounterScreenMetrics(design_dir=str(design_dir), counter_screen_dir=str(counter_dir))
    merger.run()

    # Verify merged CSV
    merged_df = pd.read_csv(design_dir / "aggregate_metrics_0.csv")
    assert "piezo_e33_mean" in merged_df.columns
    assert "piezo_asymmetry" in merged_df.columns
    assert "min_strained_iptm" in merged_df.columns
    assert "delta_iptm_vs_tension" in merged_df.columns

    # Check sample 1 values: e33_ten = (100 / 10000) / 0.10 = 0.10; e33_comp = (-100 / 10000) / -0.10 = 0.10
    row_001 = merged_df[merged_df["id"] == "sample_001"].iloc[0]
    assert pytest.approx(0.10, rel=1e-3) == row_001["piezo_e33_mean"]
    assert pytest.approx(0.0, abs=1e-4) == row_001["piezo_asymmetry"]
    assert pytest.approx(0.80, abs=1e-3) == row_001["min_strained_iptm"]
    assert pytest.approx(0.05, abs=1e-3) == row_001["delta_iptm_vs_tension"]

    # Verify sequence bias report
    report_file = design_dir / "sequence_bias_report.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text())
    assert "top_candidates_count" in report
    assert report["top_candidates_count"] >= 1
