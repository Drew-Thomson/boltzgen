"""Tests for materials-focused structural RMSD filtering and ranking."""

# ruff: noqa: INP001, PLR2004, PD901

from pathlib import Path

import pandas as pd

from boltzgen.task.filter.filter import Filter
from boltzgen.task.filter.material_presets import (
    format_additional_filters,
    material_filtering_overrides,
)


def test_filter_defaults_to_structural_similarity_only() -> None:
    task = Filter.__new__(Filter)
    task.design_dir = Path("dummy")
    task.top_budget = 10
    task.budget = 30
    task.use_affinity = False
    task.from_inverse_folded = True
    task.filter_bindingsite = False
    task.alpha = 0.01
    task.random_state = 0
    task.num_liability_plots = 0
    task.plot_seq_logos = False
    task.modality = "peptide"
    task.peptide_type = "linear"
    task.size_buckets = []
    task.refolding_rmsd_threshold = 2.5
    task.global_rmsd_threshold = 2.5
    task.outdir = Path("dummy_out")
    task.top_dir = task.outdir / "top"
    task.div_dir = task.outdir / "div"

    task.metrics = {
        "complex_plddt": 1,
        "design_ptm": 1,
        "neg_min_interaction_pae": 1,
        "neg_local_rmsd": 1,
        "neg_global_rmsd": 2,
    }

    task.filters = [
        {
            "feature": "local_rmsd",
            "lower_is_better": True,
            "threshold": task.refolding_rmsd_threshold,
        },
        {
            "feature": "global_rmsd",
            "lower_is_better": True,
            "threshold": task.global_rmsd_threshold,
        },
    ]

    features = [f["feature"] for f in task.filters]
    assert "has_x" not in features
    assert "CYS_fraction" not in features
    assert "ALA_fraction" not in features
    assert "GLY_fraction" not in features
    assert "local_rmsd" in features
    assert "global_rmsd" in features


def test_load_dataframe_two_chain_material_lattice_rmsd() -> None:
    task = Filter.__new__(Filter)
    task.from_inverse_folded = True
    task.refolding_rmsd_threshold = 2.5
    task.global_rmsd_threshold = 2.5
    task.filters = [
        {"feature": "local_rmsd", "lower_is_better": True, "threshold": 2.5},
        {"feature": "global_rmsd", "lower_is_better": True, "threshold": 2.5},
    ]

    # Mock multi-chain lattice dataframe with lattice_rmsd_refolded
    df_in = pd.DataFrame(
        {
            "id": ["d1", "d2"],
            "designed_sequence": ["ACDEF", "FGHIJ"],
            "bb_rmsd": [8.5, 9.2],
            "lattice_rmsd_refolded": [1.2, 3.4],
            "min_interaction_pae": [4.0, 7.5],
        }
    )

    task.design_dir = Path("dummy")
    # Simulate load_dataframe logic
    df = df_in.copy()
    df["global_rmsd"] = df["bb_rmsd"] if task.from_inverse_folded else df["rmsd"]
    if "lattice_rmsd_refolded" in df and df["lattice_rmsd_refolded"].notna().any():
        df["local_rmsd"] = df["lattice_rmsd_refolded"]
        task.global_rmsd_threshold = 15.0
        for filter_dict in task.filters:
            if filter_dict.get("feature") == "global_rmsd":
                filter_dict["threshold"] = task.global_rmsd_threshold

    df["neg_local_rmsd"] = -df["local_rmsd"]
    df["neg_global_rmsd"] = -df["global_rmsd"]
    df["neg_min_interaction_pae"] = -df["min_interaction_pae"]
    task.df = df

    assert task.df["local_rmsd"].tolist() == [1.2, 3.4]
    assert task.df["global_rmsd"].tolist() == [8.5, 9.2]
    assert task.global_rmsd_threshold == 15.0
    assert task.filters[1]["threshold"] == 15.0
    assert task.df["neg_local_rmsd"].tolist() == [-1.2, -3.4]


def test_load_dataframe_single_chain_fallback() -> None:
    task = Filter.__new__(Filter)
    task.from_inverse_folded = True
    task.refolding_rmsd_threshold = 2.5
    task.global_rmsd_threshold = 2.5
    task.filters = [
        {"feature": "local_rmsd", "lower_is_better": True, "threshold": 2.5},
        {"feature": "global_rmsd", "lower_is_better": True, "threshold": 2.5},
    ]

    df_in = pd.DataFrame(
        {
            "id": ["d1"],
            "designed_sequence": ["ACDEF"],
            "bb_rmsd": [1.4],
        }
    )

    df = df_in.copy()
    df["global_rmsd"] = df["bb_rmsd"]
    df["local_rmsd"] = df["global_rmsd"]
    df["neg_local_rmsd"] = -df["local_rmsd"]
    df["neg_global_rmsd"] = -df["global_rmsd"]
    task.df = df

    assert task.df["local_rmsd"].iloc[0] == 1.4
    assert task.df["global_rmsd"].iloc[0] == 1.4
    assert task.global_rmsd_threshold == 2.5


def test_format_additional_filters_inactivated_by_default() -> None:
    # Inactive by default
    assert format_additional_filters("double_tape") is None
    assert format_additional_filters("linear_tape") is None
    assert format_additional_filters("bilayer_sheet") is None

    # Opt-in works when explicitly requested
    active = format_additional_filters(
        "double_tape", enable_consecutive_hydrophobics=True
    )
    assert active is not None
    assert "max_consecutive_ilv_sheet" in active


def test_material_filtering_overrides_omits_hbonds_and_composition() -> None:
    tape = material_filtering_overrides("double_tape")
    assert "h_bonds_per_interface_refolded" not in tape
    assert "neg_max_consecutive_ilv_sheet" not in tape
    assert "neg_design_hydrophobicity" not in tape
    assert "neg_outward_ilv_fraction" not in tape
    assert tape["complex_plddt"] == 1
    assert tape["neg_lattice_rmsd_refolded"] == 1
    assert tape.get("neg_global_rmsd") is None

    sheet = material_filtering_overrides("bilayer_sheet")
    assert "h_bonds_per_interface_refolded" not in sheet
    assert sheet["sheet_planarity"] == 1
    assert sheet.get("neg_global_rmsd") is None


def test_format_additional_filters_heteromer() -> None:
    active = format_additional_filters("double_tape", is_heteromer=True)
    assert active is not None
    assert "two_chain_alternation_ratio" in active
    assert "0.8" in active
    assert "strand_polarity_target_match" in active
    assert "intra_sheet_alternation_ratio" in active
    assert "bilayer_dist_std" in active

    combined = format_additional_filters(
        "double_tape", enable_consecutive_hydrophobics=True, is_heteromer=True
    )
    assert "max_consecutive_ilv_sheet" in combined
    assert "two_chain_alternation_ratio" in combined
    assert "strand_polarity_target_match" in combined


def test_load_dataframe_double_tape_local_rmsd_priority(tmp_path: Path) -> None:
    task = Filter.__new__(Filter)
    task.from_inverse_folded = True
    task.refolding_rmsd_threshold = 2.5
    task.global_rmsd_threshold = 2.5
    task.filters = [
        {"feature": "local_rmsd", "lower_is_better": True, "threshold": 2.5},
        {"feature": "global_rmsd", "lower_is_better": True, "threshold": 2.5},
    ]

    # Both local_double_tape_rmsd and legacy lattice_rmsd_refolded present
    df_in = pd.DataFrame(
        {
            "id": ["d1", "d2"],
            "designed_sequence": ["ACDEF", "FGHIJ"],
            "bb_rmsd": [8.5, 9.2],
            "lattice_rmsd_refolded": [6.5, 7.2],
            "local_double_tape_rmsd": [1.3, 1.8],
            "min_interaction_pae": [4.0, 7.5],
        }
    )

    df_in.to_csv(tmp_path / "aggregate_metrics_0.csv", index=False)
    task.design_dir = tmp_path
    task.load_dataframe()
    # Verifies local_double_tape_rmsd was chosen as local_rmsd
    assert task.df["local_rmsd"].tolist() == [1.3, 1.8]
    assert task.df["global_rmsd"].tolist() == [8.5, 9.2]
    assert task.global_rmsd_threshold == 15.0


def test_format_additional_filters_composition() -> None:
    active = format_additional_filters("double_tape", enable_composition_filters=True)
    assert active is not None
    assert "outward_aromatic_fraction" in active
    assert "design_hydrophobicity" in active
    assert "THR_fraction" in active

