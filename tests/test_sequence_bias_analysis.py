"""Unit tests for sequence bias and charge distribution analysis."""

import numpy as np
import pandas as pd
import pytest

from boltzgen.task.analyze.sequence_bias_analysis import analyze_top_hits_sequence_bias


def test_sequence_bias_detection():
    """Verify detection of terminal charge asymmetry in top-ranked designs."""
    np.random.seed(42)

    # 10 top hits with strong N-terminal positive charges and C-terminal negative charges
    # e.g. KKKKAAAAAADDDDD (4 K, 6 A, 5 D)
    top_seqs = ["KKKKAAAAAADDDDD"] * 10
    top_scores = [1.5 - 0.01 * i for i in range(10)]

    # 90 background designs with neutral / uncharged sequences (15 A)
    bg_seqs = ["AAAAAAAAAAAAAAA"] * 90
    bg_scores = [0.5 - 0.005 * i for i in range(90)]

    df = pd.DataFrame({
        "sequence": top_seqs + bg_seqs,
        "piezo_e33_mean": top_scores + bg_scores,
    })

    result = analyze_top_hits_sequence_bias(
        df_metrics=df,
        score_column="piezo_e33_mean",
        sequence_column="sequence",
        top_fraction=0.10,
    )

    assert result["top_candidates_count"] == 10
    # Top sequences: N-term has 4 K (+4), C-term has 5 D (-5), asymmetry = +4 - (-5) = +9
    assert result["top_mean_charge_asymmetry"] > 5.0
    assert result["spontaneous_asymmetry_detected"] is True
    # K should be heavily enriched in top candidates
    assert result["aa_enrichment_ratio"]["K"] > 1.0


def test_sequence_bias_empty_or_neutral():
    """Verify handling of neutral or uniform datasets."""
    df = pd.DataFrame({
        "sequence": ["AAAAAAAAAA"] * 20,
        "piezo_e33_mean": [0.1] * 20,
    })
    result = analyze_top_hits_sequence_bias(df, top_fraction=0.20)
    assert result["top_mean_charge_asymmetry"] == 0.0
    assert result["bg_mean_charge_asymmetry"] == 0.0
    assert result["spontaneous_asymmetry_detected"] is False
