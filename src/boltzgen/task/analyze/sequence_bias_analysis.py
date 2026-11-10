"""Sequence and charge bias characterization for top-ranked piezoelectric designs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FORMAL_CHARGES: dict[str, float] = {
    "R": 1.0,
    "K": 1.0,
    "D": -1.0,
    "E": -1.0,
    "H": 0.1,
}

CANONICAL_AAS: list[str] = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]


def analyze_top_hits_sequence_bias(
    df_metrics: pd.DataFrame,
    score_column: str = "piezo_e33_mean",
    sequence_column: str = "sequence",
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    r"""Analyze charge asymmetry and amino acid enrichment in top piezoelectric candidates.

    Calculates:
    .. math::
        q_{\text{N}} = \sum_{i=1}^{\lfloor L/2 \rfloor} q(s_i), \quad
        q_{\text{C}} = \sum_{i=\lfloor L/2 \rfloor + 1}^L q(s_i), \quad
        \Delta q_{\text{term}} = q_{\text{N}} - q_{\text{C}}

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame containing candidate metrics and designed sequences.
    score_column : str
        Column name used for sorting and ranking candidates.
    sequence_column : str
        Column name storing candidate amino acid sequences.
    top_fraction : float
        Fraction of highest-scoring candidates to designate as top hits.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - ``total_candidates``: Total count of evaluated designs.
        - ``top_candidates_count``: Count of designs in the top fraction.
        - ``top_mean_q_net``: Mean net charge of top candidates.
        - ``top_mean_q_n_term``: Mean charge of N-terminal half in top candidates.
        - ``top_mean_q_c_term``: Mean charge of C-terminal half in top candidates.
        - ``top_mean_charge_asymmetry``: Mean \Delta q in top candidates.
        - ``bg_mean_charge_asymmetry``: Mean \Delta q in background candidates.
        - ``spontaneous_asymmetry_detected``: Boolean flag if asymmetry divergence exceeds 0.5 e.
        - ``aa_enrichment_ratio``: Dict mapping amino acid 1-letter codes to top vs background enrichment ratio.
    """
    if df_metrics.empty or sequence_column not in df_metrics.columns:
        return {
            "total_candidates": 0,
            "top_candidates_count": 0,
            "top_mean_q_net": 0.0,
            "top_mean_q_n_term": 0.0,
            "top_mean_q_c_term": 0.0,
            "top_mean_charge_asymmetry": 0.0,
            "bg_mean_charge_asymmetry": 0.0,
            "spontaneous_asymmetry_detected": False,
            "aa_enrichment_ratio": {},
        }

    total_candidates = len(df_metrics)
    if score_column in df_metrics.columns:
        df_sorted = df_metrics.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    else:
        df_sorted = df_metrics.copy().reset_index(drop=True)

    n_top = max(1, int(np.ceil(total_candidates * top_fraction)))
    top_df = df_sorted.iloc[:n_top]
    bg_df = df_sorted.iloc[n_top:]

    def _extract_charges(seqs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_net_list, q_n_list, q_c_list = [], [], []
        for seq in seqs:
            if not isinstance(seq, str) or len(seq) == 0:
                continue
            half = len(seq) // 2
            qn = float(sum(FORMAL_CHARGES.get(aa, 0.0) for aa in seq[:half]))
            qc = float(sum(FORMAL_CHARGES.get(aa, 0.0) for aa in seq[half:]))
            q_net_list.append(qn + qc)
            q_n_list.append(qn)
            q_c_list.append(qc)
        if not q_net_list:
            return np.zeros(1), np.zeros(1), np.zeros(1)
        return np.array(q_net_list), np.array(q_n_list), np.array(q_c_list)

    top_seqs = top_df[sequence_column].tolist()
    bg_seqs = bg_df[sequence_column].tolist()

    top_q_net, top_q_n, top_q_c = _extract_charges(top_seqs)
    bg_q_net, bg_q_n, bg_q_c = _extract_charges(bg_seqs) if bg_seqs else (np.zeros(1), np.zeros(1), np.zeros(1))

    top_mean_delta_q = float(np.mean(top_q_n - top_q_c))
    bg_mean_delta_q = float(np.mean(bg_q_n - bg_q_c)) if len(bg_seqs) > 0 else 0.0

    # Calculate per-amino-acid frequency enrichment
    def _aa_frequencies(seqs: list[str]) -> dict[str, float]:
        all_aas = "".join(seq for seq in seqs if isinstance(seq, str))
        total_len = max(len(all_aas), 1)
        return {aa: all_aas.count(aa) / total_len for aa in CANONICAL_AAS}

    top_freqs = _aa_frequencies(top_seqs)
    bg_freqs = _aa_frequencies(bg_seqs) if bg_seqs else top_freqs

    enrichment: dict[str, float] = {}
    for aa in CANONICAL_AAS:
        bg_val = bg_freqs.get(aa, 0.0)
        top_val = top_freqs.get(aa, 0.0)
        # Ratio with Laplace smoothing to avoid division by zero
        enrichment[aa] = float((top_val + 1e-4) / (bg_val + 1e-4))

    return {
        "total_candidates": total_candidates,
        "top_candidates_count": n_top,
        "top_mean_q_net": float(np.mean(top_q_net)),
        "top_mean_q_n_term": float(np.mean(top_q_n)),
        "top_mean_q_c_term": float(np.mean(top_q_c)),
        "top_mean_charge_asymmetry": top_mean_delta_q,
        "bg_mean_charge_asymmetry": bg_mean_delta_q,
        "spontaneous_asymmetry_detected": bool(abs(top_mean_delta_q - bg_mean_delta_q) > 0.5),
        "aa_enrichment_ratio": enrichment,
    }
