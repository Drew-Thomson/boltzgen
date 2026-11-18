"""Cull folded target heteromers before expensive homomer counter-screens."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from boltzgen.task.task import Task

logger = logging.getLogger(__name__)

CANDIDATES_FILENAME = "counter_screen_candidates.json"


def candidate_matches(source_id: str, candidate_ids: Iterable[str]) -> bool:
    """True if ``source_id`` is (a parent of) one of the surviving candidate ids.

    Analysis row ids may carry a suffix after the inverse-folded CIF stem
    (``<stem>_<n>``), mirroring the matching used when merging metrics.
    """
    return any(
        cid == source_id or cid.startswith(f"{source_id}_") for cid in candidate_ids
    )


def load_candidate_ids(path: Path) -> list[str]:
    """Load surviving candidate ids written by :class:`FilterTargetCandidates`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Target candidate list not found: {path}. Run the "
            "filter_target_candidates step (after folding and analysis) first."
        )
    return [str(i) for i in json.loads(path.read_text())["candidate_ids"]]


def target_rmsd(frame: pd.DataFrame, topology: Optional[str]) -> pd.Series:
    """Per-design structural deviation (lower is better) for the target fold."""
    if (
        topology == "double_tape"
        and "local_step_rmsd" in frame
        and frame["local_step_rmsd"].notna().any()
    ):
        return frame["local_step_rmsd"]
    if (
        topology == "double_tape"
        and "lattice_rmsd_geom" in frame
        and frame["lattice_rmsd_geom"].notna().any()
    ):
        return frame["lattice_rmsd_geom"]
    if (
        topology == "double_tape"
        and "local_double_tape_rmsd" in frame
        and frame["local_double_tape_rmsd"].notna().any()
    ):
        return frame["local_double_tape_rmsd"]
    if "neg_lattice_rmsd_refolded" in frame and (
        frame["neg_lattice_rmsd_refolded"].fillna(0) != 0
    ).any():
        return -frame["neg_lattice_rmsd_refolded"]
    if "bb_rmsd" in frame:
        return frame["bb_rmsd"]
    raise ValueError(
        "No structural RMSD column (local_step_rmsd, lattice_rmsd_geom, "
        "local_double_tape_rmsd, neg_lattice_rmsd_refolded or bb_rmsd) "
        "found in aggregate metrics"
    )


def _get_metric_series(frame: pd.DataFrame, col: str) -> pd.Series | None:
    """Return numeric series for ``col``, deriving from ``neg_<col>`` if needed."""
    if col in frame and frame[col].notna().any():
        return pd.to_numeric(frame[col], errors="coerce")
    neg_col = f"neg_{col}"
    if neg_col in frame and frame[neg_col].notna().any():
        return -pd.to_numeric(frame[neg_col], errors="coerce")
    return None


def _ensure_neg_metric(df: pd.DataFrame, neg_col: str) -> pd.Series | None:
    """Return higher-is-better negated series for ``neg_<base>`` if available."""
    if neg_col in df and df[neg_col].notna().any():
        return pd.to_numeric(df[neg_col], errors="coerce")
    base_col = neg_col.removeprefix("neg_")
    if base_col in df and df[base_col].notna().any():
        return -pd.to_numeric(df[base_col], errors="coerce")
    return None


_INITIAL_RANK_WEIGHTS: dict[str, float] = {
    "complex_plddt": 1.0,
    "ligand_iptm": 2.0,
    "mean_ligand_plddt": 2.0,
    "ligand_min_coord_number": 1.0,
    "neg_ligand_coord_geom_dev": 1.0,
    "neg_ligand_coord_bond_dev": 2.0,
    "ss_accuracy_refolded": 1.0,
    "neg_sym_chain_rmsd": 1.0,
    "neg_sym_ligand_rmsd": 1.0,
}


class FilterTargetCandidates(Task):
    """Select target candidates that pass structural/quality thresholds.

    Reads ``aggregate_metrics_*.csv`` from ``design_dir`` (produced by the
    analysis step on the folded targets) and writes the ids that pass to
    ``counter_screen_candidates.json``. Only those candidates are expanded into
    counter-screens by ``prepare_counter_screens``.
    """

    def __init__(
        self,
        design_dir: str,
        topology: Optional[str] = None,
        max_rmsd: Optional[float] = None,
        use_template_rmsd: bool = True,
        min_complex_plddt: Optional[float] = None,
        min_ligand_distance: Optional[float] = None,
        min_ss_accuracy: Optional[float] = 0.70,
        max_sym_chain_rmsd: Optional[float] = 2.0,
        max_sym_ligand_rmsd: Optional[float] = 1.0,
        require_sym_orbit: bool = True,
        require_sym_orientation: bool = True,
        min_coord_number: Optional[float] = 3.0,
        max_coord_bond_dev: Optional[float] = 0.5,
        max_coord_geom_dev: Optional[float] = 25.0,
        rank_initial_candidates: bool = False,
        max_candidates: Optional[int] = None,
    ) -> None:
        self.design_dir = Path(design_dir)
        self.topology = topology
        self.max_rmsd = max_rmsd
        self.use_template_rmsd = use_template_rmsd
        self.min_complex_plddt = min_complex_plddt
        self.min_ligand_distance = min_ligand_distance
        self.min_ss_accuracy = min_ss_accuracy
        self.max_sym_chain_rmsd = max_sym_chain_rmsd
        self.max_sym_ligand_rmsd = max_sym_ligand_rmsd
        self.require_sym_orbit = require_sym_orbit
        self.require_sym_orientation = require_sym_orientation
        self.min_coord_number = min_coord_number
        self.max_coord_bond_dev = max_coord_bond_dev
        self.max_coord_geom_dev = max_coord_geom_dev
        self.rank_initial_candidates = rank_initial_candidates
        self.max_candidates = max_candidates

    def _effective_max_rmsd(self, frame: pd.DataFrame) -> float | None:
        """Resolve template-RMSD cutoff, or ``None`` when template RMSD is disabled."""
        if self.max_rmsd is not None:
            return float(self.max_rmsd)
        if not self.use_template_rmsd:
            return None
        from boltzgen.task.filter.material_presets import (
            CYCLIC_LIGAND_SYMMETRY_COLUMNS,
        )

        has_cyclic_ligand = self.topology == "cyclic" and all(
            c in frame.columns and frame[c].notna().any()
            for c in CYCLIC_LIGAND_SYMMETRY_COLUMNS
        )
        return None if has_cyclic_ligand else 5.0

    def _rank_candidates(self, frame: pd.DataFrame, passed: pd.Series) -> pd.DataFrame:
        """Rank all rows (passing first) by model quality, coordination, SS, and symmetry."""
        ranked = frame.copy()
        ranked["pass_initial_screen"] = passed.astype(bool)

        rank_cols: dict[str, pd.Series] = {}
        pass_int = ranked["pass_initial_screen"].astype(int)
        for col, weight in _INITIAL_RANK_WEIGHTS.items():
            series = (
                _ensure_neg_metric(ranked, col)
                if col.startswith("neg_")
                else (
                    pd.to_numeric(ranked[col], errors="coerce")
                    if col in ranked and ranked[col].notna().any()
                    else None
                )
            )
            if series is None:
                continue
            filled = series.fillna(float("-inf"))
            keys = list(zip(pass_int.tolist(), filled.tolist()))
            ranks = pd.Series(keys, index=ranked.index).rank(
                method="min", ascending=False
            )
            rank_cols[f"rank_{col}"] = ranks.astype(float) / weight

        if rank_cols:
            rank_df = pd.DataFrame(rank_cols, index=ranked.index)
            ranked["initial_max_rank"] = rank_df.max(axis=1)
        else:
            ranked["initial_max_rank"] = (~ranked["pass_initial_screen"]).astype(float)

        tie_col = next(
            (c for c in ("complex_plddt", "ligand_iptm") if c in ranked),
            None,
        )
        sort_cols = ["pass_initial_screen", "initial_max_rank"]
        ascending = [False, True]
        if tie_col is not None:
            sort_cols.append(tie_col)
            ascending.append(False)
        ranked = ranked.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
        ranked["initial_rank"] = range(1, len(ranked) + 1)
        return ranked

    def run(self, config=None) -> None:  # noqa: ANN001, ARG002
        csv_paths = sorted(self.design_dir.glob("aggregate_metrics_*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No aggregate_metrics_*.csv found in {self.design_dir}; "
                "run folding and analysis first."
            )
        frame = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
        if "id" not in frame:
            raise ValueError("Aggregate metrics CSV lacks an id column")

        passed = pd.Series(True, index=frame.index, dtype=bool)
        effective_max_rmsd = self._effective_max_rmsd(frame)
        if effective_max_rmsd is not None:
            rmsd = target_rmsd(frame, self.topology)
            passed &= rmsd <= effective_max_rmsd

        if self.min_complex_plddt is not None and float(self.min_complex_plddt) > 0.0:
            col = next(
                (c for c in ("heteromer_complex_plddt", "complex_plddt") if c in frame),
                None,
            )
            if col is None:
                raise ValueError(
                    "min_complex_plddt requested but no complex_plddt column found"
                )
            passed &= pd.to_numeric(frame[col], errors="coerce") >= self.min_complex_plddt

        if self.min_ligand_distance is not None and "min_ligand_distance" in frame:
            dist_s = pd.to_numeric(frame["min_ligand_distance"], errors="coerce")
            passed &= dist_s.isna() | (dist_s >= self.min_ligand_distance)

        if (
            self.min_ss_accuracy is not None
            and "ss_accuracy_refolded" in frame
            and frame["ss_accuracy_refolded"].notna().any()
        ):
            passed &= (
                pd.to_numeric(frame["ss_accuracy_refolded"], errors="coerce")
                >= self.min_ss_accuracy
            )

        for col, limit, lower_is_better in (
            ("sym_chain_rmsd", self.max_sym_chain_rmsd, True),
            ("sym_ligand_rmsd", self.max_sym_ligand_rmsd, True),
            ("ligand_min_coord_number", self.min_coord_number, False),
            ("ligand_coord_bond_dev", self.max_coord_bond_dev, True),
            ("ligand_coord_geom_dev", self.max_coord_geom_dev, True),
        ):
            if limit is None:
                continue
            series = _get_metric_series(frame, col)
            if series is not None:
                passed &= (series <= limit) if lower_is_better else (series >= limit)

        if (
            self.require_sym_orbit
            and "sym_ligand_orbit_ok" in frame
            and frame["sym_ligand_orbit_ok"].notna().any()
        ):
            passed &= pd.to_numeric(frame["sym_ligand_orbit_ok"], errors="coerce") >= 1.0

        if (
            self.require_sym_orientation
            and "sym_orientation_match" in frame
            and frame["sym_orientation_match"].notna().any()
        ):
            passed &= (
                pd.to_numeric(frame["sym_orientation_match"], errors="coerce") >= 1.0
            )

        passed = passed.fillna(False)

        if self.rank_initial_candidates or self.max_candidates is not None:
            ranked = self._rank_candidates(frame, passed)
            ranked.to_csv(
                self.design_dir / "initial_screen_ranked.csv",
                index=False,
                float_format="%.5f",
            )
            passing_ids = (
                ranked.loc[ranked["pass_initial_screen"], "id"].astype(str).tolist()
            )
        else:
            passing_ids = frame.loc[passed, "id"].astype(str).tolist()

        n_passed = len(passing_ids)
        if self.max_candidates is not None and self.max_candidates > 0:
            ids = passing_ids[: int(self.max_candidates)]
        else:
            ids = passing_ids

        logger.info(
            f"{len(ids)}/{len(frame)} target designs selected ({n_passed} passed gates; "
            f"max_rmsd={effective_max_rmsd}, min_ss_accuracy={self.min_ss_accuracy}, "
            f"max_candidates={self.max_candidates})"
        )
        print(
            f"Target candidate culling: {len(ids)}/{len(frame)} designs selected "
            f"({n_passed} passed gates; max_rmsd={effective_max_rmsd}, "
            f"min_ss_accuracy={self.min_ss_accuracy}, "
            f"min_coord_number={self.min_coord_number}, "
            f"max_candidates={self.max_candidates})"
        )
        if not ids:
            logger.warning(
                "No target designs passed the structural thresholds; skipping counter-screens."
            )
        (self.design_dir / CANDIDATES_FILENAME).write_text(
            json.dumps(
                {
                    "max_rmsd": effective_max_rmsd,
                    "use_template_rmsd": self.use_template_rmsd,
                    "min_complex_plddt": self.min_complex_plddt,
                    "min_ligand_distance": self.min_ligand_distance,
                    "min_ss_accuracy": self.min_ss_accuracy,
                    "min_coord_number": self.min_coord_number,
                    "max_candidates": self.max_candidates,
                    "num_passed_gates": n_passed,
                    "candidate_ids": ids,
                },
                indent=2,
            )
            + "\n"
        )
