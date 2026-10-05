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
        "No structural RMSD column (local_double_tape_rmsd, "
        "neg_lattice_rmsd_refolded or bb_rmsd) found in aggregate metrics"
    )


class FilterTargetCandidates(Task):
    """Select target heteromers that pass structural quality thresholds.

    Reads ``aggregate_metrics_*.csv`` from ``design_dir`` (produced by the
    analysis step on the folded targets) and writes the ids that pass to
    ``counter_screen_candidates.json``. Only those candidates are expanded into
    homomer counter-screens by ``prepare_counter_screens``.
    """

    def __init__(
        self,
        design_dir: str,
        topology: Optional[str] = None,
        max_rmsd: float = 5.0,
        min_complex_plddt: Optional[float] = None,
    ) -> None:
        self.design_dir = Path(design_dir)
        self.topology = topology
        self.max_rmsd = max_rmsd
        self.min_complex_plddt = min_complex_plddt

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

        rmsd = target_rmsd(frame, self.topology)
        passed = rmsd <= self.max_rmsd
        if self.min_complex_plddt is not None:
            col = next(
                (c for c in ("heteromer_complex_plddt", "complex_plddt") if c in frame),
                None,
            )
            if col is None:
                raise ValueError(
                    "min_complex_plddt requested but no complex_plddt column found"
                )
            passed &= frame[col] >= self.min_complex_plddt
        passed = passed.fillna(False)

        ids = frame.loc[passed, "id"].astype(str).tolist()
        logger.info(
            f"{len(ids)}/{len(frame)} target designs passed "
            f"(rmsd <= {self.max_rmsd}, min_complex_plddt={self.min_complex_plddt})"
        )
        print(
            f"Target candidate culling: {len(ids)}/{len(frame)} designs pass "
            f"(rmsd <= {self.max_rmsd}, min_complex_plddt={self.min_complex_plddt})"
        )
        if not ids:
            raise RuntimeError(
                "No target designs passed the structural thresholds; refusing to "
                "generate counter-screens. Loosen --target_max_rmsd / "
                "--target_min_plddt or inspect the aggregate metrics."
            )
        (self.design_dir / CANDIDATES_FILENAME).write_text(
            json.dumps(
                {
                    "max_rmsd": self.max_rmsd,
                    "min_complex_plddt": self.min_complex_plddt,
                    "candidate_ids": ids,
                },
                indent=2,
            )
            + "\n"
        )
