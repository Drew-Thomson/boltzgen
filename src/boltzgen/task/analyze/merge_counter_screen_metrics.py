"""Join homomer counter-screen confidence scores into heteromer analysis CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from boltzgen.data import const
from boltzgen.task.task import Task


_METRICS = ("complex_plddt", "design_iptm", "protein_iptm", "min_interaction_pae")


def _load_prediction_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Counter-screen prediction is missing: {path}")
    with np.load(path, allow_pickle=True) as output:
        values = {}
        for key in _METRICS:
            if key not in output.files:
                continue
            value = np.asarray(output[key], dtype=float)
            if value.size:
                values[key] = float(np.nanmean(value))
        return values


class MergeCounterScreenMetrics(Task):
    """Add raw and contrast counter-screen metrics to the Analyze CSV."""

    def __init__(self, design_dir: str, counter_screen_dir: str) -> None:
        self.design_dir = Path(design_dir)
        self.counter_screen_dir = Path(counter_screen_dir)

    def run(self, config=None) -> None:  # noqa: ANN001, ARG002
        mapping_path = self.counter_screen_dir / "counter_screen_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Counter-screen mapping not found: {mapping_path}")
        mapping = json.loads(mapping_path.read_text())
        metrics_path = self.counter_screen_dir / const.folding_dirname
        csv_paths = sorted(self.design_dir.glob("aggregate_metrics_*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No aggregate_metrics_*.csv found in {self.design_dir}")

        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            if "id" not in frame:
                raise ValueError(f"Aggregate metrics CSV lacks an id column: {csv_path}")
            rows = {str(row_id): idx for idx, row_id in enumerate(frame["id"].astype(str))}
            for parent_id, partner_ids in mapping.items():
                matching_rows = [
                    candidate_id
                    for candidate_id in rows
                    if candidate_id == parent_id
                    or candidate_id.startswith(f"{parent_id}_")
                ]
                if not matching_rows:
                    continue
                heteromer_metrics_path = (
                    self.design_dir / const.folding_dirname / f"{parent_id}.npz"
                )
                if not heteromer_metrics_path.exists():
                    raise FileNotFoundError(
                        f"Heteromer folding prediction is missing for {parent_id}: "
                        f"{heteromer_metrics_path}"
                    )
                heteromer_prediction = _load_prediction_metrics(heteromer_metrics_path)
                partner_values = {}
                for partner in ("A", "B"):
                    screen_id = partner_ids.get(partner)
                    if not screen_id:
                        raise ValueError(f"Missing homomer {partner} mapping for {parent_id}")
                    partner_values[partner] = _load_prediction_metrics(
                        metrics_path / f"{screen_id}.npz"
                    )
                for matching_parent in matching_rows:
                    row_idx = rows[matching_parent]
                    heteromer_values = dict(heteromer_prediction)
                    heteromer_values.update({
                        key.removeprefix("heteromer_"): float(value)
                        for key, value in frame.loc[row_idx].items()
                        if str(key).startswith("heteromer_") and pd.notna(value)
                    })
                    for key, value in heteromer_values.items():
                        frame.loc[row_idx, f"heteromer_{key}"] = value
                    for partner in ("A", "B"):
                        for key, value in partner_values[partner].items():
                            frame.loc[row_idx, f"homomer_{partner.lower()}_{key}"] = value
                    for key in ("complex_plddt", "design_iptm", "protein_iptm"):
                        if all(key in vals for vals in (heteromer_values, partner_values["A"], partner_values["B"])):
                            frame.loc[row_idx, f"delta_{key}_vs_homomer_max"] = (
                                heteromer_values[key]
                                - max(partner_values["A"][key], partner_values["B"][key])
                            )
                    if all(
                        "min_interaction_pae" in vals
                        for vals in (heteromer_values, partner_values["A"], partner_values["B"])
                    ):
                        frame.loc[row_idx, "delta_neg_min_interaction_pae_vs_homomer_max"] = (
                            min(
                                partner_values["A"]["min_interaction_pae"],
                                partner_values["B"]["min_interaction_pae"],
                            )
                            - heteromer_values["min_interaction_pae"]
                        )
            counter_csvs = sorted(self.counter_screen_dir.glob("aggregate_metrics_*.csv"))
            if counter_csvs:
                counter_frame = pd.concat(
                    [pd.read_csv(counter_csv) for counter_csv in counter_csvs],
                    ignore_index=True,
                )
                if "id" in counter_frame:
                    counter_rows = {
                        str(row["id"]): row
                        for _, row in counter_frame.iterrows()
                    }
                    for parent_id, partner_ids in mapping.items():
                        matching_rows = [
                            candidate_id
                            for candidate_id in rows
                            if candidate_id == parent_id
                            or candidate_id.startswith(f"{parent_id}_")
                        ]
                        for partner in ("A", "B"):
                            screen_id = partner_ids.get(partner)
                            counter_row = counter_rows.get(str(screen_id))
                            if counter_row is None:
                                continue
                            for candidate_id in matching_rows:
                                row_idx = rows[candidate_id]
                                heteromer_values = {
                                    metric.removeprefix("heteromer_"): float(frame.loc[row_idx, metric])
                                    for metric in frame.columns
                                    if str(metric).startswith("heteromer_")
                                    and pd.notna(frame.loc[row_idx, metric])
                                }
                                for metric in ("complex_plddt", "design_iptm", "protein_iptm", "min_interaction_pae"):
                                    if metric in counter_row and pd.notna(counter_row[metric]):
                                        frame.loc[row_idx, f"homomer_{partner.lower()}_{metric}"] = float(counter_row[metric])
                                for metric in ("complex_plddt", "design_iptm", "protein_iptm"):
                                    homomer_cols = [
                                        frame.loc[row_idx, f"homomer_{label.lower()}_{metric}"]
                                        for label in ("A", "B")
                                        if f"homomer_{label.lower()}_{metric}" in frame
                                        and pd.notna(frame.loc[row_idx, f"homomer_{label.lower()}_{metric}"])
                                    ]
                                    if metric in heteromer_values and len(homomer_cols) == 2:
                                        frame.loc[row_idx, f"delta_{metric}_vs_homomer_max"] = (
                                            heteromer_values[metric] - max(homomer_cols)
                                        )
                                if (
                                    "min_interaction_pae" in heteromer_values
                                    and all(
                                        f"homomer_{label.lower()}_min_interaction_pae" in frame
                                        and pd.notna(frame.loc[row_idx, f"homomer_{label.lower()}_min_interaction_pae"])
                                        for label in ("A", "B")
                                    )
                                ):
                                    frame.loc[row_idx, "delta_neg_min_interaction_pae_vs_homomer_max"] = (
                                        min(
                                            frame.loc[row_idx, "homomer_a_min_interaction_pae"],
                                            frame.loc[row_idx, "homomer_b_min_interaction_pae"],
                                        ) - heteromer_values["min_interaction_pae"]
                                    )
            # Designs that were culled before counter-screening have no
            # contrast metrics and would break contrast-based ranking. Keep them
            # in a side file for traceability and drop them from the CSV that
            # the filtering step consumes.
            screened = np.array(
                [
                    any(
                        row_id == parent or row_id.startswith(f"{parent}_")
                        for parent in mapping
                    )
                    for row_id in frame["id"].astype(str)
                ],
                dtype=bool,
            )
            if not screened.all():
                frame.loc[~screened].to_csv(
                    csv_path.with_name(
                        csv_path.name.replace("aggregate_metrics_", "culled_target_metrics_")
                    ),
                    index=False,
                    float_format="%.5f",
                )
                frame = frame.loc[screened]
            frame.to_csv(csv_path, index=False, float_format="%.5f")
