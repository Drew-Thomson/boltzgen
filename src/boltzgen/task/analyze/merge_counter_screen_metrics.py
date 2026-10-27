"""Join homomer counter-screen confidence scores into heteromer analysis CSVs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from boltzgen.data import const
from boltzgen.task.task import Task


_METRICS = ("complex_plddt", "design_iptm", "protein_iptm", "min_interaction_pae")


def _extract_protein_backbone_coords(
    cif_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Extract protein backbone coordinates, chain IDs, and mask from MMCIF."""
    if not cif_path.exists():
        return None
    try:
        import gemmi

        st = gemmi.read_structure(str(cif_path))
        coords: list[list[float]] = []
        chain_ids: list[int] = []
        bb_mask: list[bool] = []
        chain_counter = 0
        for model in st:
            for chain in model:
                has_ca = any(atom.name == "CA" for res in chain for atom in res)
                if not has_ca:
                    continue
                for res in chain:
                    for atom in res:
                        if atom.name in {"N", "CA", "C", "O"}:
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                            chain_ids.append(chain_counter)
                            bb_mask.append(True)
                chain_counter += 1
            break
        if not coords:
            return None
        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(chain_ids, dtype=torch.long),
            torch.tensor(bb_mask, dtype=torch.bool),
        )
    except Exception as exc:
        print(f"Warning: failed to read protein backbone from {cif_path}: {exc}")
        return None


def _compute_cif_backbone_rmsd(
    refold_cif_path: Path, design_cif_path: Path, topology: str = "cyclic"
) -> float:
    """Compute chain-order independent backbone RMSD between refold and design CIFs.

    Parameters
    ----------
    refold_cif_path : Path
        Path to refolded MMCIF structure.
    design_cif_path : Path
        Path to reference design MMCIF structure.
    topology : str, optional
        Assembly topology constraint (default: 'cyclic').

    Returns
    -------
    float
        Chain-order independent backbone RMSD in Angstroms, or NaN if unavailable.
    """
    bb_r = _extract_protein_backbone_coords(refold_cif_path)
    bb_i = _extract_protein_backbone_coords(design_cif_path)
    if bb_r is None or bb_i is None:
        return float("nan")
    try:
        from boltzgen.task.analyze.material_metrics import compute_chain_order_independent_rmsd

        rmsd, _ = compute_chain_order_independent_rmsd(
            coords_refold=bb_r[0],
            coords_ideal=bb_i[0],
            chain_ids_refold=bb_r[1],
            chain_ids_ideal=bb_i[1],
            bb_mask_refold=bb_r[2],
            bb_mask_ideal=bb_i[2],
            topology=topology,
        )
        return float(rmsd)
    except Exception as exc:
        print(f"Warning: failed to compute CIF backbone RMSD: {exc}")
        return float("nan")


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

                if "apo" in partner_ids:
                    screen_id = partner_ids["apo"]
                    apo_pred_path = metrics_path / f"{screen_id}.npz"
                    if not apo_pred_path.exists():
                        apo_pred_path = self.counter_screen_dir / f"{screen_id}.npz"
                    apo_prediction = (
                        _load_prediction_metrics(apo_pred_path)
                        if apo_pred_path.exists()
                        else {}
                    )

                    refold_cif = (
                        self.counter_screen_dir / const.refold_cif_dirname / f"{screen_id}.cif"
                    )
                    if (self.counter_screen_dir / f"{screen_id}.cif").exists():
                        refold_cif = self.counter_screen_dir / f"{screen_id}.cif"

                    base_parent = parent_id.rsplit("_", 1)[0] if "_" in parent_id else parent_id
                    design_cif = self.design_dir / f"{parent_id}.cif"
                    for cand in [
                        self.design_dir / f"{parent_id}.cif",
                        self.design_dir / f"{base_parent}.cif",
                        self.design_dir.parent / "intermediate_designs" / f"{parent_id}.cif",
                        self.design_dir.parent / "intermediate_designs" / f"{base_parent}.cif",
                    ]:
                        if cand.exists():
                            design_cif = cand
                            break

                    holo_refold_cif = (
                        self.design_dir / const.refold_cif_dirname / f"{parent_id}.cif"
                    )
                    for cand in [
                        self.design_dir / const.refold_cif_dirname / f"{parent_id}.cif",
                        self.design_dir / f"{parent_id}.cif",
                        self.design_dir / const.refold_cif_dirname / f"{base_parent}.cif",
                    ]:
                        if cand.exists():
                            holo_refold_cif = cand
                            break

                    apo_rmsd = _compute_cif_backbone_rmsd(refold_cif, design_cif)
                    holo_apo_rmsd = (
                        _compute_cif_backbone_rmsd(holo_refold_cif, refold_cif)
                        if holo_refold_cif.exists()
                        else float("nan")
                    )
                    holo_rmsd_cif = (
                        _compute_cif_backbone_rmsd(holo_refold_cif, design_cif)
                        if holo_refold_cif.exists() and design_cif.exists()
                        else float("nan")
                    )

                    for matching_parent in matching_rows:
                        row_idx = rows[matching_parent]
                        for key, value in apo_prediction.items():
                            frame.loc[row_idx, f"apo_{key}"] = value

                        if not math.isnan(holo_apo_rmsd):
                            frame.loc[row_idx, "holo_apo_rmsd"] = holo_apo_rmsd

                        if not math.isnan(apo_rmsd):
                            frame.loc[row_idx, "apo_rmsd"] = apo_rmsd
                            holo_rmsd = (
                                holo_rmsd_cif
                                if not math.isnan(holo_rmsd_cif)
                                else -float(frame.loc[row_idx, "neg_lattice_rmsd_refolded"])
                                if "neg_lattice_rmsd_refolded" in frame
                                and pd.notna(frame.loc[row_idx, "neg_lattice_rmsd_refolded"])
                                else float(frame.loc[row_idx, "lattice_rmsd_refolded"])
                                if "lattice_rmsd_refolded" in frame
                                and pd.notna(frame.loc[row_idx, "lattice_rmsd_refolded"])
                                else float(frame.loc[row_idx, "rmsd"])
                                if "rmsd" in frame and pd.notna(frame.loc[row_idx, "rmsd"])
                                else float("nan")
                            )
                            if not math.isnan(holo_rmsd):
                                frame.loc[row_idx, "delta_rmsd_vs_apo"] = apo_rmsd - holo_rmsd

                        for metric_key in ("complex_plddt", "protein_iptm", "design_iptm"):
                            if (
                                metric_key in apo_prediction
                                and metric_key in frame
                                and pd.notna(frame.loc[row_idx, metric_key])
                            ):
                                frame.loc[row_idx, f"delta_{metric_key}_vs_apo"] = (
                                    float(frame.loc[row_idx, metric_key]) - apo_prediction[metric_key]
                                )

                        if (
                            "min_interaction_pae" in apo_prediction
                            and "min_interaction_pae" in frame
                            and pd.notna(frame.loc[row_idx, "min_interaction_pae"])
                        ):
                            frame.loc[row_idx, "delta_neg_min_interaction_pae_vs_apo"] = (
                                apo_prediction["min_interaction_pae"] - float(frame.loc[row_idx, "min_interaction_pae"])
                            )
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
                        if "apo" in partner_ids:
                            screen_id = partner_ids["apo"]
                            counter_row = counter_rows.get(str(screen_id))
                            if counter_row is not None:
                                for candidate_id in matching_rows:
                                    row_idx = rows[candidate_id]
                                    for metric in ("complex_plddt", "design_iptm", "protein_iptm", "min_interaction_pae"):
                                        if metric in counter_row and pd.notna(counter_row[metric]):
                                            frame.loc[row_idx, f"apo_{metric}"] = float(counter_row[metric])
                                            if metric in frame and pd.notna(frame.loc[row_idx, metric]):
                                                frame.loc[row_idx, f"delta_{metric}_vs_apo"] = (
                                                    float(frame.loc[row_idx, metric]) - float(counter_row[metric])
                                                )
                            continue
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
