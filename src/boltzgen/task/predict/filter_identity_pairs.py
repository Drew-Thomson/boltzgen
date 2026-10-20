import json
import logging
from pathlib import Path

import numpy as np
from Bio import Align

from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.task.task import Task

logger = logging.getLogger(__name__)


class FilterHighIdentityPairs(Task):
    """Remove inverse-folded candidates where partner A and B sequences are too similar."""

    def __init__(
        self,
        design_dir: str,
        layout_path: str,
        moldir: str = None,
        max_partner_identity: float = 0.5,
        max_surviving_designs: int = None,
        zoned: bool | None = None,
        weight_core: float = 0.0,
        weight_surface: float = 1.0,
        orientation: str = "auto",
    ):
        self.design_dir = Path(design_dir)
        self.layout_path = Path(layout_path)
        self.moldir = moldir
        self.max_partner_identity = float(max_partner_identity)
        if isinstance(max_surviving_designs, str):
            if max_surviving_designs.lower() == "none":
                max_surviving_designs = None
            else:
                max_surviving_designs = int(max_surviving_designs)
        self.max_surviving_designs = max_surviving_designs
        self.zoned = zoned
        self.weight_core = float(weight_core)
        self.weight_surface = float(weight_surface)
        self.orientation = str(orientation).lower()

    @staticmethod
    def _detect_strand_orientation(
        cif_path: Path, chain_a: str | None, chain_b: str | None
    ) -> bool | None:
        """Detect whether chain_a and chain_b have antiparallel strand directions.

        Returns True if antiparallel, False if parallel, or None if undetermined.
        """
        if not chain_a or not chain_b:
            return None
        try:
            import gemmi

            st = gemmi.read_structure(str(cif_path))
            ca_coords: dict[str, list[list[float]]] = {chain_a: [], chain_b: []}
            for model in st:
                for chain in model:
                    if chain.name in ca_coords and not ca_coords[chain.name]:
                        for res in chain:
                            for atom in res:
                                if atom.name == "CA":
                                    ca_coords[chain.name].append(
                                        [atom.pos.x, atom.pos.y, atom.pos.z]
                                    )
                if ca_coords[chain_a] and ca_coords[chain_b]:
                    break
            pts_a = np.array(ca_coords[chain_a], dtype=np.float32)
            pts_b = np.array(ca_coords[chain_b], dtype=np.float32)
            if len(pts_a) >= 2 and len(pts_b) >= 2:
                k_a = max(1, len(pts_a) // 4)
                k_b = max(1, len(pts_b) // 4)
                v_a = pts_a[-k_a:].mean(axis=0) - pts_a[:k_a].mean(axis=0)
                v_b = pts_b[-k_b:].mean(axis=0) - pts_b[:k_b].mean(axis=0)
                norm_a = np.linalg.norm(v_a)
                norm_b = np.linalg.norm(v_b)
                if norm_a > 1e-4 and norm_b > 1e-4:
                    dot = float(np.dot(v_a, v_b) / (norm_a * norm_b))
                    if abs(dot) > 1e-3:
                        return dot < 0.0
        except Exception:
            pass
        return None

    def run(self, config=None):
        if not self.design_dir.exists():
            logger.warning(f"Design dir {self.design_dir} does not exist. Skipping.")
            return

        try:
            layout = json.loads(self.layout_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load layout from {self.layout_path}: {e}")
            return

        topology = layout.get("topology")
        if not topology and "heteromer_screening" in layout:
            topology = layout["heteromer_screening"].get("topology")

        is_sheet_tape = topology in {"double_tape", "linear_tape", "bilayer_sheet"}
        use_zoned = self.zoned if self.zoned is not None else is_sheet_tape

        chain_to_partner = {
            r["chain_id"]: r.get("partner_label")
            for r in layout.get("chains", []) if r.get("role") == "protein"
        }

        all_candidates = []
        for cif_path in sorted(self.design_dir.glob("*.cif")):
            if cif_path.name.endswith("_native.cif"):
                continue

            sequences = {}
            try:
                parsed = parse_mmcif(cif_path, moldir=self.moldir, use_original_res_idx=False)
                sequences = parsed.sequences
            except Exception:
                try:
                    import gemmi
                    from boltzgen.data import const

                    st = gemmi.read_structure(str(cif_path))
                    for model in st:
                        for chain in model:
                            if chain.name not in sequences:
                                seq_chars = [
                                    const.prot_token_to_letter.get(res.name.upper(), "X")
                                    for res in chain
                                ]
                                if seq_chars:
                                    sequences[chain.name] = "".join(seq_chars)
                        if sequences:
                            break
                except Exception as e:
                    logger.warning(f"Failed to parse {cif_path}: {e}")
                    continue

            seq_a = seq_b = None
            chain_a_name = chain_b_name = None
            for chain_name, seq in sequences.items():
                partner = chain_to_partner.get(chain_name)
                if partner == "A" and seq_a is None:
                    seq_a = seq
                    chain_a_name = chain_name
                elif partner == "B" and seq_b is None:
                    seq_b = seq
                    chain_b_name = chain_name

            if seq_a is None or seq_b is None:
                continue

            if self.orientation == "antiparallel":
                is_antiparallel = True
            elif self.orientation == "parallel":
                is_antiparallel = False
            else:
                detected = self._detect_strand_orientation(
                    cif_path, chain_a_name, chain_b_name
                )
                if detected is not None:
                    is_antiparallel = detected
                else:
                    topo_params = layout.get("topology_params") or {}
                    if not topo_params and "heteromer_screening" in layout:
                        topo_params = layout["heteromer_screening"].get(
                            "topology_params", {}
                        )
                    anti_prob = float(topo_params.get("antiparallel_prob", 0.0))
                    is_antiparallel = anti_prob > 0.0 or bool(
                        layout.get("is_antiparallel", False)
                    )

            seq_b_aligned = seq_b[::-1] if is_antiparallel else seq_b

            if len(seq_a) == len(seq_b_aligned):
                if use_zoned:
                    # In beta-sheet tapes:
                    # 0-indexed even (1-indexed odd: 1, 3, 5...) is outward/solvent
                    # 0-indexed odd (1-indexed even: 2, 4, 6...) is inward/core
                    surf_idx = [i for i in range(len(seq_a)) if i % 2 == 0]
                    core_idx = [i for i in range(len(seq_a)) if i % 2 == 1]
                    matches_surf = sum(
                        seq_a[i] == seq_b_aligned[i] for i in surf_idx
                    )
                    matches_core = sum(
                        seq_a[i] == seq_b_aligned[i] for i in core_idx
                    )
                    id_surf = matches_surf / max(1, len(surf_idx))
                    id_core = matches_core / max(1, len(core_idx))
                    identity = (
                        self.weight_core * id_core
                        + self.weight_surface * id_surf
                    )
                else:
                    matches = sum(a == b for a, b in zip(seq_a, seq_b_aligned))
                    identity = matches / max(len(seq_a), len(seq_b_aligned))
            else:
                aligner = Align.PairwiseAligner()
                aligner.mode = "global"
                aligner.open_gap_score = -10.0
                aligner.extend_gap_score = -1.0
                aligner.target_end_gap_score = 0.0
                aligner.query_end_gap_score = 0.0
                aln = aligner.align(seq_a, seq_b_aligned)[0]

                if use_zoned:
                    matches_surf = 0
                    matches_core = 0
                    n_surf = 0
                    n_core = 0
                    for (a_start, a_end), (b_start, b_end) in zip(
                        aln.aligned[0], aln.aligned[1]
                    ):
                        for i, j in zip(
                            range(a_start, a_end), range(b_start, b_end)
                        ):
                            is_surface = (i % 2 == 0)
                            if is_surface:
                                n_surf += 1
                                if seq_a[i] == seq_b_aligned[j]:
                                    matches_surf += 1
                            else:
                                n_core += 1
                                if seq_a[i] == seq_b_aligned[j]:
                                    matches_core += 1

                    id_surf = matches_surf / max(1, n_surf)
                    id_core = matches_core / max(1, n_core)
                    identity = (
                        self.weight_core * id_core
                        + self.weight_surface * id_surf
                    )
                else:
                    identity = aln.score / max(len(seq_a), len(seq_b_aligned))

            all_candidates.append((identity, cif_path))

        survived = []
        removed = 0
        
        # Check if all would be removed
        if len(all_candidates) > 0 and all(identity > self.max_partner_identity for identity, _ in all_candidates):
            logger.warning(
                f"All candidate designs exceeded max_partner_identity ({self.max_partner_identity}). "
                f"Bypassing the threshold to avoid an empty pipeline."
            )
            survived = list(all_candidates)
        else:
            for identity, cif_path in all_candidates:
                if identity > self.max_partner_identity:
                    self._remove_files(cif_path)
                    removed += 1
                else:
                    survived.append((identity, cif_path))

        logger.info(f"Removed {removed} candidates with partner identity > {self.max_partner_identity}")

        if len(survived) == 0 and len(all_candidates) == 0:
            raise RuntimeError("No candidate designs were found to filter.")

        if self.max_surviving_designs is not None and len(survived) > self.max_surviving_designs:
            # Sort by identity ascending (lower identity = more heterotypic = better)
            survived.sort(key=lambda x: x[0])
            culling_count = len(survived) - self.max_surviving_designs
            for _, cif_path in survived[self.max_surviving_designs:]:
                self._remove_files(cif_path)
            logger.info(f"Culled an additional {culling_count} candidates to enforce max_surviving_designs={self.max_surviving_designs}")
            
    def _remove_files(self, cif_path: Path):
        cif_path.unlink(missing_ok=True)
        npz_path = cif_path.with_suffix(".npz")
        npz_path.unlink(missing_ok=True)
        native_path = cif_path.with_stem(cif_path.stem + "_native").with_suffix(".cif")
        native_path.unlink(missing_ok=True)

