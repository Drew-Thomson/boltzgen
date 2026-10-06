import json
import logging
from pathlib import Path

from Bio import Align

from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.task.task import Task

logger = logging.getLogger(__name__)


class FilterHighIdentityPairs(Task):
    """Remove inverse-folded candidates where partner A and B sequences are too similar."""

    def __init__(self, design_dir: str, layout_path: str, moldir: str = None, max_partner_identity: float = 0.5, max_surviving_designs: int = None):
        self.design_dir = Path(design_dir)
        self.layout_path = Path(layout_path)
        self.moldir = moldir
        self.max_partner_identity = max_partner_identity
        if isinstance(max_surviving_designs, str):
            if max_surviving_designs.lower() == "none":
                max_surviving_designs = None
            else:
                max_surviving_designs = int(max_surviving_designs)
        self.max_surviving_designs = max_surviving_designs

    def run(self, config=None):
        if not self.design_dir.exists():
            logger.warning(f"Design dir {self.design_dir} does not exist. Skipping.")
            return

        try:
            layout = json.loads(self.layout_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load layout from {self.layout_path}: {e}")
            return

        chain_to_partner = {
            r["chain_id"]: r.get("partner_label")
            for r in layout.get("chains", []) if r.get("role") == "protein"
        }

        removed = 0
        survived = []
        for cif_path in sorted(self.design_dir.glob("*.cif")):
            if cif_path.name.endswith("_native.cif"):
                continue

            try:
                parsed = parse_mmcif(cif_path, moldir=self.moldir, use_original_res_idx=False)
            except Exception as e:
                logger.warning(f"Failed to parse {cif_path}: {e}")
                continue

            seq_a = seq_b = None
            for chain_name, seq in parsed.sequences.items():
                partner = chain_to_partner.get(chain_name)
                if partner == "A" and seq_a is None:
                    seq_a = seq
                elif partner == "B" and seq_b is None:
                    seq_b = seq
            
            if seq_a is None or seq_b is None:
                continue

            if len(seq_a) == len(seq_b):
                matches = sum(a == b for a, b in zip(seq_a, seq_b))
                identity = matches / max(len(seq_a), len(seq_b))
            else:
                aligner = Align.PairwiseAligner()
                aln = aligner.align(seq_a, seq_b)[0]
                identity = aln.score / max(len(seq_a), len(seq_b))

            if identity > self.max_partner_identity:
                self._remove_files(cif_path)
                removed += 1
            else:
                survived.append((identity, cif_path))

        logger.info(f"Removed {removed} candidates with partner identity > {self.max_partner_identity}")

        if len(survived) == 0:
            raise RuntimeError(
                f"All candidate designs were removed because partner identity exceeded "
                f"max_partner_identity ({self.max_partner_identity}). "
                f"Please relax --max_partner_identity or increase --anti_correlation_strength."
            )

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

