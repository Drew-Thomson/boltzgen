# Double-tape robust design

When designing `double_tape` materials, a robust design workflow is used automatically:
- **Low-temperature sampling (0.2):** Forces temperature to 0.2 to preserve amphipathic patterning.
- **Side-chain orientation (`SIGMA_ORIENTATION = 0.2`):** Correctly classifies residues as inward vs outward to handle beta-register flips using a robust vector approach.
- **New metrics:** `local_double_tape_rmsd` (which measures local A-B pair RMSD instead of brittle global lattice RMSD) and `chi_outward` (which scores the proportion of outward-facing aromatic/cation-π contacts that successfully alternate between chains). The old global `neg_lattice_rmsd_refolded` cutoff is disabled for this topology.

**Example Invocation:**
```bash
boltzgen run --topology double_tape --temp 0.2
```
