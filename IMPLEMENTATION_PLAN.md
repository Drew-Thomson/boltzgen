# Implementation Plan: Topology-Aware Ligand Multiplicity

## Goal

Separate topology-guided peptide positions from ligand replication so ligands do not consume topology slots. Support non-1:1 ligand ratios with the builder config fields `ligands_per_peptide` and `ligands_per_asym_unit`, and support an explicit `attach_to` reference for per-peptide ligands. Keep emitted design input in standard BoltzGen YAML format; keep builder-only placement metadata in a sidecar file.

## Semantics and compatibility contract

- `copies` means repeated topology placements, as it does today.
- Each protein entry in `asym_unit` is one guided protein template per placement.
- Ligand templates do not count as guided topology slots.
- `ligands_per_peptide: N` expands that ligand template at ratio N per selected parent peptide across all placements. N may be a positive integer or rational number (for example, `0.5` for one ligand per two peptides), provided the total expanded count is an integer.
- `ligands_per_asym_unit: N` expands that ligand template N times per placement, independent of protein count.
- A ligand may specify only one multiplicity field. `ligands_per_asym_unit` must be a positive integer; `ligands_per_peptide` may be an integer or positive rational ratio when its total count across placements is integral.
- A ligand with neither field retains legacy behavior: one ligand per placement.
- `attach_to` is valid only with `ligands_per_peptide`; it can be a protein template `name` or zero-based protein-template index. If the unit has one protein, omitted `attach_to` resolves to it. If it has multiple proteins, it is required.
- Protein `name` is builder metadata and must not leak into the BoltzGen entity schema.
- Topology geometry/COM fitting uses protein atoms only. Ligand atoms are carried with their associated parent protein transform when parented, or with a deterministic unit transform when unit-scoped.
- Existing peptide-only configurations must retain their current emitted entity counts and behavior.
- Existing ligand-only topology entries continue to expand once per placement, but no longer alter guided protein counts.
- Special topologies that require one protein template per placement continue to enforce that protein-layout constraint, while allowing any number of ligands.

## Proposed builder config

```yaml
copies: 8
topology: cyclic
asym_unit:
  - type: protein
    name: helix
    length: 20
    secondary_structure: H
  - type: ligand
    ccd: ZN
    ligands_per_peptide: 0.5
    attach_to: helix
  - type: ligand
    ccd: CA
    ligands_per_asym_unit: 3
```

The builder should emit ordinary entity records and `material_layout.json`. The sidecar describes ordered emitted chains by zero-based BoltzGen asym-chain index, placement index, role, protein slot, ligand scope, and optional parent slot. It must not depend on chain names surviving parser preprocessing.

## Implementation steps

### Step 1 — Normalize templates and expand builder input

**Files:** `material_builder.py`; new focused tests in `tests/test_material_builder_layout.py`.

- [ ] Add a pure normalization/expansion helper that validates protein and ligand templates, names, multiplicities, and `attach_to`; allow positive rational `ligands_per_peptide` ratios if total ligand count is integral.
- [ ] Define protein template indices as zero-based order among protein entries only. Reject duplicate/missing names where a name reference is used.
- [ ] Expand each placement in stable order: protein templates first in template order, followed by each ligand template's generated instances in config order. Store this exact order in layout metadata.
- [ ] Keep `name`, `attach_to`, `ligands_per_peptide`, and `ligands_per_asym_unit` out of emitted BoltzGen YAML entities.
- [ ] Preserve existing protein `symmetric_group` behavior; use the protein-template index, not the raw asymmetric-unit entry index, for a default group.
- [ ] Write sidecar `material_layout.json` adjacent to the generated material spec and return both paths (or return a structured result without breaking current callers).
- [ ] Keep `generate_yaml_from_spec()` backwards compatible for existing direct callers or replace it with a wrapper retaining its signature.
- [ ] Do not overwrite unrelated files; preserve the current configurable spec output path behavior.

**Acceptance checks:** exact emitted entity count/order and layout mappings for peptide-only, legacy-ligand, 2 ligands per peptide, 3 per unit, and multiple proteins; invalid config cases fail before BoltzGen starts.

**Verify:**

```bash
python -m py_compile material_builder.py
PYTHONPATH=src pytest tests/test_material_builder_layout.py -q
```

### Step 2 — Make topology validation use guided protein slots

**Files:** `material_builder.py`; tests from Step 1.

- [ ] Derive guided protein count from normalized protein templates, never `len(asym_unit)`.
- [ ] Replace raw `asym_unit_size == 1` guards for `bilayer_sheet`, `hexagonal_mesh`, `nanotube`, `multi_helical`, and cage topologies with the intended protein-template constraint.
- [ ] Preserve existing exact topology placement formulas (`copies`, grid dimensions, ring/tier dimensions, fixed cage copy counts); explicitly define whether copies refer to placements versus individual protein chains for each topology and assert the mapping consistently.
- [ ] Permit ligand instances without changing topology validation.
- [ ] Set guidance layout environment metadata only after successful validation.

**Acceptance checks:** every ligand-bearing restricted topology validates with its required protein layout and rejects the wrong number of protein templates / placements; existing peptide-only validation remains unchanged.

**Verify:**

```bash
python -m py_compile material_builder.py
PYTHONPATH=src pytest tests/test_material_builder_layout.py -q
```

### Step 3 — Load layout metadata and select protein-only guidance masks

**Files:** `material_builder.py`, `src/boltzgen/model/modules/diffusion.py`; add `tests/test_material_guided_grouping.py`.

- [ ] Export an absolute `MAT_LAYOUT_FILE` path for the BoltzGen subprocess.
- [ ] Add a small, validated sidecar loader/caching helper; tolerate missing sidecar and preserve legacy grouping for non-builder BoltzGen runs.
- [ ] Map feature `asym_id` values to the sidecar's emitted chain order and use feature `mol_type`/atom-to-token mapping to form protein-only masks. Validate chain counts/order instead of silently applying a mismatched layout.
- [ ] Build topology placement masks from the guided protein slot mapping. Ligand chains must not contribute to COM or Kabsch alignment.
- [ ] Rewrite `double_tape` and `bilayer_sheet` consensus indexing to use explicit protein slot indices; preserve their existing layer/grid ordering.
- [ ] Ensure cages' required copy counts refer to guided placements, not total protein-plus-ligand chains.
- [ ] Keep padding/invalid atoms and batch broadcasting behavior unchanged.

**Acceptance checks:** a synthetic feature set with proteins interleaved with ligands yields identical protein masks/COMs to the protein-only layout; layout-chain mismatch raises a diagnostic error; absent sidecar follows legacy behavior.

**Verify:**

```bash
python -m py_compile src/boltzgen/model/modules/diffusion.py
PYTHONPATH=src python -c "from boltzgen.model.modules.diffusion import AtomDiffusion; print('diffusion import ok')"
PYTHONPATH=src pytest tests/test_material_guided_grouping.py -q
```

### Step 4 — Carry ligand coordinates with their declared scope

**Files:** `src/boltzgen/model/modules/diffusion.py`, optionally `src/boltzgen/model/modules/materials.py`; add `tests/test_material_ligand_attachment.py`.

- [ ] During topology guidance, retain the original ligand atom masks and their parent/unit mapping from the layout.
- [ ] For a per-peptide ligand, estimate the rigid transform applied to its parent protein for this guidance step and apply the same transform to ligand atoms. Do not use ligand coordinates to estimate the transform.
- [ ] For a per-unit ligand, define and document deterministic unit-transform behavior. If the unit has one guided protein, use that protein's transform. If it has multiple guided proteins, require an explicit supported reference policy or fail validation rather than choosing a parent implicitly.
- [ ] Ensure each ligand instance is updated exactly once; do not include ligand atoms in consensus averaging or protein masks.
- [ ] Preserve `guidance_scale` blending for ligand coordinates and no-op behavior when guidance is disabled.
- [ ] Account for antiparallel transforms where applicable so attached ligands follow peptide orientation flips.

**Acceptance checks:** rigid synthetic peptide+ligand coordinates preserve ligand-parent relative geometry after translation/rotation; ligand atoms do not affect the fitted COM; per-unit ambiguous transform handling is explicit and tested.

**Verify:**

```bash
python -m py_compile src/boltzgen/model/modules/diffusion.py
PYTHONPATH=src pytest tests/test_material_ligand_attachment.py -q
```

### Step 5 — Documentation and examples

**Files:** `topologies.md`; add representative builder config examples if appropriate.

- [ ] Document that topology slots count protein templates, not ligand instances.
- [ ] Document `ligands_per_peptide`, `ligands_per_asym_unit`, and `attach_to`, including defaults, validation, indexing, and examples with non-1:1 ratios.
- [ ] Clarify topology-specific protein-template and placement-count requirements remain in effect with ligands present.
- [ ] Document legacy behavior for ligand entries with no multiplicity field.

**Verify:** review docs against builder validation and example configs; `git diff --check`.

### Step 6 — Full regression and runtime validation

- [ ] Run targeted new tests and existing test suite relevant to parsing, symmetry, and inverse folding.
- [ ] Generate specs without `--run` for peptide-only and ligand-bearing examples; inspect entity count, order, and sidecar consistency.
- [ ] Run at least one cyclic and one special topology (recommended nanotube or bilayer) with ligands end-to-end when GPU/runtime is available.
- [ ] Confirm output CIF contains expected peptide and ligand chain counts and no topology count mismatch.
- [ ] Record any environment-dependent end-to-end test limitation; do not mark this step complete based solely on imports.

**Verify:**

```bash
PYTHONPATH=src pytest tests/test_material_builder_layout.py tests/test_material_guided_grouping.py tests/test_material_ligand_attachment.py -q
PYTHONPATH=src pytest tests/test_residue_constraints.py tests/test_inverse_fold_constraint_masks.py -q
python -m py_compile material_builder.py src/boltzgen/model/modules/diffusion.py src/boltzgen/model/modules/materials.py
git diff --check
```

## Risks and safeguards

- BoltzGen parsing may group repeated entity definitions by sequence; the sidecar must map the *actual parsed asym-chain order*, verified against `YamlDesignParser`, not assume raw YAML item order without a test.
- `double_tape` and `bilayer_sheet` have topology-specific slot/consensus behavior. Their mapping must be tested independently.
- A ligand placed once per asymmetric unit with multiple protein templates has no unique parent. The v1 implementation must use an explicit deterministic unit transform or reject ambiguous multi-protein units; never infer a parent silently.
- Metals may be CCD ligands without covalent bonds. Geometric co-motion is handled by the sidecar mapping; it must not rely on covalent design-mask propagation.
- Backward compatibility for legacy ligands means one emitted ligand per placement, not that the ligand remains included in topology COM calculations.
- Environment variables are process-global. Set/clear `MAT_LAYOUT_FILE` carefully so a subsequent unrelated run in the same process cannot reuse stale layout metadata.

## Implementation status

- [x] Step 1 — Normalize templates and expand builder input
- [x] Step 2 — Protein-based topology validation
- [x] Step 3 — Protein-only topology guidance masks
- [x] Step 4 — Parent/unit ligand coordinate transforms
- [x] Step 5 — Documentation and examples
- [ ] Step 6 — Regression and runtime validation (builder/diffusion targeted tests pass; full suite has 5 unrelated pre-existing constraint-test failures; end-to-end BoltzGen run pending)
