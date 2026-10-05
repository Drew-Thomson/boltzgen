# Engineering Handoff: Two-Chain Heteromer Counter-Screening

## Objective

Implement an **opt-in** workflow for two-chain heteromeric material designs that compares the intended heteromeric assembly (`A_(N/2) + B_(N/2)`) against homomeric counter-screens (`A_N` and `B_N`). Prioritize candidates that fold/assemble confidently in the intended heteromeric topology and less confidently as either homomer.

## Fixed decisions and v1 scope

- Support exactly two distinct protein templates, A and B.
- Counter-screening is opt-in; default behavior and outputs must remain unchanged.
- Support `cyclic`, `linear_tape`, `double_tape`, `open_arc`, and `helical`.
- Use the existing `helical` topology; do not add `helical_tape`.
- Alternate A/B around cyclic assemblies and along the ordered positions of linear tapes, open arcs, and helices.
- For `double_tape`, alternate A/B **along each side**; do not assign one partner exclusively to each layer.
- Homomer counter-screens use the same total protein copy count and topology parameters as the heteromer.
- Do not overload the existing `designfolding` feature; homomer screening is a distinct workflow.

## Relevant code paths

- Builder/layout: `material_builder.py`, `src/boltzgen/model/modules/material_layout.py`
- Topology guidance: `src/boltzgen/model/modules/materials.py`, `src/boltzgen/model/modules/diffusion.py`
- Pipeline/prediction: `src/boltzgen/cli/boltzgen.py`, `src/boltzgen/task/predict/data_from_generated.py`, `src/boltzgen/task/predict/writer.py`
- Analysis: `src/boltzgen/task/analyze/analyze.py`, `src/boltzgen/task/analyze/analyze_utils.py`, `src/boltzgen/task/analyze/material_metrics.py`
- Filtering: `src/boltzgen/task/filter/filter.py`, `src/boltzgen/task/filter/material_presets.py`, `src/boltzgen/resources/config/filtering.yaml`
- Docs: `topologies.md`, `example/README.md`, representative example YAMLs

## Implementation steps

### Step 1 — Add opt-in validation and define placement semantics

**Files:** `material_builder.py`, `src/boltzgen/model/modules/material_layout.py`

- Add a counter-screen opt-in flag/config setting and pass it into the material pipeline.
- When enabled, require exactly two protein templates and a supported topology (`cyclic`, `linear_tape`, `double_tape`, `open_arc`, `helical`). Fail before launching BoltzGen for unsupported input.
- Define topology placement order as the source of A/B assignment. Alternate partners around/along that order; explicitly encode `double_tape` as alternating along each side.
- Preserve current behavior when the option is disabled.

**Acceptance:** valid enabled inputs pass; one or more than two protein templates and unsupported topologies fail clearly; legacy inputs behave unchanged.

**Verify:**
```bash
python -m py_compile material_builder.py src/boltzgen/model/modules/material_layout.py
PYTHONPATH=src pytest tests/test_material_builder_layout.py -q
```

### Step 2 — Persist partner identity in layout metadata

**Files:** `src/boltzgen/model/modules/material_layout.py`, `material_builder.py`

- Extend `material_layout.json` with stable per-protein metadata: partner label (`A`/`B`), protein-template index, placement/order index, and topology pattern.
- Add run-level metadata indicating whether heteromer screening is enabled and whether the topology is counter-screen eligible.
- Keep emitted BoltzGen entity order and existing ligand-layout behavior backward-compatible.
- Do not rely on chain names surviving parser preprocessing; map using actual emitted/parsed asym-chain order.

**Acceptance:** every emitted protein chain maps deterministically to A or B; layout ordering is covered by tests; existing layout tests remain valid.

**Verify:**
```bash
python -m py_compile src/boltzgen/model/modules/material_layout.py
PYTHONPATH=src pytest tests/test_material_builder_layout.py tests/test_material_guided_grouping.py -q
```

### Step 3 — Generate deterministic homomer counter-screen inputs

**Files:** add a focused helper near the prediction workflow; update layout/builder code only as needed.

- From each inverse-folded heteromer candidate, generate two derived inputs using the designed sequences:
  - A homomer: N copies of sequence A
  - B homomer: N copies of sequence B
- Preserve the original topology and all relevant topology parameters for each derived input.
- Assign deterministic IDs and persist an explicit mapping from the original heteromer design to its A and B counter-screen IDs.
- Ensure derived input construction does not change the original heteromer input.

**Acceptance:** helper output has the expected sequences, counts, topology parameters, and stable parent/child IDs; invalid or incomplete mappings produce diagnostic errors.

**Verify:**
```bash
PYTHONPATH=src pytest --mock-heavy-deps tests/test_heteromer_material_counter_screens.py -q
```

### Step 4 — Run counter-screen folds as a separate prediction workflow

**Files:** `src/boltzgen/cli/boltzgen.py`, `src/boltzgen/task/predict/data_from_generated.py`, `src/boltzgen/task/predict/writer.py`

- Add a pipeline stage for the derived A and B homomer inputs when counter-screening is enabled.
- Keep existing heteromer refolding and `designfolding` paths unchanged.
- Store counter-screen predictions/CIFs in explicit, separate output locations or use an equivalently unambiguous naming scheme.
- Retain required confidence values in prediction outputs, including `complex_plddt`, `design_iptm`, `protein_iptm`, and `min_interaction_pae` where available.
- Ensure pipeline resume/skip-existing behavior distinguishes heteromer, homomer A, and homomer B jobs.

**Acceptance:** enabled runs produce all three fold results; disabled runs produce only existing outputs; resume behavior does not confuse the three result types.

**Verify:**
```bash
python -m py_compile src/boltzgen/cli/boltzgen.py src/boltzgen/task/predict/data_from_generated.py src/boltzgen/task/predict/writer.py
PYTHONPATH=src python -c "from boltzgen.task.predict.writer import FoldingWriter; print('writer import ok')"
```

### Step 5 — Aggregate raw metrics and heteromer/homomer contrasts

**Files:** `src/boltzgen/task/analyze/analyze.py`, `src/boltzgen/task/analyze/analyze_utils.py`

- Join counter-screen results to the original design using the persisted parent/child mapping; do not infer joins from file ordering.
- Preserve the raw heteromer and homomer metrics in the aggregate row. At minimum include:
  - `heteromer_complex_plddt`, `heteromer_design_iptm`, `heteromer_protein_iptm`, `heteromer_min_interaction_pae`
  - corresponding `homomer_a_*` and `homomer_b_*` values
- Derive higher-is-better contrast columns, at minimum:
  - `delta_complex_plddt_vs_homomer_max`
  - `delta_design_iptm_vs_homomer_max`
  - `delta_protein_iptm_vs_homomer_max`
- If retaining PAE contrasts, document/sign them so higher-is-better ranking is unambiguous.
- If confidence outputs contain dictionaries (e.g. pairwise chain metrics), convert them into deterministic scalar summaries before writing CSV.
- Keep non-counter-screen analysis operational and avoid silent joins when a counter-screen result is missing.

**Acceptance:** enabled aggregate CSV rows contain the correct raw and derived values; missing results are diagnosed; existing analysis remains usable without screening.

**Verify:**
```bash
python -m py_compile src/boltzgen/task/analyze/analyze.py src/boltzgen/task/analyze/analyze_utils.py
PYTHONPATH=src pytest --mock-heavy-deps tests/test_heteromer_material_analysis.py -q
```

### Step 6 — Add/reuse topology-fidelity metrics

**Files:** `src/boltzgen/task/analyze/material_metrics.py`, `src/boltzgen/task/analyze/analyze.py`

- Determine whether existing `neg_lattice_rmsd_refolded` sufficiently measures topology fidelity for all five v1 topologies; reuse it where reliable.
- Add lightweight CPU metrics only where needed: cyclic radial regularity, tape spacing consistency, double-tape side-wise arrangement, arc curvature consistency, or helical radius/pitch consistency.
- Make metric definitions topology-aware and deterministic. Provide corresponding heteromer/homomer values and a higher-is-better contrast only if the underlying metric adds information beyond existing lattice RMSD.

**Acceptance:** supported topologies are handled without regressions; geometry metrics are tested on ideal and perturbed synthetic layouts; no redundant score is added without justification.

**Verify:**
```bash
python -m py_compile src/boltzgen/task/analyze/material_metrics.py
PYTHONPATH=src pytest tests/test_material_metrics.py -q
```

### Step 7 — Add filtering/ranking support

**Files:** `src/boltzgen/task/filter/filter.py`, `src/boltzgen/task/filter/material_presets.py`, `src/boltzgen/resources/config/filtering.yaml`

- Add an opt-in heteromer preset/ranking configuration using the contrast columns.
- Prioritize `delta_design_iptm_vs_homomer_max` and `delta_complex_plddt_vs_homomer_max`; use heteromer quality/topology metrics as complementary objectives.
- Support optional hard filters for minimum heteromer confidence and positive heteromer-over-homomer contrast.
- Ensure filtering fails clearly or skips the heteromer preset when required columns are absent; preserve existing non-heteromer presets.

**Acceptance:** ranking favors strong heteromer-specific candidates in synthetic test data; legacy filter configurations remain unchanged.

**Verify:**
```bash
python -m py_compile src/boltzgen/task/filter/filter.py src/boltzgen/task/filter/material_presets.py
PYTHONPATH=src pytest --mock-heavy-deps tests/test_heteromer_material_filtering.py -q
```

### Step 8 — Tests and documentation

**Files:** add focused heteromer tests; update `topologies.md`, `example/README.md`, and examples.

- Add tests for opt-in gating, exactly-two-template validation, partner placement order for every supported topology, and `double_tape` side-wise alternation.
- Test deterministic counter-screen input generation and parent/child ID mapping.
- Test aggregation and contrast calculations, including missing-result behavior.
- Test filtering/ranking on contrast metrics and legacy fallback.
- Document supported topologies, partner order, the opt-in flag, and the `A_N`/`B_N` vs `A_(N/2)+B_(N/2)` comparison.
- Add representative cyclic, linear-tape, and double-tape examples.

**Verify:**
```bash
pytest --mock-heavy-deps tests/test_heteromer_material_counter_screens.py tests/test_heteromer_material_analysis.py tests/test_heteromer_material_filtering.py -q
git diff --check
```

## Risks and safeguards

- **Do not overload `designfolding`:** counter-screening is a separate prediction job and output type.
- **Double-tape ordering:** implement and test side-wise alternation explicitly; do not infer it from layer identity.
- **Join correctness:** use explicit persisted IDs/metadata for all homomer results.
- **Topology consistency:** homomer screens must reuse the heteromer's topology and parameters.
- **Metric direction:** derived ranking columns should consistently be higher-is-better.
- **Backward compatibility:** gate new jobs, metrics, and presets behind opt-in; retain existing behavior otherwise.
- **Cost:** two additional folds per candidate are expected when enabled; do not launch them by default.
- **Metal/ligand ternary complexes:** do not apply the current A/B homomer contrast to ligand-bearing heteromers. Ligands can bridge or stabilize the assembly, so protein-only A_N/B_N controls confound partner specificity with missing ligand coordination. V1 rejects counter-screening when any ligand is present in `asym_unit`; normal ligand-bearing design remains available without the counter-screen flag.

## Verification and completion criteria

### Targeted existing material tests
```bash
PYTHONPATH=src pytest \
  tests/test_material_builder_layout.py \
  tests/test_material_guided_grouping.py \
  tests/test_material_ligand_attachment.py \
  tests/test_material_metrics.py -q
```

### New focused tests
```bash
pytest --mock-heavy-deps \
  tests/test_heteromer_material_counter_screens.py \
  tests/test_heteromer_material_analysis.py \
  tests/test_heteromer_material_filtering.py -q
```

### Full regression
```bash
pytest tests/
```

### Runtime validation when GPU/model environment is available
- Run one cyclic and one linear-tape or double-tape two-chain heteromer with counter-screening enabled.
- Verify the heteromer, homomer A, and homomer B prediction outputs exist and map to the original candidate.
- Verify aggregate CSV contains raw and contrast metrics and filtering uses the intended columns.
- Run a non-counter-screen material design and confirm its behavior/output remains unchanged.

Do not mark implementation complete until targeted tests pass, full-suite results are recorded (including any known unrelated failures), and at least one enabled end-to-end runtime check is completed or explicitly documented as environment-blocked.

## Execution status

- [x] Step 1 — Added opt-in `material_builder.py --heteromer_counter_screen`, supported-topology/two-template/even-placement validation, and deterministic partner assignment (including alternating identity along each `double_tape` side).
- [x] Step 2 — Added partner/template/order/pattern and topology parameter metadata to the material layout sidecar.
- [x] Step 3 — Added deterministic A/B homomer entity/spec helpers and a preparation task that extracts each partner from the inverse-folded assembly, places N homomer copies, writes paired CIF/NPZ inputs, and persists a parent-to-screen mapping.
- [x] Step 4 — Added opt-in preparation and separate homomer folding pipeline stages; added stage-selection validation and output confidence keys. Counter-screening is rejected with `--skip_inverse_folding`.
- [x] Step 5 — Added heteromer confidence extraction, counter-screen metric merge, and higher-is-better confidence contrast columns.
- [ ] Step 6 — Dedicated cyclic/tape/arc/helical topology-fidelity contrast metrics are not implemented yet. Existing lattice RMSD can be used in later work after validating its applicability to all listed topologies.
- [x] Step 7 — Added contrast-ranking filter overrides for opt-in counter-screening.
- [x] Step 8 — Added focused tests and documented the supported feature in `topologies.md`.
- [x] Policy safeguard — Ligand-bearing heteromer specs are rejected for this screen with an explicit ternary-assembly explanation; ligand-free heteromer designs remain supported.
- [ ] Step 9 — Examples and full end-to-end validation remain pending.

### Verification run

- Passed compile checks for changed Python modules.
- Passed focused tests: 53 passed across material builder/layout, guidance masks, existing material metrics, counter-screen helpers, analysis, and filtering. One environment warning reports an older `numexpr` version than pandas recommends.
- `python material_builder.py --help` confirms the new opt-in flag is registered.
- `git diff --check` passed.
- Full `pytest tests/` collection is blocked by the pre-existing untracked `tests/test_materials.py`, which imports `OCTAHEDRAL_ROTATIONS` from the current `materials.py` implementation although that symbol is not present in the checked-in implementation. This task did not alter that untracked test or the topology implementation to satisfy it.
- GPU end-to-end execution was not run. The counter-screen CIF/NPZ preparation path and pipeline orchestration still need runtime validation before this feature is considered complete.
