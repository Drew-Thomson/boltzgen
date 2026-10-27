# BoltzGen Material Builder Topologies

## Two-chain heteromer counter-screening

`material_builder.py --heteromer_counter_screen` opts into homomer counter-screening for a two-protein asymmetric unit. The supported topologies are `cyclic`, `cylindrical`, `linear_tape`, `double_tape`, `open_arc`, and `helical`. The feature currently requires an even number of placements and a protein-only `asym_unit`; ligand-bearing heteromers are explicitly rejected as unsuitable because the metal/ligand can mediate assembly, making this a ternary interaction rather than a protein-only A/B comparison. Partner identity alternates around cyclic assemblies and along linear, arc, and helical position order. For `double_tape`, identity alternates along each side.

After inverse folding, BoltzGen creates and folds N-copy homomers for each designed partner, where N is the total number of protein chains in the intended heteromer. This compares `A_N` and `B_N` with the intended equal-stoichiometry heteromer. The feature is disabled by default and is separate from `designfolding`. Confidence differences such as `delta_design_iptm_vs_homomer_max` are model-confidence proxies, not thermodynamic measurements or experimental evidence of specificity.

`material_builder.py` can guide BoltzGen toward a range of repeated peptide-assembly geometries. Choose a topology with `--topology`; the remaining parameters below are command-line options to `material_builder.py`. Distances are in Å unless otherwise noted. Values listed as defaults reflect the builder's current CLI defaults, and some constraints apply only to the topology named.

## Shared guidance parameters

- `--guidance_scale` (default `1.0`): Strength of COM shape guidance during diffusion. Larger values adhere more strongly to the target lattice; `0` disables guidance.
- `--spacing_noise` (default `0.0`): Standard deviation of random variation in spacing targets. For `double_tape` and `bilayer_sheet`, it perturbs inter-chain and inter-layer spacing. Effective spacings are clamped to avoid values below 4.8 Å.
- `--antiparallel_prob` (default `0.0`): Probability from `0.0` to `1.0` of applying an alternating antiparallel arrangement, where supported. This setting does not control orientation for cage, mesh, nanotube, or multi-start helical topologies.
- `--asym_unit_size` (default `1`): Number of generated protein templates when using CLI options instead of an `asym_unit` config. When `asym_unit` is provided, the builder derives this value from protein entries only; ligand entries and ligand multiplicities do not affect topology slot counts. Cage, sheet, mesh, nanotube, and multi-helical topologies require one guided protein template per placement.

## Ligands in topology-guided assemblies

In material-builder YAML, topology placements are based on protein templates. Ligands are expanded as entities in the output spec but do not consume topology positions or contribute to topology COM fitting. Ligand coordinates are carried with their parent peptide (or placement for unit-scoped ligands) during guidance.

Ligand entries may specify exactly one of:

- `ligands_per_peptide` (positive integer or ratio): Number of ligand copies per parent peptide across the assembly. Integer values support multiple ligands per peptide; rational values such as `0.5` mean one ligand per two peptides. The total across all copies must be an integer, and the deterministic expansion distributes ligand instances across placements.
- `ligands_per_asym_unit` (positive integer): Number of copies generated per topology placement, independent of the number of peptide templates.
- `placement: z_axis` (or `axial`, `center`, `central`): Positions assembly-scoped ligand(s) strictly on the assembly's symmetry $z$-axis (e.g. at the pore center or opposite heights $\pm z$). Steric clash resolution operates strictly along the $z$-axis, preserving the on-axis constraint.
  - `constraint: z_axis`: Explicitly constrains the ligand to the $z$-axis.
  - `count` (positive integer, default `1`): Number of axial ligands.
  - `axial_z` (float or list of floats): Fixed axial position(s) along the $z$-axis. A scalar $z_0$ places 1 ligand at $z_0$ (default `0.0`), or 2 ligands symmetrically at $+z_0$ and $-z_0$.
  - `distribution` (`'fixed'` or `'random_even'`): When `'random_even'`, partitions `z_range` into `count` intervals and randomly samples positions within each sub-segment (respecting `min_distance`) per diffusion design candidate.
  - `z_range` (float or `[z_min, z_max]`, default `[-5.0, 5.0]`): Axial interval along $z$ for placement.
  - `min_distance` (float, default `3.5`): Minimum separation between ligands along $z$.
- `attach_to` (protein template name or zero-based protein-template index): Required with `ligands_per_peptide` when the asymmetric unit contains more than one protein template. With one protein template it may be omitted. It is not valid with `ligands_per_asym_unit` or axial ligands.

Ligands can also specify residue anchoring options to space metal ions/ligands out along the peptide rather than collapsing into the assembly core:

- `anchor_residue` (1-indexed integer or `'random'`): Pin the ligand to a specific residue's $\text{C}_\alpha$ position (or a randomly sampled internal residue in $[3, L-2]$). When set, all symmetric copies of the template within an assembly share the same anchor residue index, inheriting the parent protein's axial displacement, tilt, and antiparallel flip.
- `anchor_distance` (float, default `4.0`): Distance in $\text{\AA}$ from the anchor $\text{C}_\alpha$ coordinate to the ligand center.
- `anchor_direction` (`'outward'` / `'radial'`, `'sidechain'`, or `'auto'`, default `'outward'`): Direction to project the ligand from the anchor $\text{C}_\alpha$. `'outward'` projects radially outward from the assembly axis, keeping ions at the outer surface and interface.

If neither multiplicity field is supplied, a ligand retains the legacy default of one copy per topology placement. Give protein templates a `name` to use a stable `attach_to` reference. `copies` continues to count topology placements; total emitted chains can be higher due to ligands.

Example: one zinc ligand per two peptides and three calcium ligands per placement:

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

## Topologies

### Floating (`floating`)

No COM lattice constraints are applied. Chain placement is left to the model.

**Topology-specific parameters:** None.

### Linear tape (`linear_tape`)

Places chains in a one-dimensional array along the Z-axis, useful for tapes and fibrils.

- `--target_pitch` (default `4.8`): Target translation between adjacent chains along the tape.

### Cyclic (`cyclic`)

Places chains on a ring in the XY-plane. The angular interval is determined by the number of copies.

- `--target_radius` (default `15.0`): Ring radius.

### Cylindrical (`cylindrical`)

Variant of `cyclic` specialized for $\beta$-barrels and $\alpha$-helical barrels (coiled-coil barrels). While `cyclic` only constrains the aggregate center of mass of each chain to a circle in the XY-plane, `cylindrical` constrains the central helical axis (for $\alpha$-helices) or strand trace (for $\beta$-strands) of each peptide to lie on the surface of a cylinder with radius `--target_radius`, in accordance with the Crick parameterization. All internal 3D secondary-structure geometries and covalent bonds are preserved without flattening.

Supports multi-helix symmetrical units (`--asym_unit_size 2` or higher) with axial and angular stagger (`--target_dz`), breaking the planar COM constraint while preserving $C_{N/U}$ symmetry (or $D_{N/U}$ dihedral symmetry for antiparallel barrels). When `--spacing_noise` is positive (or explicitly via `--slide_noise` and `--radius_noise`), zero-centered slide vectors perturb each helix along its helical geodesic trajectory ($\Delta z = s_u \cos\alpha, \Delta \phi = s_u \sin\alpha / R$), while cylinder radius perturbations explore varied barrel radii across designs, preserving exact group symmetry.

- `--target_radius` (default `15.0`): Cylinder barrel radius.
- `--target_dz` (default `0.0`): Intra-unit axial stagger height between helices/strands.
- `--target_angle` (default `0.0`): Helix crossing/tilt angle relative to cylinder axis (degrees).
- `--antiparallel_prob` (default `0.0`): Probability of alternating antiparallel barrel arrangement.
- `--radius_noise` (defaults to `--spacing_noise`): Std of noise added to cylinder barrel radius across design samples.
- `--slide_noise` (defaults to `--spacing_noise`): Std of noise added to the intra-unit helical axis slide vector.

### Helical (`helical`)

Places chains on a helix about the Z-axis.

- `--target_radius` (default `15.0`): Helix radius.
- `--target_angle` (default `30.0`): Rotation between adjacent chains, in degrees.
- `--target_dz` (default `5.0`): Axial rise per chain.

### Open arc (`open_arc`)

Places chains along an incomplete circular arc in the XY-plane.

- `--arc_radius` (default `100.0`): Radius of curvature.
- `--target_arc_spacing` (default `10.0`): Target chord distance between adjacent chains.

### Double tape (`double_tape`)

Places chains in two parallel layers, with chains advancing along the Z-axis. Supports both antiparallel sheets (`--antiparallel_prob 1.0`, where alternating chains reverse direction) and parallel sheets (`--antiparallel_prob 0.0`, where all chains run in the same direction). In parallel assemblies, surface charges produce direct identical-charge self-repulsion ($q^2 > 0$) at $4.8\,\text{Å}$ in homomers, while antiparallel assemblies require sequence asymmetry ($q(i) \cdot q(L - 1 - i) \ge 0$) to destabilize homomers.

- `--target_pitch` (default `4.8`): Spacing between neighboring chains along each tape.
- `--layer_dist` (default `10.0`): Separation between layers.
- `--double_tape_core_bias` (default `0.0`): Sequence generation bias for the inward-facing core. Decreases Threonine/Serine and boosts hydrophobic residues (Ala, Val, Ile, Leu) at structurally inward-facing positions.
- `--double_tape_aromatic_bias` (default `1.0`): Sequence generation bias for the outward-facing exterior. Boosts Tryptophan and Tyrosine at outward-facing positions to promote stabilizing aromatic pairs.
- `--double_tape_polar_bias` (default `0.0`): Sequence generation bias for the outward-facing exterior. Boosts Arg, Lys, Asp, Glu, Asn, Gln, His, Thr, Ser at outward-facing positions to support salt-bridge and hydrogen-bonding networks.

## Sequence Biasing and Heteromer Design

While `material_builder.py` primarily drives topology placement, it also exposes advanced flags for controlling sequences during the inverse-folding and identity-filtering stages:

- `--heteromer_counter_screen`: Instructs the pipeline to design a two-chain heteromer with alternating partner identities (e.g. A-B-A-B). Includes A-only and B-only counter-screen folding steps to filter out designs that might form unintended homomers.
- `--heteromer_charge_bias` (default `0.5`): Strength of global charge complementarity bias applied during inverse folding to enforce positive/negative charge pairing across partners.
- `--anti_correlation_strength` (default `20.0` when counter-screen active, `0.0` = disabled): Strength of cross-partner anti-correlation penalty in raw logit units applied during inverse folding to maximize sequence divergence between partners (recommended range: 20.0–50.0).
- `--anti_correlation_mode` (default `exact`): Penalty weighting scheme for cross-partner anti-correlation (`exact` for identity ban, `blosum62` for similarity-weighted penalty).
- `--anti_correlation_surface_only`: Restricts the cross-partner logit penalty during inverse folding to solvent-facing positions. Leaves inward core positions unpenalized so both partners can share compatible hydrophobic core motifs (e.g. alternating Val/Ala) required for stable bilayer packing.
- `--partner_orientation` (default `auto`, choices: `auto`, `parallel`, `antiparallel`): Relative strand orientation for partner sequence identity filtering and anti-correlation mapping. Under `auto`, relative orientation $\sigma = \operatorname{sgn}(\hat{\mathbf{u}}_A \cdot \hat{\mathbf{u}}_B)$ is detected dynamically from $\text{C}_\alpha$ vectors in the generated structure; if coordinates are absent, falls back to lattice metadata.
- `--max_partner_identity` (default `0.5`): Maximum allowed sequence identity between partners A and B before running expensive structural counter-screens.
- `--identity_filter_zoned` / `--no-identity_filter_zoned`: Activates zoned sequence identity evaluation. Automatically enabled for `double_tape`, `linear_tape`, and `bilayer_sheet`.
- `--identity_filter_weight_core` (default `0.0`): Weight for inward core residue identity in zoned filtering. Setting to `0.0` allows core residues to be identical across partners without triggering false-positive counter-screen filtering.
- `--identity_filter_weight_surface` (default `1.0`): Weight for outward solvent-facing residue identity in zoned filtering, concentrating divergence requirements where salt bridges and homomer repulsion operate.
- `--max_surviving_designs`: Optional cap on the number of sequence designs to pass into the counter-screen folding step.

### Tetrahedral cage (`cage_tetrahedral`)

Arranges a fixed set of chains on a tetrahedral cage lattice.

- `--target_cage_radius` (default `20.0`): Cage radius.
- **Copy constraint:** exactly 12 copies; `--asym_unit_size 1`.

### Octahedral cage (`cage_octahedral`)

Arranges a fixed set of chains on an octahedral cage lattice.

- `--target_cage_radius` (default `35.0`): Cage radius.
- **Copy constraint:** exactly 24 copies; `--asym_unit_size 1`.

### Bilayer sheet (`bilayer_sheet`)

Creates two parallel, rectangular chain sheets. Each sheet has `grid_dim_x` by `grid_dim_y` positions.

- `--grid_dim_x` (default `2`): Grid width.
- `--grid_dim_y` (default `2`): Grid height.
- `--row_pitch` (default `10.0`): In-plane row spacing.
- `--layer_dist` (default `10.0`): Separation between the sheets.
- `--target_pitch` (default `4.8`): Spacing along the other in-plane direction.
- **Copy constraint:** `--copies` must equal `2 * grid_dim_x * grid_dim_y`; `--asym_unit_size 1`.

### Hexagonal mesh (`hexagonal_mesh`)

Builds a hexagonal pore mesh from three chain orientations at each grid position.

- `--grid_dim_x` (default `2`): Grid width.
- `--grid_dim_y` (default `2`): Grid height.
- `--pore_diameter` (default unset): Target pore diameter. Use either this or `--lattice_constant`, not both. If only pore diameter is given, the builder derives the lattice constant as `pore_diameter + 10.0`.
- `--lattice_constant` (default unset): Hexagonal lattice constant; alternative to `--pore_diameter`.
- **Copy constraint:** `--copies` must equal `3 * grid_dim_x * grid_dim_y`; `--asym_unit_size 1`.
- **Required input:** provide either `--pore_diameter` or `--lattice_constant`.

### Nanotube (`nanotube`)

Stacks oligomeric rings into a tube along the Z-axis.

- `--ring_size` (default `4`): Number of chains in each ring.
- `--num_tiers` (default `4`): Number of stacked rings.
- `--target_radius` (default `15.0`): Tube radius.
- `--target_dz` (default `4.8`): Axial rise between tiers (default is topology-specific).
- `--chiral_stagger` (default `0.0`): Rotation offset per tier, in degrees.
- **Copy constraint:** `--copies` must equal `ring_size * num_tiers`; `--asym_unit_size 1`.

### Multi-helical (`multi_helical`)

Arranges chains as multiple interleaved helical starts around a common axis.

- `--num_starts` (default `3`): Number of helical strands.
- `--target_radius` (default `15.0`): Helix radius.
- `--target_angle` (default `30.0`): Rotation between sequential chain positions, in degrees.
- `--target_dz` (default `5.0`): Axial rise per chain.
- **Copy constraint:** `--copies` must be divisible by `num_starts`; `--asym_unit_size 1`.

## Examples

Generate an antiparallel helical assembly:

```bash
python material_builder.py --copies 20 --length 15 --topology helical \
    --target_radius 15 --target_dz 5 --target_angle 30 \
    --antiparallel_prob 1.0 --run
```

Generate a bilayer with 3-by-4 grids (24 chains total):

```bash
python material_builder.py --copies 24 --length 15 --topology bilayer_sheet \
    --grid_dim_x 3 --grid_dim_y 4 --row_pitch 10 --layer_dist 10 --run
```

Generate a hexagonal mesh with a requested pore diameter:

```bash
python material_builder.py --copies 12 --length 15 --topology hexagonal_mesh \
    --grid_dim_x 2 --grid_dim_y 2 --pore_diameter 20 --run
```

Generate a four-tier, six-member nanotube:

```bash
python material_builder.py --copies 24 --length 15 --topology nanotube \
    --ring_size 6 --num_tiers 4 --target_radius 15 --target_dz 4.8 --run
```
