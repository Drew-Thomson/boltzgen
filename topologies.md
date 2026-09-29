# BoltzGen Material Builder Topologies

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
- `attach_to` (protein template name or zero-based protein-template index): Required with `ligands_per_peptide` when the asymmetric unit contains more than one protein template. With one protein template it may be omitted. It is not valid with `ligands_per_asym_unit`.

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

Places chains in two parallel layers, with chains advancing along the Z-axis. With `--antiparallel_prob 1.0`, alternating chains form antiparallel sheets in both layers.

- `--target_pitch` (default `4.8`): Spacing between neighboring chains along each tape.
- `--layer_dist` (default `10.0`): Separation between layers.

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
