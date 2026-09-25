# BoltzGen Material Builder Topologies

The `material_builder.py` script allows you to generate highly symmetrical peptide assemblies using the BoltzGen diffusion model. By applying specific COM (Center of Mass) guidance at each diffusion timestep, you can coerce the model into generating specific geometric structures. 

Below is an overview of the available topologies, their behavior, and the parameters used to configure them.

## General Guidance Parameters

These parameters can be applied to any guided topology to adjust how the model samples geometries:

- `--guidance_scale` (Default: `1.0`): The strength of the shape guidance applied during the diffusion process. Higher values force the chains to strictly adhere to the ideal coordinates, while lower values give the network more flexibility.
- `--spacing_noise` (Default: `0.0`): The standard deviation of random noise (in Ångströms) to add to the spacing target. Setting this > 0 allows the model to sample more diverse packing arrangements across a batch. It has a hard lower bound of 4.8 Å to prevent chains from clashing (the typical spacing of a beta-sheet).
- `--antiparallel_prob` (Default: `0.0`): The probability (0.0 to 1.0) of generating an antiparallel arrangement. If triggered, every alternating chain in the assembly is flipped 180° around its local radial/lateral axis. This allows for C2/D2-like symmetries (e.g., antiparallel beta tapes or alternating alpha solenoids).

---

## 1. Floating (`floating`)
The default topology. No COM constraints are applied during diffusion. The chains are allowed to pack freely based purely on the diffusion model's learned physics and interactions.
* **Relevant Arguments**: None (Ignores all structural constraints).

## 2. Linear Tape (`linear_tape`)
Aligns the chains in a straight, 1-dimensional array along the Z-axis. Ideal for generating beta-tapes, amyloid-like fibrils, or parallel/antiparallel flat assemblies.
* **Relevant Arguments**:
  * `--target_pitch` (Default: `10.0` Å): The target spacing (translation) between adjacent chains along the tape.

## 3. Cyclic (`cyclic`)
Arranges the chains in a closed ring (C_n symmetry) in the XY-plane. The angle between each chain is automatically calculated based on the total number of `--copies` to form a perfect circle.
* **Relevant Arguments**:
  * `--target_radius` (Default: `15.0` Å): The target radius of the ring.

## 4. Helical (`helical`)
Arranges the chains in a continuous helical spiral. Chains translate along the Z-axis while simultaneously rotating around it. Ideal for generating alpha solenoids, helical filaments, and nanotubes.
* **Relevant Arguments**:
  * `--target_radius` (Default: `15.0` Å): The target radius of the helix from the central Z-axis.
  * `--target_angle` (Default: `30.0` degrees): The target rotation angle applied between each adjacent chain.
  * `--target_dz` (Default: `5.0` Å): The target axial translation (rise) per chain along the Z-axis. 

## 5. Open Arc (`open_arc`)
Arranges the chains along a curved path (an incomplete ring) in the XY-plane. Unlike `cyclic`, the chains do not close into a full circle. This is useful for crescent-shaped assemblies or large curved fragments.
* **Relevant Arguments**:
  * `--arc_radius` (Default: `100.0` Å): The target radius of curvature for the arc.
  * `--target_arc_spacing` (Default: `10.0` Å): The target distance (chord length) between adjacent chains along the arc.

## 6. Double Tape (`double_tape`)
Aligns the chains into two parallel planes to form a double-layer tape (e.g., a steric zipper or sandwich). The chains alternate evenly between the two layers and propagate along the Z-axis. When combined with `--antiparallel_prob 1.0`, it natively enforces alternating orientations within each layer (forming two true antiparallel beta sheets) that are also antiparallel to each other face-to-face.
* **Relevant Arguments**:
  * `--target_pitch` (Default for this topology: `4.8` Å): The spacing between adjacent chains within the same layer along the Z-axis.
  * `--layer_dist` (Default: `10.0` Å): The distance between the two parallel layers.

---

### Example Usage

Generate an antiparallel helical alpha-solenoid (20 copies, length 15):
```bash
python material_builder.py --copies 20 --length 15 --topology helical \
    --target_radius 15.0 --target_dz 5.0 --target_angle 30.0 \
    --antiparallel_prob 1.0 --run
```

Generate diverse linear beta-tapes with slight variations in spacing:
```bash
python material_builder.py --copies 10 --length 8 --topology linear_tape \
    --target_pitch 4.8 --spacing_noise 0.5 --run
```