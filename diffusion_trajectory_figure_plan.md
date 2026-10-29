# Self-Contained Implementation Plan & Execution Guide: Diffusion Point-Cloud Trajectory Figure

This document contains all instructions, code changes, and execution steps to run this pipeline on a separate GPU machine.

---

## 1. Overview & Visual Objective
Generate a publication-quality figure comparing:
- **Row 1: Traditional Binder Design**: Fixed target protein (slate gray dots) acting as external context; binder chain begins as a random Gaussian point cloud and coalesces into the target binding pocket.
- **Row 2: Symmetry-Driven Helical Design**: No external target; subunits start as random coordinates replicated according to helical screw symmetry $(R_z(i \cdot \theta), \mathbf{t}_z(i \cdot dz))$, where each chain serves as its own structural context for neighboring chains as they coalesce into a self-assembled helix.

---

## 2. Code Changes

### Step 2.1: Update `DesignWriter` in `src/boltzgen/task/predict/writer.py`
Add trajectory `.npz` point-cloud export support with unaligned coordinates (`traj_align="none"`) to preserve the helical $Z$-axis.

In `src/boltzgen/task/predict/writer.py`:
- In `__init__`:
  ```python
  self.save_traj = save_traj
  self.save_x0_traj = save_x0_traj
  self.traj_stride = traj_stride
  self.traj_align = traj_align
  self.save_traj_npz = save_traj_npz
  ```
- In `write_on_batch_end`:
  Extract `coords_traj` ($x_t$), `x0_coords_traj` ($\hat{x}_0(t)$), `design_mask`, and `asym_id`:
  ```python
  if (self.save_traj or self.save_x0_traj) and self.save_traj_npz:
      traj_x = (
          torch.stack(prediction["coords_traj"], dim=1)[n]
          if "coords_traj" in prediction
          else None
      )
      traj_x0 = (
          torch.stack(prediction["x0_coords_traj"], dim=1)[n]
          if "x0_coords_traj" in prediction
          else None
      )

      traj_ref = traj_x if traj_x is not None else traj_x0
      T = traj_ref.shape[0]
      stride = max(1, getattr(self, "traj_stride", 1))
      indices = list(range(0, T, stride))
      if (T - 1) not in indices:
          indices.append(T - 1)

      atom_design_mask = (
          sample["atom_to_token"].float() @ batch["design_mask"][0].unsqueeze(-1).float()
      ).squeeze().bool()

      token_indices = sample["atom_to_token"].long().argmax(dim=-1)
      atom_asym_id = sample["asym_id"][token_indices]

      np.savez_compressed(
          self.outdir / f"{file_name}_diffusion_traj.npz",
          coords=traj_x[indices].cpu().numpy() if traj_x is not None else None,
          x0_coords=traj_x0[indices].cpu().numpy() if traj_x0 is not None else None,
          is_target=(~atom_design_mask).cpu().numpy(),
          asym_id=atom_asym_id.cpu().numpy(),
          atom_pad_mask=sample["atom_pad_mask"].cpu().numpy(),
          step_indices=np.array(indices),
      )
  ```

---

### Step 2.2: Add CLI Arguments to `src/boltzgen/cli/boltzgen.py`
Add `--save_traj`, `--save_x0_traj`, and `--traj_stride` to diffusion arguments and pass them to Hydra step configurations.

---

### Step 2.3: Point-Cloud Scatter Plotting Script
Create `scripts/plot_diffusion_points_figure.py`:
- Loads `*_diffusion_traj.npz` for both binder and helical runs.
- Extracts 5 key timepoints ($t = 1.0, 0.75, 0.50, 0.25, 0.0$).
- Plots 3D scatter point clouds:
  - Row 1: Target in muted slate gray; binder in coral/orange points.
  - Row 2: 16 helical chains in viridis gradient along the screw axis.

---

## 3. Execution Commands on the Target Machine

### 3.1 Run Traditional Binder Diffusion (GPU)
```bash
boltzgen run example/vanilla_peptide_with_target_binding_site/beetletert.yaml \
  --output run_binder \
  --num_designs 1 \
  --steps design \
  --config design writer.save_traj=True writer.save_x0_traj=True writer.traj_stride=25 writer.traj_align=none
```

### 3.2 Run Helical Symmetry Diffusion (GPU)
```bash
python material_builder.py \
  --topology helical \
  --copies 16 \
  --length 18 \
  --target_radius 15.0 \
  --target_dz 5.0 \
  --target_angle 30.0 \
  --num_designs 1 \
  --steps design \
  --output_dir run_helical \
  --config design writer.save_traj=True writer.save_x0_traj=True writer.traj_stride=25 writer.traj_align=none
```

### 3.3 Render Comparison Figure
```bash
python scripts/plot_diffusion_points_figure.py \
  --binder run_binder/intermediate_designs/*_diffusion_traj.npz \
  --helical run_helical/intermediate_designs/*_diffusion_traj.npz \
  --output figures/diffusion_points_comparison.png
```
