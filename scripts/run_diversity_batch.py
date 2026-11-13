"""Batch runner to sample diverse material topologies under decoupled guidance."""

# ruff: noqa: INP001,PLR2004,E501,S603,NPY002

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Set deterministic seed
np.random.seed(42)

PYTHON_EXE = "/home/drew/anaconda3/bin/python"
OUTPUT_BASE = Path("diversity_runs")
FIGURE_BASE = Path("figures/piezo_diversity")

TOPOLOGY_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "01_helical_standard",
        "desc": "Standard open helical ring (R=15A, dz=5A, ang=30deg)",
        "topology": "helical",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "15.0",
            "--target_dz", "5.0",
            "--target_angle", "30.0",
        ],
    },
    {
        "id": "02_helical_steep_polar",
        "desc": "Steep pitch polar helix (R=9A, dz=12A, ang=45deg)",
        "topology": "helical",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "9.0",
            "--target_dz", "12.0",
            "--target_angle", "45.0",
        ],
    },
    {
        "id": "03_helical_wide_supercoil",
        "desc": "Wide open supercoil spring (R=22A, dz=4A, ang=25deg)",
        "topology": "helical",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "22.0",
            "--target_dz", "4.0",
            "--target_angle", "25.0",
        ],
    },
    {
        "id": "04_helical_compact_barrel",
        "desc": "Compact helical cylinder (R=11A, dz=3A, ang=40deg)",
        "topology": "helical",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "11.0",
            "--target_dz", "3.0",
            "--target_angle", "40.0",
        ],
    },
    {
        "id": "05_helical_left_handed",
        "desc": "Left-handed chiral helix (R=15A, dz=6A, ang=-35deg)",
        "topology": "helical",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "15.0",
            "--target_dz", "6.0",
            "--target_angle", "-35.0",
        ],
    },
    {
        "id": "06_cyclic_c8_ring",
        "desc": "Planar C8 cyclic ring (R=16A, dz=0)",
        "topology": "cyclic",
        "copies": 8,
        "length": 20,
        "args": [
            "--target_radius", "16.0",
        ],
    },
    {
        "id": "07_cyclic_c6_channel",
        "desc": "Planar C6 hexameric pore (R=12A, dz=0)",
        "topology": "cyclic",
        "copies": 6,
        "length": 20,
        "args": [
            "--target_radius", "12.0",
        ],
    },
    {
        "id": "08_double_tape_bilayer",
        "desc": "Double-tape bilayer sheet (layer_dist=10A, spacing=7.5A)",
        "topology": "double_tape",
        "copies": 8,
        "length": 20,
        "args": [
            "--layer_dist", "10.0",
            "--strand_spacing", "7.5",
        ],
    },
    {
        "id": "09_open_arc_crescent",
        "desc": "Open arc crescent (arc_radius=35A, spacing=8.5A)",
        "topology": "open_arc",
        "copies": 8,
        "length": 20,
        "args": [
            "--arc_radius", "35.0",
            "--target_arc_spacing", "8.5",
        ],
    },
    {
        "id": "10_multi_helical_3start",
        "desc": "Braided 3-start coiled-coil (R=12A, dz=6A, ang=30deg)",
        "topology": "multi_helical",
        "copies": 9,
        "length": 20,
        "args": [
            "--num_starts", "3",
            "--target_radius", "12.0",
            "--target_dz", "6.0",
            "--target_angle", "30.0",
        ],
    },
]


def find_trajectory_path(out_dir: Path) -> Path | None:
    """Find the exported diffusion trajectory npz in the output directory."""
    inter = out_dir / "intermediate_designs"
    for cand in [
        inter / "material_spec_diffusion_traj.npz",
        inter / "material_spec_0_diffusion_traj.npz",
    ]:
        if cand.exists():
            return cand
    matches = list(inter.glob("*diffusion_traj.npz"))
    if matches:
        return matches[0]
    return None


def run_diffusion_job(cfg: Dict[str, Any]) -> Path:
    """Execute a single diffusion trajectory generation job.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary describing the target topology.

    Returns
    -------
    Path
        Path to the exported trajectory archive.
    """
    out_dir = OUTPUT_BASE / cfg["id"]
    existing_traj = find_trajectory_path(out_dir)

    if existing_traj is not None:
        print(f"[{cfg['id']}] Found existing trajectory at {existing_traj}")
        return existing_traj

    print(f"\n{'='*70}")
    print(f"[{cfg['id']}] Running: {cfg['desc']}")
    print(f"{'='*70}")

    cmd = [
        PYTHON_EXE,
        "material_builder.py",
        "--topology", cfg["topology"],
        "--copies", str(cfg["copies"]),
        "--length", str(cfg["length"]),
        "--secondary_structure", "H",
        "--num_designs", "1",
        "--steps", "design",
        "--save_traj",
        "--save_x0_traj",
        "--traj_stride", "25",
        "--traj_align", "none",
        "--guidance_scale", "1.0",
        "--guidance_mode", "decoupled",
        "--guidance_pos_start", "0.85",
        "--guidance_pos_end", "0.50",
        "--guidance_shape_start", "0.50",
        "--guidance_shape_end", "0.30",
        "--output_dir", str(out_dir),
        "--run",
        *cfg["args"],
    ]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[{cfg['id']}] Diffusion completed in {dt:.1f}s")
    traj_path = find_trajectory_path(out_dir)
    if traj_path is None:
        msg = f"Trajectory archive not found in {out_dir / 'intermediate_designs'}"
        raise FileNotFoundError(msg)
    return traj_path


def render_figure(cfg: Dict[str, Any], traj_path: Path) -> Path:
    """Render tri-state deformation figure for a completed trajectory.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary describing the target topology.
    traj_path : Path
        Path to the input trajectory archive.

    Returns
    -------
    Path
        Path to the rendered figure image.
    """
    fig_path = FIGURE_BASE / f"figure_{cfg['id']}.png"

    if fig_path.exists():
        print(f"[{cfg['id']}] Found existing figure at {fig_path}")
        return fig_path

    print(f"[{cfg['id']}] Rendering figure to {fig_path}...")
    cmd = [
        PYTHON_EXE,
        "scripts/plot_diffusion_points_figure.py",
        "--helical", str(traj_path),
        "--tri_state_deformation",
        "--output", str(fig_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"[{cfg['id']}] Figure successfully written: {fig_path}")
    return fig_path


def analyze_final_geometry(traj_path: Path) -> Dict[str, Any]:
    """Calculate geometric parameters of the final converged structure.

    Parameters
    ----------
    traj_path : Path
        Path to the trajectory archive.

    Returns
    -------
    Dict[str, Any]
        Dictionary of geometric metrics.
    """
    data = np.load(traj_path)
    coords = data["coords"][-1]
    asym_id = data["asym_id"]
    pad = data["atom_pad_mask"] > 0.5
    u_asym = np.unique(asym_id)

    centroids = []
    for a in u_asym:
        sub = coords[(asym_id == a) & pad]
        centroids.append(np.mean(sub, axis=0))
    centroids = np.array(centroids)
    center = np.mean(centroids, axis=0)

    radii_xy = np.linalg.norm(centroids[:, :2] - center[:2], axis=1)
    z_span = float(np.ptp(centroids[:, 2]))

    # Subunit 0 CA metrics
    sub0_atoms = np.where((asym_id == u_asym[0]) & pad)[0]
    ca_atoms = sub0_atoms[1::14]
    cas = coords[ca_atoms]
    cas_c = cas - np.mean(cas, axis=0)
    _, s, _ = np.linalg.svd(cas_c)
    dists = np.linalg.norm(np.diff(cas, axis=0), axis=1)

    return {
        "mean_radius": float(np.mean(radii_xy)),
        "min_radius": float(np.min(radii_xy)),
        "max_radius": float(np.max(radii_xy)),
        "z_span": z_span,
        "ca_svd": s.tolist(),
        "mean_ca_dist": float(np.mean(dists)),
    }


def main() -> None:
    """Run all 10 configurations and summarize results."""
    FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for idx, cfg in enumerate(TOPOLOGY_CONFIGS, 1):
        print(f"\n>>> Running Topology Job {idx}/10: {cfg['id']}")
        traj_p = run_diffusion_job(cfg)
        fig_p = render_figure(cfg, traj_p)
        geom = analyze_final_geometry(traj_p)
        results.append({
            "id": cfg["id"],
            "desc": cfg["desc"],
            "traj": str(traj_p),
            "fig": str(fig_p),
            **geom,
        })

    total_time = time.time() - total_start
    print("\n" + "=" * 120)
    print(f"ALL 10 RUNS COMPLETE (Total Time: {total_time/60:.2f} min)")
    print("=" * 120)
    fmt = "{:<25} | {:<40} | {:<12} | {:<8} | {:<16} | {:<8}"
    print(fmt.format("Topology ID", "Description", "Mean R (A)", "Z-span", "CA SVD [s0, s1, s2]", "CA-CA (A)"))
    print("-" * 120)
    for r in results:
        svd_str = f"[{r['ca_svd'][0]:.1f}, {r['ca_svd'][1]:.1f}, {r['ca_svd'][2]:.1f}]"
        print(fmt.format(
            r["id"],
            r["desc"][:40],
            f"{r['mean_radius']:.2f}",
            f"{r['z_span']:.1f}",
            svd_str,
            f"{r['mean_ca_dist']:.2f}",
        ))
    print("=" * 120)


if __name__ == "__main__":
    main()
