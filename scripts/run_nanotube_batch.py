# ruff: noqa: INP001,PLR2004,E501,S603,NPY002
"""Batch runner to sample 10 diverse 3-layer nanotube configurations under decoupled guidance."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Set deterministic seed
np.random.seed(42)

PYTHON_EXE = "/home/drew/anaconda3/bin/python"
OUTPUT_BASE = Path("nanotube_runs")
FIGURE_BASE = Path("figures/nanotube_diversity")

NANOTUBE_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "nanotube01_helical_c4_eclipsed",
        "desc": "3-layer C4 helical tube (eclipsed square, L=14, dz=21A, R=9A)",
        "topology": "nanotube",
        "secondary_structure": "H",
        "num_tiers": 3,
        "ring_size": 4,
        "copies": 12,
        "length": 14,
        "args": [
            "--target_radius", "9.0",
            "--target_dz", "21.0",
            "--chiral_stagger", "0.0",
        ],
    },
    {
        "id": "nanotube02_helical_c4_staggered",
        "desc": "3-layer C4 helical tube (staggered 45deg, L=15, dz=22.5A, R=9.5A)",
        "topology": "nanotube",
        "secondary_structure": "H",
        "num_tiers": 3,
        "ring_size": 4,
        "copies": 12,
        "length": 15,
        "args": [
            "--target_radius", "9.5",
            "--target_dz", "22.5",
            "--chiral_stagger", "45.0",
        ],
    },
    {
        "id": "nanotube03_helical_c6_channel",
        "desc": "3-layer C6 helical channel (collinear hexamer, L=12, dz=18A, R=12A)",
        "topology": "nanotube",
        "secondary_structure": "H",
        "num_tiers": 3,
        "ring_size": 6,
        "copies": 18,
        "length": 12,
        "args": [
            "--target_radius", "12.0",
            "--target_dz", "18.0",
            "--chiral_stagger", "0.0",
        ],
    },
    {
        "id": "nanotube04_helical_c6_chiral_twist",
        "desc": "3-layer C6 chiral helical tube (stagger 25deg, L=14, dz=21A, R=13A)",
        "topology": "nanotube",
        "secondary_structure": "H",
        "num_tiers": 3,
        "ring_size": 6,
        "copies": 18,
        "length": 14,
        "args": [
            "--target_radius", "13.0",
            "--target_dz", "21.0",
            "--chiral_stagger", "25.0",
        ],
    },
    {
        "id": "nanotube05_helical_c8_wide_lumen",
        "desc": "3-layer C8 wide helical tube (stagger 22.5deg, L=16, dz=24A, R=16A)",
        "topology": "nanotube",
        "secondary_structure": "H",
        "num_tiers": 3,
        "ring_size": 8,
        "copies": 24,
        "length": 16,
        "args": [
            "--target_radius", "16.0",
            "--target_dz", "24.0",
            "--chiral_stagger", "22.5",
        ],
    },
    {
        "id": "nanotube06_beta_c4_square_barrel",
        "desc": "3-layer C4 square beta-barrel (collinear, L=7, dz=24A, R=8A)",
        "topology": "nanotube",
        "secondary_structure": "E",
        "num_tiers": 3,
        "ring_size": 4,
        "copies": 12,
        "length": 7,
        "args": [
            "--target_radius", "8.0",
            "--target_dz", "24.0",
            "--chiral_stagger", "0.0",
        ],
    },
    {
        "id": "nanotube07_beta_c5_pentamer_barrel",
        "desc": "3-layer C5 pentamer beta-barrel (stagger 36deg, L=8, dz=27A, R=10A)",
        "topology": "nanotube",
        "secondary_structure": "E",
        "num_tiers": 3,
        "ring_size": 5,
        "copies": 15,
        "length": 8,
        "args": [
            "--target_radius", "10.0",
            "--target_dz", "27.0",
            "--chiral_stagger", "36.0",
        ],
    },
    {
        "id": "nanotube08_beta_c6_hexamer_barrel",
        "desc": "3-layer C6 hexamer beta-nanotube (collinear, L=8, dz=27A, R=12A)",
        "topology": "nanotube",
        "secondary_structure": "E",
        "num_tiers": 3,
        "ring_size": 6,
        "copies": 18,
        "length": 8,
        "args": [
            "--target_radius", "12.0",
            "--target_dz", "27.0",
            "--chiral_stagger", "0.0",
        ],
    },
    {
        "id": "nanotube09_beta_c6_chiral_shear",
        "desc": "3-layer C6 sheared beta-cylinder (stagger -20deg, L=9, dz=30A, R=13A)",
        "topology": "nanotube",
        "secondary_structure": "E",
        "num_tiers": 3,
        "ring_size": 6,
        "copies": 18,
        "length": 9,
        "args": [
            "--target_radius", "13.0",
            "--target_dz", "30.0",
            "--chiral_stagger", "-20.0",
        ],
    },
    {
        "id": "nanotube10_beta_c8_wide_pore",
        "desc": "3-layer C8 wide beta-pore (stagger 22.5deg, L=10, dz=34A, R=16A)",
        "topology": "nanotube",
        "secondary_structure": "E",
        "num_tiers": 3,
        "ring_size": 8,
        "copies": 24,
        "length": 10,
        "args": [
            "--target_radius", "16.0",
            "--target_dz", "34.0",
            "--chiral_stagger", "22.5",
        ],
    },
]


def find_trajectory_path(out_dir: Path) -> Path | None:
    """Find exported diffusion trajectory npz in output directory."""
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
    """Execute a single 3-layer nanotube diffusion generation job."""
    out_dir = OUTPUT_BASE / cfg["id"]
    existing_traj = find_trajectory_path(out_dir)

    if existing_traj is not None:
        print(f"[{cfg['id']}] Found existing trajectory at {existing_traj}")
        return existing_traj

    print(f"\n{'='*75}")
    print(f"[{cfg['id']}] Running: {cfg['desc']}")
    print(f"{'='*75}")

    cmd = [
        PYTHON_EXE,
        "material_builder.py",
        "--topology", cfg["topology"],
        "--num_tiers", str(cfg["num_tiers"]),
        "--ring_size", str(cfg["ring_size"]),
        "--copies", str(cfg["copies"]),
        "--length", str(cfg["length"]),
        "--secondary_structure", cfg["secondary_structure"],
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
    """Render tri-state deformation figure for a completed nanotube trajectory."""
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


def analyze_nanotube_geometry(traj_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate geometric metrics of the final converged nanotube structure."""
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
        "sec_struct": cfg["secondary_structure"],
    }


def main() -> None:
    """Run all 10 nanotube configurations and summarize results."""
    FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for idx, cfg in enumerate(NANOTUBE_CONFIGS, 1):
        print(f"\n>>> Running Nanotube Job {idx}/10: {cfg['id']}")
        traj_p = run_diffusion_job(cfg)
        fig_p = render_figure(cfg, traj_p)
        geom = analyze_nanotube_geometry(traj_p, cfg)
        results.append({
            "id": cfg["id"],
            "desc": cfg["desc"],
            "traj": str(traj_p),
            "fig": str(fig_p),
            **geom,
        })

    total_time = time.time() - total_start
    print("\n" + "=" * 125)
    print(f"ALL 10 NANOTUBE RUNS COMPLETE (Total Time: {total_time/60:.2f} min)")
    print("=" * 125)
    fmt = "{:<32} | {:<4} | {:<10} | {:<10} | {:<18} | {:<8}"
    print(fmt.format("Nanotube ID", "Sec", "Mean R (A)", "Z-span (A)", "CA SVD [s0, s1, s2]", "CA-CA (A)"))
    print("-" * 125)
    for r in results:
        svd_str = f"[{r['ca_svd'][0]:.1f}, {r['ca_svd'][1]:.1f}, {r['ca_svd'][2]:.1f}]"
        print(fmt.format(
            r["id"],
            r["sec_struct"],
            f"{r['mean_radius']:.2f}",
            f"{r['z_span']:.1f}",
            svd_str,
            f"{r['mean_ca_dist']:.2f}",
        ))
    print("=" * 125)


if __name__ == "__main__":
    main()
