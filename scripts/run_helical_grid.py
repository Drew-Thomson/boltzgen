"""Orchestrate and benchmark parameter grid for axial helical alignment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.analyze_helical_alignment import analyze_structure_helical_alignment

CONFIGS = [
    {
        "name": "run_dz08_ang30",
        "dz": 8.0,
        "angle": 30.0,
        "radius": 8.0,
        "desc": "Reference rise (dz=8A, ang=30, pitch=96A)",
    },
    {
        "name": "run_dz15_ang30",
        "dz": 15.0,
        "angle": 30.0,
        "radius": 8.0,
        "desc": "High rise (dz=15A, ang=30, pitch=180A)",
    },
    {
        "name": "run_dz12_ang45",
        "dz": 12.0,
        "angle": 45.0,
        "radius": 8.0,
        "desc": "Reduced pitch (dz=12A, ang=45, pitch=96A)",
    },
    {
        "name": "run_dz15_ang20",
        "dz": 15.0,
        "angle": 20.0,
        "radius": 7.5,
        "desc": "Steep axial pitch (dz=15A, ang=20, pitch=270A, r=7.5A)",
    },
]


def run_condition(cfg: dict) -> Path:
    out_dir = Path("test_helical_runs") / cfg["name"]
    cif_target = out_dir / "intermediate_designs" / "material_spec.cif"

    if cif_target.exists():
        print(f"[{cfg['name']}] Found existing design output at {cif_target}")
        return cif_target

    print(f"\n=======================================================")
    print(f"Launching {cfg['name']}: {cfg['desc']}")
    print(f"=======================================================")

    cmd = [
        sys.executable,
        "material_builder.py",
        "--topology", "helical",
        "--copies", "8",
        "--length", "20",
        "--secondary_structure", "H",
        "--target_radius", str(cfg["radius"]),
        "--target_dz", str(cfg["dz"]),
        "--target_angle", str(cfg["angle"]),
        "--num_designs", "1",
        "--steps", "design",
        "--save_traj",
        "--save_x0_traj",
        "--traj_stride", "25",
        "--traj_align", "none",
        "--output_dir", str(out_dir),
        "--run",
    ]

    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)
    return cif_target


def main() -> None:
    results = []

    # Include existing dz12_ang30
    dz12_cif = Path("test_helical_runs/run_dz12_ang30/intermediate_designs/material_spec.cif")
    if dz12_cif.exists():
        res12 = analyze_structure_helical_alignment(dz12_cif)
        results.append({
            "name": "run_dz12_ang30",
            "desc": "Increased rise (dz=12A, ang=30, pitch=144A)",
            "cif": str(dz12_cif),
            **res12,
        })

    for cfg in CONFIGS:
        cif_path = run_condition(cfg)
        res = analyze_structure_helical_alignment(cif_path)
        results.append({
            "name": cfg["name"],
            "desc": cfg["desc"],
            "cif": str(cif_path),
            **res,
        })

    print("\n" + "=" * 115)
    print("FINAL SUMMARY: HELICAL ALIGNMENT ALONG ASSEMBLY LONG AXIS")
    print("=" * 115)
    header = f"{'Configuration':<20} | {'Description':<48} | {'Mean Tilt (deg)':<15} | {'Std':<6} | {'Min-Max':<12}"
    print(header)
    print("-" * 115)
    for r in results:
        min_max = f"{r['min_tilt_deg']:.1f}-{r['max_tilt_deg']:.1f}"
        print(f"{r['name']:<20} | {r['desc']:<48} | {r['mean_tilt_deg']:<15.2f} | {r['std_tilt_deg']:<6.2f} | {min_max:<12}")
    print("=" * 115)


if __name__ == "__main__":
    main()
