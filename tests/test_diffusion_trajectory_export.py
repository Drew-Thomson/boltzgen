# Copyright (c) 2024 BoltzGen Contributors
# Distributed under the MIT license.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch
from scripts.plot_diffusion_points_figure import (
    kabsch_rigid_align,
    orient_coords_along_z,
    plot_point_cloud_comparison,
    select_coalescence_frames,
)

from boltzgen.cli import boltzgen as boltzgen_cli
from boltzgen.cli.boltzgen import BinderDesignPipeline, build_parser
from boltzgen.task.predict.writer import DesignWriter

if TYPE_CHECKING:
    import pytest

MAX_TOLERANCE = 1e-6


def test_cli_trajectory_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify CLI trajectory flags are passed to design step."""
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "dummy.yaml",
        "--num_designs",
        "1",
        "--save_traj",
        "--save_x0_traj",
        "--traj_stride",
        "25",
        "--traj_align",
        "none",
    ])

    def mock_get_artifact_path(*_args: object, **_kwargs: object) -> Path:
        return Path("dummy_checkpoint")

    monkeypatch.setattr(boltzgen_cli, "get_artifact_path", mock_get_artifact_path)

    args.output = tmp_path / "test_out"
    args.config_dir = tmp_path / "test_config"
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_checkpoint"]

    pipeline = BinderDesignPipeline(args, Path("dummy_moldir"))
    design_step = next(s for s in pipeline.steps if s.name == "design")

    assert "writer.save_traj=True" in design_step.args
    assert "writer.save_x0_traj=True" in design_step.args
    assert "writer.traj_stride=25" in design_step.args
    assert "writer.traj_align=none" in design_step.args


def test_design_writer_npz_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify DesignWriter saves trajectory archives with valid keys."""
    writer = DesignWriter(
        output_dir=str(tmp_path),
        res_atoms_only=False,
        save_traj=True,
        save_x0_traj=True,
        traj_stride=2,
        traj_align="none",
        save_traj_npz=True,
        write_native=False,
    )

    num_atoms = 10
    num_tokens = 4
    num_steps = 5

    monkeypatch.setattr(
        "boltzgen.task.predict.writer.BoltzMasker.__call__",
        lambda _self, _b: {
            "ref_element": torch.zeros(1),
            "ref_atom_name_chars": torch.zeros(1),
        },
    )

    class MockStructure:
        def __init__(self) -> None:
            self.atoms: Dict[str, Any] = {}
            self.residues: List[Any] = []
            self.coords = np.zeros((num_atoms, 3))

    monkeypatch.setattr(
        "boltzgen.task.predict.writer.Structure.from_feat",
        lambda *_args, **_kwargs: (MockStructure(), None, None),
    )
    monkeypatch.setattr(
        "boltzgen.task.predict.writer.res_from_atom14",
        lambda s: s,
    )
    monkeypatch.setattr(
        "boltzgen.task.predict.writer.to_mmcif",
        lambda *_args, **_kwargs: "dummy_cif",
    )
    monkeypatch.setattr(
        "boltzgen.task.predict.writer.to_pdb",
        lambda *_args, **_kwargs: "dummy_pdb",
    )

    atom_to_token = torch.zeros((num_atoms, num_tokens), dtype=torch.float32)
    for a in range(num_atoms):
        atom_to_token[a, min(a // 3, num_tokens - 1)] = 1.0

    asym_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    atom_pad_mask = torch.ones(num_atoms, dtype=torch.bool)
    coords_final = torch.randn(1, num_atoms, 3)

    coords_traj = [torch.randn(1, num_atoms, 3) for _ in range(num_steps + 1)]
    x0_coords_traj = [torch.randn(1, num_atoms, 3) for _ in range(num_steps)]

    batch = {
        "id": ["test_sample"],
        "coords": torch.zeros(1, 1, num_atoms, 3),
        "design_mask": torch.tensor([[True, True, False, False]]),
        "binding_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "mol_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "ss_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "token_pad_mask": torch.ones((1, num_tokens), dtype=torch.bool),
        "token_resolved_mask": torch.ones((1, num_tokens), dtype=torch.bool),
        "atom_pad_mask": atom_pad_mask.unsqueeze(0),
        "atom_to_token": atom_to_token.unsqueeze(0),
        "asym_id": asym_id.unsqueeze(0),
        "extra_mols": None,
    }

    prediction = {
        "coords": coords_final,
        "coords_traj": coords_traj,
        "x0_coords_traj": x0_coords_traj,
        "exception": False,
        "atom_to_token": atom_to_token.unsqueeze(0),
        "asym_id": asym_id.unsqueeze(0),
        "atom_pad_mask": atom_pad_mask.unsqueeze(0),
        "token_pad_mask": torch.ones((1, num_tokens), dtype=torch.bool),
        "token_resolved_mask": torch.ones((1, num_tokens), dtype=torch.bool),
        "mol_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "ss_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "binding_type": torch.zeros((1, num_tokens), dtype=torch.long),
        "design_mask": batch["design_mask"],
        "chain_design_mask": batch["design_mask"],
        "token_to_res": torch.arange(num_tokens).unsqueeze(0),
    }

    writer.write_on_batch_end(
        trainer=None,
        pl_module=None,
        prediction=prediction,
        batch=batch,
        batch_indices=[0],
        sample_id="sample0",
    )

    npz_path = tmp_path / "sample0_0_diffusion_traj.npz"
    assert npz_path.exists(), f"NPZ file not generated at {npz_path}"

    with np.load(npz_path) as data:
        assert "coords" in data
        assert "x0_coords" in data
        assert "is_target" in data
        assert "asym_id" in data
        assert "atom_pad_mask" in data
        assert "step_indices" in data

        # 6 coords_traj steps with stride 2 -> [0, 2, 4, 5]
        assert list(data["step_indices"]) == [0, 2, 4, 5]
        # 5 x0_coords_traj steps with stride 2 -> [0, 2, 4]
        assert list(data["x0_step_indices"]) == [0, 2, 4]
        assert data["coords"].shape == (4, num_atoms, 3)
        assert data["x0_coords"].shape == (3, num_atoms, 3)
        assert data["is_target"].shape == (num_atoms,)
        assert data["asym_id"].shape == (num_atoms,)


def test_kabsch_rigid_align() -> None:
    """Verify exact Kabsch alignment on rotated and translated coordinates."""
    rng = np.random.default_rng(42)
    source = rng.normal(size=(20, 3))

    theta = np.pi / 4
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    trans = np.array([5.0, -3.0, 2.0])
    target = source @ rot_z.T + trans

    aligned = kabsch_rigid_align(source, target)
    rmsd = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=-1)))
    assert rmsd < MAX_TOLERANCE, f"Kabsch RMSD too high: {rmsd}"


def test_orient_coords_along_z() -> None:
    """Verify principal component orientation aligns major axis along Z."""
    rng = np.random.default_rng(42)
    pts = np.zeros((100, 3))
    pts[:, 0] = np.linspace(-50.0, 50.0, 100)
    pts[:, 1] = rng.normal(scale=1.0, size=100)
    pts[:, 2] = rng.normal(scale=1.0, size=100)

    oriented, _ = orient_coords_along_z(pts)
    std_x = np.std(oriented[:, 0])
    std_y = np.std(oriented[:, 1])
    std_z = np.std(oriented[:, 2])

    assert std_z > std_x * 5.0
    assert std_z > std_y * 5.0


def test_plot_point_cloud_comparison(tmp_path: Path) -> None:
    """Verify figure plotting with synthetic binder and helical trajectories."""
    num_frames = 6
    num_atoms = 16

    b_coords = np.random.default_rng(42).normal(size=(num_frames, num_atoms, 3))
    b_is_target = np.array([True] * 8 + [False] * 8)
    b_asym = np.array([0] * 8 + [1] * 8)
    b_mask = np.ones(num_atoms, dtype=bool)

    binder_data = {
        "coords": b_coords,
        "is_target": b_is_target,
        "asym_id": b_asym,
        "atom_pad_mask": b_mask,
        "step_indices": np.arange(num_frames),
    }

    h_coords = np.random.default_rng(43).normal(size=(num_frames, num_atoms, 3))
    h_is_target = np.zeros(num_atoms, dtype=bool)
    h_asym = np.repeat(np.arange(4), 4)
    h_mask = np.ones(num_atoms, dtype=bool)

    helical_data = {
        "coords": h_coords,
        "is_target": h_is_target,
        "asym_id": h_asym,
        "atom_pad_mask": h_mask,
        "step_indices": np.arange(num_frames),
    }

    out_file = tmp_path / "comparison.png"
    plot_point_cloud_comparison(
        binder_traj=binder_data,
        helical_traj=helical_data,
        output_path=out_file,
        num_timepoints=5,
        dpi=100,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_select_coalescence_frames() -> None:
    """Verify automatic coalescence frame selection identifies collapsing stages."""
    rng = np.random.default_rng(42)
    num_frames = 20
    num_atoms = 30
    num_targets = 5
    final_coords = rng.normal(size=(num_atoms, 3)) * 10.0

    # Progressive noise decay from large noise down to converged structure
    sigmas = np.geomspace(500.0, 0.01, num_frames - 1)
    coords = [
        final_coords + rng.normal(scale=s, size=(num_atoms, 3))
        for s in sigmas
    ]
    coords.append(final_coords)
    coords_arr = np.stack(coords, axis=0)

    selected = select_coalescence_frames(coords_arr, num_timepoints=num_targets)
    assert len(selected) == num_targets
    assert selected == sorted(selected)
    assert selected[-1] == num_frames - 1
