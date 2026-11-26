import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_material_screen.ipynb"


@pytest.mark.parametrize(
    ("cuda_available", "capability", "precision", "expected"),
    [
        (True, (7, 5), "bf16-mixed", "32-true"),
        (True, (7, 5), "bf16", "32-true"),
        (True, (8, 9), "bf16-mixed", "bf16-mixed"),
        (True, (7, 5), "32", "32"),
        (False, (0, 0), "bf16-mixed", "bf16-mixed"),
    ],
)
def test_resolve_trainer_kwargs_t4_precision_fallback(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    capability: tuple[int, int],
    precision: str,
    expected: str,
) -> None:
    """Verify bf16 precision downgrades to 32-true only on CUDA CC < 8.0."""
    import torch
    from boltzgen.task.predict.predict import _resolve_trainer_kwargs

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    base = {"accelerator": "gpu", "devices": 1, "precision": precision}
    resolved = _resolve_trainer_kwargs(base)
    assert resolved["precision"] == expected
    assert base["precision"] == precision


def test_pipeline_init_cpu_only_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify BinderDesignPipeline initializes cleanly without CUDA."""
    import torch
    import boltzgen.cli.boltzgen as boltzgen_cli

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _raise_if_called(*_args: object, **_kwargs: object) -> tuple[int, int]:
        raise AssertionError("CUDA device capability should not be queried on CPU")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _raise_if_called)
    monkeypatch.setattr(
        boltzgen_cli,
        "get_artifact_path",
        lambda *_args, **_kwargs: Path("dummy_ckpt"),
    )

    parser = boltzgen_cli.build_parser()
    args = parser.parse_args(["run", "dummy.yaml", "--num_designs", "4"])
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_ckpt"]

    pipeline = boltzgen_cli.BinderDesignPipeline(args, Path("dummy_moldir"))
    design_step = next(s for s in pipeline.steps if s.name == "design")
    assert "override.use_kernels=False" in design_step.args
    assert "trainer.devices=1" in design_step.args


def test_counter_screen_config_overrides_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --config overrides propagate to counter-screen pipeline steps."""
    import torch
    import boltzgen.cli.boltzgen as boltzgen_cli

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        boltzgen_cli,
        "get_artifact_path",
        lambda *_args, **_kwargs: Path("dummy_ckpt"),
    )

    parser = boltzgen_cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "dummy.yaml",
            "--ligand_counter_screen",
            "--num_designs",
            "4",
            "--config",
            "counter_screen_folding",
            "diffusion_samples=2",
            "--config",
            "filter_target_candidates",
            "max_candidates=7",
        ]
    )
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_ckpt"]

    pipeline = boltzgen_cli.BinderDesignPipeline(args, Path("dummy_moldir"))
    cs_fold = next(s for s in pipeline.steps if s.name == "counter_screen_folding")
    cand_filter = next(s for s in pipeline.steps if s.name == "filter_target_candidates")
    assert "diffusion_samples=2" in cs_fold.args
    assert "max_candidates=7" in cand_filter.args


def test_material_builder_forwards_colab_quota_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify material_builder.py forwards --reuse, --cache, --budget, checkpoints, workers, kernels."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import material_builder

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "copies": 6,
                "topology": "cyclic",
                "num_designs": 2,
                "design_checkpoints": "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
                "reuse": True,
                "budget": 12,
                "num_workers": 2,
                "use_kernels": "false",
                "cache": str(tmp_path / "hf_cache"),
            }
        )
    )

    captured_cmds: list[list[str]] = []

    def _mock_run(cmd: list[str], check: bool = True) -> None:  # noqa: ARG001
        captured_cmds.append(list(cmd))

    monkeypatch.setattr(material_builder.subprocess, "run", _mock_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "material_builder.py",
            "--config",
            str(cfg_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--run",
        ],
    )
    material_builder.main()

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert "--reuse" in cmd
    assert cmd[cmd.index("--cache") + 1] == str(tmp_path / "hf_cache")
    assert cmd[cmd.index("--budget") + 1] == "12"
    assert cmd[cmd.index("--num_workers") + 1] == "2"
    assert cmd[cmd.index("--use_kernels") + 1] == "false"
    ckpt_idx = cmd.index("--design_checkpoints")
    assert cmd[ckpt_idx + 1] == "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt"


def test_colab_notebook_structure_syntax_and_dry_run(tmp_path: Path) -> None:
    """Validate Colab notebook JSON, AST syntax of all cells, and cyclic Zn 6-mer dry run."""
    assert NOTEBOOK_PATH.exists()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 9

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 9

    for idx, cell in enumerate(code_cells):
        src_lines = [
            line
            for line in cell["source"]
            if not line.lstrip().startswith(("!", "%"))
        ]
        ast.parse("".join(src_lines), filename=f"colab_cell_{idx + 1}.py")

    # Execute function definitions from Cell 3 to verify config matches cyclic_AP_Zn_6mer.yaml
    cell3_src = "".join(code_cells[2]["source"])
    cell3_tree = ast.parse(cell3_src)
    func_nodes = [
        node
        for node in cell3_tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PRESET_TABLE"
        )
    ]
    ns: dict = {"Path": Path, "subprocess": subprocess, "sys": sys, "yaml": yaml}
    from typing import Any
    ns["Any"] = Any
    exec(compile(ast.Module(body=func_nodes, type_ignores=[]), "cell3_helpers", "exec"), ns)

    cfg = ns["build_cyclic_material_config"](
        copies=6,
        target_radius=9.0,
        spacing_noise=1.0,
        antiparallel_prob=1.0,
        peptide_length=20,
        secondary_structure="H",
        ligand_ccd="ZN",
        ligand_placement_mode="interface_anchored",
        ligands_per_peptide=0.5,
        anchor_residue="random",
        anchor_distance=3.5,
        anchor_direction="interface",
        axial_count=2,
        axial_z_min=-6.0,
        axial_z_max=6.0,
        min_ligand_distance=3.5,
        ligand_counter_screen=True,
        num_designs=10,
        seqs_per_backbone=3,
        max_target_candidates=10,
        inverse_temp=0.15,
        backbone_noise=0.1,
        repetition_penalty=0.5,
        avoid_aa="M",
        aa_bias="",
    )
    assert cfg["copies"] == 6
    assert cfg["topology"] == "cyclic"
    assert cfg["ligand_counter_screen"] is True
    assert len(cfg["asym_unit"]) == 2
    assert cfg["asym_unit"][1]["ccd"] == "ZN"

    cfg_yaml = tmp_path / "cyclic_scout.yaml"
    cfg_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out_dir = tmp_path / "dry_out"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "material_builder.py"),
            "--config",
            str(cfg_yaml),
            "--output_dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Generated design spec" in proc.stdout
    spec = yaml.safe_load((out_dir / "material_spec.yaml").read_text())
    entities = spec["entities"]
    proteins = [e for e in entities if "protein" in e]
    ligands = [e for e in entities if "ligand" in e]
    assert len(proteins) == 6
    assert len(ligands) == 3
    assert all(lig["ligand"]["ccd"] == "ZN" for lig in ligands)
