import argparse
import sys
from pathlib import Path


def test_diffusion_batch_size_threshold_dropped(monkeypatch):
    from boltzgen.cli.boltzgen import build_parser, BinderDesignPipeline

    parser = build_parser()

    # Test 1: num_designs = 50
    args = parser.parse_args(["run", "dummy.yaml", "--num_designs", "50"])

    import boltzgen.cli.boltzgen as boltzgen_cli

    def mock_get_artifact_path(*args, **kwargs):
        return Path("dummy_checkpoint")

    monkeypatch.setattr(boltzgen_cli, "get_artifact_path", mock_get_artifact_path)

    # Setting some required paths that pipeline init expects
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_checkpoint"]

    p1 = BinderDesignPipeline(args, Path("dummy_moldir"))
    # Pipeline step 0 should be design
    design_step = next(s for s in p1.steps if s.name == "design")
    assert "diffusion_samples=1" in design_step.args
    assert "data.cfg.multiplicity=50" in design_step.args

    # Test 2: num_designs = 150
    args = parser.parse_args(["run", "dummy.yaml", "--num_designs", "150"])
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_checkpoint"]

    p2 = BinderDesignPipeline(args, Path("dummy_moldir"))
    design_step2 = next(s for s in p2.steps if s.name == "design")
    assert "diffusion_samples=1" in design_step2.args
    assert "data.cfg.multiplicity=150" in design_step2.args

    # Test 3: explicit diffusion_batch_size
    args = parser.parse_args(
        ["run", "dummy.yaml", "--num_designs", "150", "--diffusion_batch_size", "5"]
    )
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_checkpoint"]

    p3 = BinderDesignPipeline(args, Path("dummy_moldir"))
    design_step3 = next(s for s in p3.steps if s.name == "design")
    assert "diffusion_samples=5" in design_step3.args
    assert "data.cfg.multiplicity=30" in design_step3.args


def test_material_builder_cmd_args(monkeypatch):
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import material_builder

    test_args = [
        "material_builder.py",
        "--topology",
        "linear_tape",
        "--num_designs",
        "100",
        "--seqs_per_backbone",
        "2",
        "--run",
    ]

    cmd_args = []

    def mock_run(cmd, check=True):
        cmd_args.extend(cmd)

    monkeypatch.setattr(material_builder.subprocess, "run", mock_run)
    monkeypatch.setattr(sys, "argv", test_args)

    material_builder.main()

    # We should have the diffusion batch size and inverse_fold args in the command sent to boltzgen
    assert "--num_designs" in cmd_args
    assert cmd_args[cmd_args.index("--num_designs") + 1] == "100"

    assert "--diffusion_batch_size" in cmd_args
    assert cmd_args[cmd_args.index("--diffusion_batch_size") + 1] == "1"

    assert "--inverse_fold_num_sequences" in cmd_args
    assert cmd_args[cmd_args.index("--inverse_fold_num_sequences") + 1] == "2"


def test_boltzgen_cli_underscore_flags_and_aliases():
    from boltzgen.cli.boltzgen import build_parser

    parser = build_parser()

    # Test underscore versions
    args_underscore = parser.parse_args(
        [
            "run",
            "dummy.yaml",
            "--heteromer_counter_screen",
            "--max_partner_identity",
            "0.4",
            "--max_surviving_designs",
            "12",
            "--target_max_rmsd",
            "3.0",
            "--target_min_plddt",
            "80.0",
            "--anti_correlation_strength",
            "1.5",
            "--double_tape_core_bias",
            "0.8",
            "--double_tape_aromatic_bias",
            "0.9",
        ]
    )
    assert args_underscore.heteromer_counter_screen is True
    assert args_underscore.max_partner_identity == 0.4
    assert args_underscore.max_surviving_designs == 12
    assert args_underscore.target_max_rmsd == 3.0
    assert args_underscore.target_min_plddt == 80.0
    assert args_underscore.anti_correlation_strength == 1.5
    assert args_underscore.double_tape_core_bias == 0.8
    assert args_underscore.double_tape_aromatic_bias == 0.9

    # Test hyphenated alias versions
    args_alias = parser.parse_args(
        [
            "run",
            "dummy.yaml",
            "--heteromer-counter-screen",
            "--max-partner-identity",
            "0.4",
            "--max-surviving-designs",
            "12",
            "--target-max-rmsd",
            "3.0",
            "--target-min-plddt",
            "80.0",
            "--anti-correlation-strength",
            "1.5",
            "--double-tape-core-bias",
            "0.8",
            "--double-tape-aromatic-bias",
            "0.9",
        ]
    )
    assert args_alias.heteromer_counter_screen is True
    assert args_alias.max_partner_identity == 0.4
    assert args_alias.max_surviving_designs == 12
    assert args_alias.target_max_rmsd == 3.0
    assert args_alias.target_min_plddt == 80.0
    assert args_alias.anti_correlation_strength == 1.5
    assert args_alias.double_tape_core_bias == 0.8
    assert args_alias.double_tape_aromatic_bias == 0.9


def test_material_builder_underscore_and_alias_args(monkeypatch, tmp_path):
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import material_builder
    import yaml

    # Prepare a minimal heteromer YAML config with two chains
    config_file = tmp_path / "heteromer.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "asym_unit": [
                    {"type": "protein", "length": 10},
                    {"type": "protein", "length": 10},
                ],
                "copies": 4,
                "topology": "linear_tape",
            }
        )
    )

    # 1. Test running material_builder with underscore flags
    cmd_args_underscore = []
    monkeypatch.setattr(
        material_builder.subprocess,
        "run",
        lambda cmd, check=True: cmd_args_underscore.extend(cmd),
    )
    test_args_underscore = [
        "material_builder.py",
        "--config",
        str(config_file),
        "--heteromer_counter_screen",
        "--heteromer_charge_bias",
        "0.7",
        "--max_partner_identity",
        "0.35",
        "--max_surviving_designs",
        "8",
        "--anti_correlation_strength",
        "2.2",
        "--double_tape_core_bias",
        "0.6",
        "--double_tape_aromatic_bias",
        "0.7",
        "--run",
    ]
    monkeypatch.setattr(sys, "argv", test_args_underscore)
    material_builder.main()

    assert "--heteromer_counter_screen" in cmd_args_underscore
    assert "--heteromer-counter-screen" not in cmd_args_underscore
    assert "--max_partner_identity" in cmd_args_underscore
    assert (
        cmd_args_underscore[cmd_args_underscore.index("--max_partner_identity") + 1]
        == "0.35"
    )
    assert "--max_surviving_designs" in cmd_args_underscore
    assert (
        cmd_args_underscore[cmd_args_underscore.index("--max_surviving_designs") + 1]
        == "8"
    )
    assert "--anti_correlation_strength" in cmd_args_underscore
    assert (
        cmd_args_underscore[
            cmd_args_underscore.index("--anti_correlation_strength") + 1
        ]
        == "2.2"
    )
    assert "--double_tape_core_bias" in cmd_args_underscore
    assert (
        cmd_args_underscore[cmd_args_underscore.index("--double_tape_core_bias") + 1]
        == "0.6"
    )
    assert "--double_tape_aromatic_bias" in cmd_args_underscore
    assert (
        cmd_args_underscore[
            cmd_args_underscore.index("--double_tape_aromatic_bias") + 1
        ]
        == "0.7"
    )

    # 2. Test running material_builder with hyphenated alias flags
    cmd_args_alias = []
    monkeypatch.setattr(
        material_builder.subprocess,
        "run",
        lambda cmd, check=True: cmd_args_alias.extend(cmd),
    )
    test_args_alias = [
        "material_builder.py",
        "--config",
        str(config_file),
        "--heteromer-counter-screen",
        "--heteromer-charge-bias",
        "0.7",
        "--max-partner-identity",
        "0.35",
        "--max-surviving-designs",
        "8",
        "--anti-correlation-strength",
        "2.2",
        "--double-tape-core-bias",
        "0.6",
        "--double-tape-aromatic-bias",
        "0.7",
        "--run",
    ]
    monkeypatch.setattr(sys, "argv", test_args_alias)
    material_builder.main()

    # Even when passed with hyphens, downstream boltzgen command should receive underscore flags
    assert "--heteromer_counter_screen" in cmd_args_alias
    assert "--heteromer-counter-screen" not in cmd_args_alias
    assert "--max_partner_identity" in cmd_args_alias
    assert cmd_args_alias[cmd_args_alias.index("--max_partner_identity") + 1] == "0.35"
    assert "--max_surviving_designs" in cmd_args_alias
    assert cmd_args_alias[cmd_args_alias.index("--max_surviving_designs") + 1] == "8"
    assert "--anti_correlation_strength" in cmd_args_alias
    assert (
        cmd_args_alias[cmd_args_alias.index("--anti_correlation_strength") + 1] == "2.2"
    )
    assert "--double_tape_core_bias" in cmd_args_alias
    assert cmd_args_alias[cmd_args_alias.index("--double_tape_core_bias") + 1] == "0.6"
    assert "--double_tape_aromatic_bias" in cmd_args_alias
    assert (
        cmd_args_alias[cmd_args_alias.index("--double_tape_aromatic_bias") + 1] == "0.7"
    )
