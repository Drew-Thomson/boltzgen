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
    args = parser.parse_args(["run", "dummy.yaml", "--num_designs", "150", "--diffusion_batch_size", "5"])
    args.output = Path("test_out")
    args.config_dir = Path("test_config")
    args.design_spec = [Path("dummy.yaml")]
    args.design_checkpoints = ["dummy_checkpoint"]
    
    p3 = BinderDesignPipeline(args, Path("dummy_moldir"))
    design_step3 = next(s for s in p3.steps if s.name == "design")
    assert "diffusion_samples=5" in design_step3.args
    assert "data.cfg.multiplicity=30" in design_step3.args

def test_material_builder_cmd_args(monkeypatch):
    import material_builder
    import sys
    
    test_args = ["material_builder.py", "--topology", "linear_tape", "--num_designs", "100", "--seqs_per_backbone", "2", "--run"]
    
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
