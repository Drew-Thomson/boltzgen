import json
import math
from pathlib import Path
import pytest
import torch

import boltzgen.cli.boltzgen as boltzgen_cli
from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.model.modules.materials import (
    enforce_minimum_axial_separation,
    resolve_protein_metal_clashes,
    sample_stratified_axial_positions,
)


def test_expand_material_spec_single_central_ligand() -> None:
    entities, layout = expand_material_spec(
        6,
        [
            {
                "type": "protein",
                "name": "peptide",
                "length": 20,
                "secondary_structure": "H",
            },
            {
                "type": "ligand",
                "ccd": "CU",
                "placement": "center",
            },
        ],
    )
    # 6 protein chains + 1 central ligand
    assert len(entities) == 7
    ligands = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligands) == 1
    lig = ligands[0]
    assert lig["attachment_scope"] == "axial"
    assert lig["constraint"] == "z_axis"
    assert lig["parent_asym_index"] is None
    assert lig["placement_index"] is None
    assert math.isclose(lig["axial_z"], 0.0, abs_tol=1e-5)


def test_expand_material_spec_dual_axial_ligands() -> None:
    entities, layout = expand_material_spec(
        6,
        [
            {
                "type": "protein",
                "name": "peptide",
                "length": 20,
                "secondary_structure": "H",
            },
            {
                "type": "ligand",
                "ccd": "CU",
                "placement": "z_axis",
                "constraint": "z_axis",
                "count": 2,
                "axial_z": 3.5,
            },
        ],
    )
    assert len(entities) == 8
    ligands = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligands) == 2
    assert ligands[0]["attachment_scope"] == "axial"
    assert ligands[1]["attachment_scope"] == "axial"
    assert math.isclose(ligands[0]["axial_z"], -3.5, abs_tol=1e-5)
    assert math.isclose(ligands[1]["axial_z"], 3.5, abs_tol=1e-5)


def test_expand_material_spec_explicit_z_offsets() -> None:
    entities, layout = expand_material_spec(
        6,
        [
            {"type": "protein", "name": "p", "length": 20},
            {
                "type": "ligand",
                "ccd": "CU",
                "placement": "axial",
                "axial_z": [-2.0, 4.0],
            },
        ],
    )
    assert len(entities) == 8
    ligands = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligands) == 2
    assert math.isclose(ligands[0]["axial_z"], -2.0, abs_tol=1e-5)
    assert math.isclose(ligands[1]["axial_z"], 4.0, abs_tol=1e-5)


def test_expand_material_spec_random_even_distribution() -> None:
    entities, layout = expand_material_spec(
        6,
        [
            {"type": "protein", "name": "p", "length": 20},
            {
                "type": "ligand",
                "ccd": "CU",
                "placement": "z_axis",
                "count": 3,
                "distribution": "random_even",
                "z_range": 6.0,
                "min_distance": 3.5,
            },
        ],
    )
    assert len(entities) == 9
    ligands = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligands) == 3
    for lig in ligands:
        assert lig["attachment_scope"] == "axial"
        assert lig["distribution"] == "random_even"
        assert lig["z_range"] == [-6.0, 6.0]
        assert lig["min_distance"] == 3.5


def test_expand_material_spec_axial_rejects_incompatible_options() -> None:
    with pytest.raises(ValueError, match="attach_to is not supported"):
        expand_material_spec(
            6,
            [
                {"type": "protein", "name": "p", "length": 20},
                {
                    "type": "ligand",
                    "ccd": "CU",
                    "placement": "z_axis",
                    "attach_to": "p",
                },
            ],
        )

    with pytest.raises(ValueError, match="cannot be combined with axial"):
        expand_material_spec(
            6,
            [
                {"type": "protein", "name": "p", "length": 20},
                {
                    "type": "ligand",
                    "ccd": "CU",
                    "placement": "z_axis",
                    "ligands_per_peptide": 1,
                },
            ],
        )


def test_resolve_protein_metal_clashes_1d_axial_only() -> None:
    # Metal at origin (0, 0, 0)
    metal = torch.tensor([0.0, 0.0, 0.0])
    # Protein atom located radially at (1.0, 0.0, 0.5)
    protein = torch.tensor([[1.0, 0.0, 0.5]])
    z_axis = torch.tensor([0.0, 0.0, 1.0])

    resolved = resolve_protein_metal_clashes(
        metal,
        protein,
        min_distance=2.8,
        axis=z_axis,
    )

    # Must remain strictly on the z-axis: x = 0, y = 0
    assert abs(resolved[0].item()) < 1e-6
    assert abs(resolved[1].item()) < 1e-6
    dist = torch.norm(resolved - protein[0]).item()
    assert dist >= 2.799


def test_sample_stratified_axial_positions() -> None:
    gen = torch.Generator().manual_seed(42)
    positions = sample_stratified_axial_positions(
        count=3,
        z_range=(-6.0, 6.0),
        min_distance=3.0,
        generator=gen,
    )
    assert len(positions) == 3
    assert positions[0] < positions[1] < positions[2]
    assert all(-6.0 <= p <= 6.0 for p in positions)
    assert positions[1] - positions[0] >= 3.0
    assert positions[2] - positions[1] >= 3.0


def test_sample_stratified_axial_positions_symmetric() -> None:
    gen = torch.Generator().manual_seed(42)
    positions = sample_stratified_axial_positions(
        count=2,
        z_range=(-5.0, 5.0),
        min_distance=3.5,
        symmetric=True,
        generator=gen,
    )
    assert len(positions) == 2
    assert math.isclose(positions[0], -positions[1], abs_tol=1e-5)
    assert positions[1] - positions[0] >= 3.5


def test_enforce_minimum_axial_separation() -> None:
    # Two ligands very close together at z = -0.5 and z = 0.5 (separation = 1.0 < 3.5)
    positions = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    axis = torch.tensor([0.0, 0.0, 1.0])

    adjusted = enforce_minimum_axial_separation(positions, axis=axis, min_distance=3.5)
    assert adjusted.shape == (2, 3)
    # x and y should remain zero
    assert torch.allclose(adjusted[:, :2], torch.zeros(2, 2), atol=1e-6)
    # z separation should now be >= 3.5
    sep = adjusted[1, 2] - adjusted[0, 2]
    assert sep.item() >= 3.499


def test_expand_material_spec_min_distances() -> None:
    entities, layout = expand_material_spec(
        6,
        [
            {"type": "protein", "name": "p", "length": 20},
            {
                "type": "ligand",
                "ccd": "CU",
                "placement": "z_axis",
                "count": 2,
                "distribution": "random_even",
                "z_range": 12.0,
                "min_distance": 3.5,
                "min_protein_distance": 2.2,
                "min_ligand_distance": 4.0,
            },
        ],
    )
    ligands = [r for r in layout["chains"] if r["role"] == "ligand"]
    assert len(ligands) == 2
    for lig in ligands:
        assert lig["min_protein_distance"] == 2.2
        assert lig["min_ligand_distance"] == 4.0
        assert lig["min_distance"] == 4.0


def test_pipeline_min_ligand_distance_filtering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        boltzgen_cli,
        "get_artifact_path",
        lambda *_args, **_kwargs: Path("dummy_checkpoint"),
    )

    parser = boltzgen_cli.build_parser()

    # Case 1: --min_ligand_distance passed via CLI
    args = parser.parse_args(["run", "dummy.yaml", "--min_ligand_distance", "3.5"])
    args.output = tmp_path / "out1"
    args.config_dir = Path("test_config")
    args.design_spec = [tmp_path / "dummy.yaml"]
    args.design_checkpoints = ["dummy_checkpoint"]

    pipeline = boltzgen_cli.BinderDesignPipeline(args, Path("dummy_moldir"))
    filter_step = next(s for s in pipeline.steps if s.name == "filtering")
    assert any("min_ligand_distance" in arg for arg in filter_step.args)
    assert any("3.5" in arg for arg in filter_step.args)

    # Case 2: Inferred from material_layout.json next to design_spec
    spec_dir = tmp_path / "spec_dir"
    spec_dir.mkdir()
    spec_file = spec_dir / "design.yaml"
    spec_file.write_text("dummy")
    layout_file = spec_dir / "material_layout.json"
    layout_file.write_text(json.dumps({"ligands": [{"min_ligand_distance": 4.2}]}))

    args2 = parser.parse_args(["run", str(spec_file)])
    args2.output = tmp_path / "out2"
    args2.config_dir = Path("test_config")
    args2.design_spec = [spec_file]
    args2.design_checkpoints = ["dummy_checkpoint"]

    pipeline2 = boltzgen_cli.BinderDesignPipeline(args2, Path("dummy_moldir"))
    filter_step2 = next(s for s in pipeline2.steps if s.name == "filtering")
    assert any("min_ligand_distance" in arg for arg in filter_step2.args)
    assert any("4.2" in arg for arg in filter_step2.args)


