"""CPU tests for material analysis metrics and ranking presets."""

# ruff: noqa: INP001

from __future__ import annotations

import inspect
import math

import pytest
import torch

from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.analyze.material_metrics import (
    backbone_ca_coords,
    cage_metrics,
    compute_material_metrics,
    pore_aspect_ratio,
    sheet_planarity,
    void_radius,
)
from boltzgen.task.analyze.analyze_utils import heteromer_counter_screen_contrasts
from boltzgen.task.filter.material_presets import (
    format_metrics_override,
    heteromer_filtering_overrides,
    material_filtering_overrides,
)


def test_heteromer_counter_screen_contrasts_use_strongest_homomer():
    result = heteromer_counter_screen_contrasts(
        {
            "complex_plddt": 82.0,
            "design_iptm": 0.81,
            "protein_iptm": 0.78,
            "min_interaction_pae": 3.0,
        },
        {
            "complex_plddt": 55.0,
            "design_iptm": 0.25,
            "protein_iptm": 0.31,
            "min_interaction_pae": 12.0,
        },
        {
            "complex_plddt": 61.0,
            "design_iptm": 0.30,
            "protein_iptm": 0.35,
            "min_interaction_pae": 9.0,
        },
    )
    assert result["delta_complex_plddt_vs_homomer_max"] == pytest.approx(21.0)
    assert result["delta_design_iptm_vs_homomer_max"] == pytest.approx(0.51)
    assert result["delta_protein_iptm_vs_homomer_max"] == pytest.approx(0.43)
    assert result["delta_neg_min_interaction_pae_vs_homomer_max"] == pytest.approx(6.0)


def test_heteromer_filter_preset_prioritizes_contrasts():
    assert heteromer_filtering_overrides() == {
        "delta_design_iptm_vs_homomer_max": 1,
        "delta_complex_plddt_vs_homomer_max": 1,
        "delta_protein_iptm_vs_homomer_max": 2,
    }


def test_cage_sphericity_cube_and_perturbation() -> None:
    cube = torch.tensor(
        [
            [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    atom_coords = torch.tensor([[3.7, 0.0, 0.0]], dtype=torch.float64)
    result = cage_metrics(cube, atom_coords)
    assert result["cage_sphericity_rmsd"] == pytest.approx(0.0)
    assert result["cage_void_radius"] == pytest.approx(2.0)

    perturbed = cube.clone()
    perturbed[0, 0] -= 1.0
    perturbed_radii = torch.linalg.vector_norm(
        perturbed - perturbed.mean(dim=0), dim=1
    )
    perturbed_result = cage_metrics(perturbed, atom_coords)
    assert perturbed_result["cage_sphericity_rmsd"] == pytest.approx(
        perturbed_radii.std(correction=0).item()
    )
    assert perturbed_result["cage_sphericity_rmsd"] > 0


def test_cage_metrics_validate_geometry_inputs() -> None:
    with pytest.raises(ValueError, match="at least two chain COMs"):
        cage_metrics(torch.zeros((1, 3)), torch.zeros((1, 3)))
    invalid_coms = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="finite"):
        cage_metrics(invalid_coms, torch.zeros((1, 3)))


def test_void_radius_supports_non_cage_hollow_geometries() -> None:
    coms = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    atoms = torch.tensor([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]])
    assert void_radius(coms, atoms) == pytest.approx(3.3)


def test_sheet_planarity_for_flat_isotropic_and_buckled_points() -> None:
    flat = torch.tensor(
        [[x, y, 0.0] for x in range(3) for y in range(3)], dtype=torch.float64
    )
    cube = torch.tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=torch.float64,
    )
    buckled = flat.clone()
    buckled[:, 2] = 0.5 * torch.sin(flat[:, 0] * math.pi / 2)
    assert sheet_planarity(flat) == pytest.approx(1.0)
    assert sheet_planarity(cube) == pytest.approx(0.0)
    assert 0.0 < sheet_planarity(buckled) < sheet_planarity(flat)
    assert sheet_planarity(torch.zeros((2, 3))) == 0.0


def test_pore_aspect_ratio_uses_interior_delaunay_cells() -> None:
    # A 3x3 lattice has four bounded Delaunay triangles around its center.
    # Synthetic atom coordinates give equal clearance from each candidate.
    grid = torch.tensor(
        [[x, y] for x in range(3) for y in range(3)], dtype=torch.float64
    )
    atoms = torch.tensor(
        [[x + 0.5, y + 0.5] for x in range(2) for y in range(2)],
        dtype=torch.float64,
    )
    ratio = pore_aspect_ratio(grid, atoms, vdw_radius=0.0)
    assert ratio is not None
    assert ratio >= 1.0


@pytest.mark.parametrize(
    ("coms", "atoms"),
    [
        (torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), torch.ones((2, 2))),
        (
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            torch.ones((2, 2)),
        ),
        (torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), torch.empty((0, 2))),
    ],
)
def test_pore_aspect_ratio_returns_none_for_unavailable_cases(
    coms: torch.Tensor, atoms: torch.Tensor
) -> None:
    assert pore_aspect_ratio(coms, atoms) is None


def test_backbone_ca_extraction_preserves_chain_boundaries() -> None:
    coords = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    chain_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    mask = torch.ones(8, dtype=torch.bool)
    actual = backbone_ca_coords(coords, chain_ids, mask)
    torch.testing.assert_close(actual, coords[[1, 5]])


def test_material_topology_dispatch_and_negative_rank_columns() -> None:
    coms = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
        dtype=torch.float64,
    )
    atoms = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64)
    ca = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cage = compute_material_metrics("cage_tetrahedral", coms, atoms, ca)
    assert "cage_void_radius" in cage
    assert "neg_cage_sphericity_rmsd" in cage
    assert compute_material_metrics("floating", coms, atoms, ca) == {}
    assert "cage_void_radius" in compute_material_metrics(
        "nanotube", coms, atoms, ca
    )


def test_filtering_presets_orient_metrics_and_exclude_cage_target_metrics() -> None:
    cage = material_filtering_overrides("cage_octahedral")
    assert cage is not None
    assert cage["design_to_target_iptm"] is None
    assert cage["neg_min_design_to_target_pae"] is None
    assert "neg_cage_sphericity_rmsd" in cage
    assert "cage_void_radius" in cage

    mesh = material_filtering_overrides("hexagonal_mesh")
    assert mesh == {
        "sheet_planarity": 1,
        "neg_pore_aspect_ratio": 1,
        "h_bonds_per_interface_refolded": 1,
    }
    assert material_filtering_overrides("floating") is None
    assert format_metrics_override("cage_tetrahedral") == (
        "metrics_override={design_to_target_iptm: null, "
        "neg_min_design_to_target_pae: null, neg_cage_sphericity_rmsd: 1, "
        "cage_void_radius: 1, neg_lattice_rmsd_refolded: 1}"
    )
    assert format_metrics_override("cyclic") is None


def test_analyze_material_metric_toggle_defaults_on() -> None:
    parameter = inspect.signature(Analyze.__init__).parameters["material_metrics"]
    assert parameter.default is True
