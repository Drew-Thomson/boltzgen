"""CPU tests for material analysis metrics and ranking presets."""

# ruff: noqa: INP001, ANN201, PLR2004, D401

from __future__ import annotations

import inspect
import math

import pytest
import torch

from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.analyze.analyze_utils import heteromer_counter_screen_contrasts
from boltzgen.task.analyze.material_metrics import (
    backbone_ca_coords,
    cage_metrics,
    compute_double_tape_neatness_metrics,
    compute_material_metrics,
    compute_two_chain_alternating_metrics,
    pore_aspect_ratio,
    sheet_planarity,
    void_radius,
)
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


def test_two_chain_alternating_metrics_perfect_and_parity_flip() -> None:
    torch.manual_seed(42)
    chains = 4
    atoms_per_chain = 16
    pitch = 4.8

    base_chain = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.0, 1.2, 0.0],
            [1.5, 2.3, 0.0],
        ]
        * 4,
        dtype=torch.float32,
    )

    ideal_coords_list = []
    ideal_chain_ids_list = []
    partner_slots_ideal = {}

    for c in range(chains):
        chain_coords = base_chain.clone()
        chain_coords[:, 2] += c * pitch
        ideal_coords_list.append(chain_coords)
        ideal_chain_ids_list.append(torch.full((atoms_per_chain,), c, dtype=torch.long))
        partner_slots_ideal[c] = c % 2

    coords_ideal = torch.cat(ideal_coords_list)
    chain_ids_ideal = torch.cat(ideal_chain_ids_list)
    bb_mask = torch.ones(coords_ideal.shape[0], dtype=torch.bool)

    # 1. Perfect matching (A-B-A-B)
    res_perfect = compute_two_chain_alternating_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids_ideal,
        chain_ids_ideal=chain_ids_ideal,
        partner_slots_refold=partner_slots_ideal,
        partner_slots_ideal=partner_slots_ideal,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res_perfect["two_chain_alternation_ratio"] == pytest.approx(1.0)
    assert res_perfect["lattice_rmsd_geom"] == pytest.approx(0.0, abs=1e-5)
    assert res_perfect["lattice_rmsd_refolded"] == pytest.approx(0.0, abs=1e-5)

    # 2. Parity flipped / shifted (B-A-B-A)
    partner_slots_flipped = {c: (c + 1) % 2 for c in range(chains)}
    res_flipped = compute_two_chain_alternating_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids_ideal,
        chain_ids_ideal=chain_ids_ideal,
        partner_slots_refold=partner_slots_flipped,
        partner_slots_ideal=partner_slots_ideal,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res_flipped["two_chain_alternation_ratio"] == pytest.approx(1.0)
    assert res_flipped["lattice_rmsd_phase_1"] == pytest.approx(0.0, abs=1e-5)
    assert res_flipped["lattice_rmsd_geom"] == pytest.approx(0.0, abs=1e-5)
    assert res_flipped["lattice_rmsd_refolded"] == pytest.approx(0.0, abs=1e-5)


def test_two_chain_alternating_metrics_homomeric_defect_penalized() -> None:
    torch.manual_seed(42)
    chains = 4
    atoms_per_chain = 16
    pitch = 4.8
    base_chain = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.0, 1.2, 0.0],
            [1.5, 2.3, 0.0],
        ]
        * 4,
        dtype=torch.float32,
    )
    coords_list = [base_chain.clone() for _ in range(chains)]
    for c in range(chains):
        coords_list[c][:, 2] += c * pitch
    coords = torch.cat(coords_list)
    chain_ids = torch.cat(
        [torch.full((atoms_per_chain,), c, dtype=torch.long) for c in range(chains)]
    )
    bb_mask = torch.ones(coords.shape[0], dtype=torch.bool)

    partner_slots_ideal = {0: 0, 1: 1, 2: 0, 3: 1}  # A-B-A-B
    partner_slots_defective = {0: 0, 1: 0, 2: 1, 3: 1}  # A-A-B-B

    res = compute_two_chain_alternating_metrics(
        coords_refold=coords,
        coords_ideal=coords,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots_defective,
        partner_slots_ideal=partner_slots_ideal,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        lambda_penalty=5.0,
    )
    assert res["two_chain_homomeric_contacts"] > 0
    assert res["two_chain_alternation_ratio"] < 1.0
    assert res["lattice_rmsd_refolded"] > res["lattice_rmsd_geom"]


def test_two_chain_alternating_metrics_se3_invariance() -> None:
    torch.manual_seed(42)
    chains = 4
    atoms_per_chain = 16
    pitch = 4.8
    base_chain = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.0, 1.2, 0.0],
            [1.5, 2.3, 0.0],
        ]
        * 4,
        dtype=torch.float32,
    )
    coords_list = [base_chain.clone() for _ in range(chains)]
    for c in range(chains):
        coords_list[c][:, 2] += c * pitch
    coords_ideal = torch.cat(coords_list)
    chain_ids = torch.cat(
        [torch.full((atoms_per_chain,), c, dtype=torch.long) for c in range(chains)]
    )
    bb_mask = torch.ones(coords_ideal.shape[0], dtype=torch.bool)
    partner_slots = {0: 0, 1: 1, 2: 0, 3: 1}

    theta = math.pi / 2
    r_mat = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    t_vec = torch.tensor([15.0, -20.0, 30.0], dtype=torch.float32)
    coords_rotated = (coords_ideal @ r_mat) + t_vec

    res = compute_two_chain_alternating_metrics(
        coords_refold=coords_rotated,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res["lattice_rmsd_geom"] == pytest.approx(0.0, abs=1e-4)
    assert res["lattice_rmsd_refolded"] == pytest.approx(0.0, abs=1e-4)
    assert res["two_chain_alternation_ratio"] == pytest.approx(1.0)


def _build_mock_double_tape(
    num_steps: int = 8,
    atoms_per_strand: int = 36,
    pitch: float = 4.8,
    layer_dist: float = 8.0,
    is_antiparallel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int], torch.Tensor]:
    """Helper to synthesize a double-tape lattice geometry for testing."""
    num_chains = 2 * num_steps
    res_y = torch.linspace(-15.0, 15.0, atoms_per_strand)
    base_strand = torch.stack(
        [torch.zeros_like(res_y), res_y, torch.zeros_like(res_y)], dim=1
    )

    coords_list = []
    chain_ids_list = []
    partner_slots = {}

    for c in range(num_chains):
        layer = c % 2
        step = c // 2
        strand = base_strand.clone()
        if is_antiparallel and (layer + step) % 2 == 1:
            strand[:, 1] = -strand[:, 1]
        strand[:, 0] += (layer - 0.5) * layer_dist
        strand[:, 2] += step * pitch
        coords_list.append(strand)
        chain_ids_list.append(torch.full((atoms_per_strand,), c, dtype=torch.long))
        partner_slots[c] = step % 2

    coords = torch.cat(coords_list)
    chain_ids = torch.cat(chain_ids_list)
    bb_mask = torch.ones(coords.shape[0], dtype=torch.bool)
    return coords, chain_ids, partner_slots, bb_mask


def test_double_tape_neatness_antiparallel_perfect_and_supercoiled() -> None:
    torch.manual_seed(42)
    coords_ideal, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )

    # 1. Perfect refold
    res_perf = compute_double_tape_neatness_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res_perf["strand_polarity_target_match"] == pytest.approx(1.0)
    assert res_perf["intra_sheet_alternation_ratio"] == pytest.approx(1.0)
    assert res_perf["local_step_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert res_perf["local_double_tape_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert res_perf["bilayer_dist_std"] == pytest.approx(0.0, abs=1e-4)
    assert res_perf["double_tape_neatness_score"] == pytest.approx(1.0, abs=1e-3)

    # 2. Supercoiled assembly (3 deg twist per step along tape axis)
    num_steps = 8
    atoms_per_strand = 36
    num_chains = 2 * num_steps
    coords_twisted_list = []
    twist_per_step_rad = math.radians(3.0)

    for c in range(num_chains):
        step = c // 2
        angle = step * twist_per_step_rad
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_z = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        c_coords = coords_ideal[c * atoms_per_strand : (c + 1) * atoms_per_strand]
        coords_twisted_list.append(c_coords @ rot_z)

    coords_twisted = torch.cat(coords_twisted_list)

    res_twisted = compute_double_tape_neatness_metrics(
        coords_refold=coords_twisted,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    # Local step RMSD remains small despite cumulative supercoiling
    assert res_twisted["strand_polarity_target_match"] == pytest.approx(1.0)
    assert res_twisted["intra_sheet_alternation_ratio"] == pytest.approx(1.0)
    assert res_twisted["local_step_rmsd"] < 0.5
    assert res_twisted["local_double_tape_rmsd"] < 0.5
    assert res_twisted["double_tape_neatness_score"] > 0.85


def test_double_tape_neatness_antiparallel_parallel_failure_mode() -> None:
    torch.manual_seed(42)
    coords_ideal, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    # Refold is completely parallel (failure mode)
    coords_par, _, _, _ = _build_mock_double_tape(
        num_steps=8, is_antiparallel=False
    )

    res = compute_double_tape_neatness_metrics(
        coords_refold=coords_par,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        lambda_penalty=5.0,
    )
    assert res["strand_polarity_target_match"] == pytest.approx(0.0)
    assert res["local_double_tape_rmsd"] >= 5.0
    assert res["double_tape_neatness_score"] == pytest.approx(0.0)


def test_double_tape_neatness_antiparallel_homomer_collapse() -> None:
    torch.manual_seed(42)
    coords_ideal, chain_ids, ideal_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    # Defective partner alternation (A-A-B-B within sheets)
    defective_slots = {}
    for c in range(16):
        step = c // 2
        defective_slots[c] = (step // 2) % 2

    res = compute_double_tape_neatness_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=defective_slots,
        partner_slots_ideal=ideal_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        lambda_penalty=5.0,
    )
    assert res["intra_sheet_alternation_ratio"] < 0.6
    assert res["local_double_tape_rmsd"] >= 2.0


def test_double_tape_neatness_parallel_target() -> None:
    torch.manual_seed(42)
    coords_par, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=False
    )
    # Refold is correctly parallel
    res_par = compute_double_tape_neatness_metrics(
        coords_refold=coords_par,
        coords_ideal=coords_par,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res_par["strand_polarity_target_match"] == pytest.approx(1.0)
    assert res_par["local_step_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert res_par["local_double_tape_rmsd"] == pytest.approx(0.0, abs=1e-4)

    # Refold is antiparallel (failure mode when parallel targeted)
    coords_anti, _, _, _ = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    res_anti_fail = compute_double_tape_neatness_metrics(
        coords_refold=coords_anti,
        coords_ideal=coords_par,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        lambda_penalty=5.0,
    )
    assert res_anti_fail["strand_polarity_target_match"] == pytest.approx(0.0)
    assert res_anti_fail["local_double_tape_rmsd"] >= 5.0
    assert res_anti_fail["double_tape_neatness_score"] == pytest.approx(0.0)


def test_double_tape_neatness_se3_invariance() -> None:
    torch.manual_seed(42)
    coords_ideal, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    theta = math.pi / 3
    r_mat = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    t_vec = torch.tensor([50.0, -100.0, 25.0], dtype=torch.float32)
    coords_rotated = (coords_ideal @ r_mat) + t_vec

    res_orig = compute_double_tape_neatness_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    res_rot = compute_double_tape_neatness_metrics(
        coords_refold=coords_rotated,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    for key in res_orig:
        assert res_rot[key] == pytest.approx(res_orig[key], abs=1e-4)


def test_double_tape_neatness_homomer_segregation_penalized() -> None:
    coords_ideal, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    # Perfectly matching geometry, but homomer-segregated sequence:
    # two_chain_alternation_ratio = 0.0
    res_segregated = compute_double_tape_neatness_metrics(
        coords_refold=coords_ideal,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
        lambda_penalty=5.0,
        two_chain_alternation_ratio=0.0,
    )
    assert res_segregated["double_tape_neatness_score"] == pytest.approx(0.0)
    assert res_segregated["local_double_tape_rmsd"] >= 5.0
    assert res_segregated["intra_sheet_alternation_ratio"] == pytest.approx(0.0)


def test_double_tape_neatness_tot_intra_zero_returns_zero() -> None:
    coords_ideal, chain_ids, partner_slots, bb_mask = _build_mock_double_tape(
        num_steps=8, is_antiparallel=True
    )
    # Translate every chain so no chains are within contact_distance (5.0 A)
    coords_dispersed = coords_ideal.clone()
    for c in range(16):
        c_mask = chain_ids == c
        coords_dispersed[c_mask] += torch.tensor([c * 100.0, 0.0, 0.0])

    res_dispersed = compute_double_tape_neatness_metrics(
        coords_refold=coords_dispersed,
        coords_ideal=coords_ideal,
        chain_ids_refold=chain_ids,
        chain_ids_ideal=chain_ids,
        partner_slots_refold=partner_slots,
        partner_slots_ideal=partner_slots,
        bb_mask_refold=bb_mask,
        bb_mask_ideal=bb_mask,
    )
    assert res_dispersed["intra_sheet_alternation_ratio"] == pytest.approx(0.0)
    assert res_dispersed["double_tape_neatness_score"] == pytest.approx(0.0)

