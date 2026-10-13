"""Unit tests for cylindrical topology geometry, trace constraints, and metrics."""

# ruff: noqa: INP001, PLR2004

from __future__ import annotations

import math

import torch

from boltzgen.model.modules.material_layout import (
    build_heteromer_counter_screen_specs,
)
from boltzgen.model.modules.materials import (
    compute_element_trace,
    constrain_trace_to_cylinder,
    generate_ideal_lattice,
)
from boltzgen.task.analyze.material_metrics import (
    compute_material_metrics,
    cylinder_metrics,
)


def test_generate_ideal_lattice_cylindrical_matches_cyclic_at_dz_zero() -> None:
    """With asym_unit_size=1 and target_dz=0, cylindrical matches cyclic lattice."""
    params = {"target_radius": 14.0, "target_dz": 0.0}
    cyl_coms, cyl_rots = generate_ideal_lattice(
        "cylindrical", 6, 1, params, device=torch.device("cpu"), dtype=torch.float64
    )
    cyclic_coms, cyclic_rots = generate_ideal_lattice(
        "cyclic", 6, 1, params, device=torch.device("cpu"), dtype=torch.float64
    )
    torch.testing.assert_close(cyl_coms, cyclic_coms, atol=1e-12, rtol=0)
    torch.testing.assert_close(cyl_rots, cyclic_rots, atol=1e-12, rtol=0)


def test_generate_ideal_lattice_cylindrical_intra_unit_stagger() -> None:
    """Test multi-helix symmetrical unit stagger and exact Cn/U rotational symmetry."""
    n_chains = 8
    asym_unit_size = 2
    n_placements = n_chains // asym_unit_size  # 4 placements -> C4 symmetry
    target_radius = 12.0
    stagger_dz = 3.0
    params = {"target_radius": target_radius, "target_dz": stagger_dz}

    coms, rots = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert coms.shape == (8, 3)
    assert rots.shape == (8, 3, 3)

    # In placement 0: helix 0 has z = -1.5, helix 1 has z = +1.5
    assert math.isclose(coms[0, 2].item(), -1.5, abs_tol=1e-9)
    assert math.isclose(coms[1, 2].item(), 1.5, abs_tol=1e-9)

    # Check exact C4 rotational symmetry across placements
    base_step = 2.0 * math.pi / n_placements
    for p in range(n_placements):
        expected_r = torch.tensor(
            [
                [math.cos(p * base_step), -math.sin(p * base_step), 0.0],
                [math.sin(p * base_step), math.cos(p * base_step), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        for u in range(asym_unit_size):
            idx = p * asym_unit_size + u
            ref_idx = u
            expected_com = expected_r @ coms[ref_idx]
            torch.testing.assert_close(coms[idx], expected_com, atol=1e-12, rtol=0)


def test_generate_ideal_lattice_cylindrical_spacing_noise_preserves_symmetry() -> None:
    """Coupled spacing noise perturbs stagger while preserving exact group symmetry."""
    n_chains = 6
    asym_unit_size = 2
    n_placements = 3  # C3 symmetry
    params = {
        "target_radius": 15.0,
        "target_dz": 2.0,
        "stagger_z_noise": 0.75,
        "stagger_phi_noise": 0.05,
    }

    coms, _ = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # Axial stagger is (2.0 + 0.75) = 2.75 -> +/- 1.375
    assert math.isclose(coms[0, 2].item(), -1.375, abs_tol=1e-9)
    assert math.isclose(coms[1, 2].item(), 1.375, abs_tol=1e-9)

    # Verify C3 symmetry across placements
    base_step = 2.0 * math.pi / n_placements
    for p in range(n_placements):
        expected_r = torch.tensor(
            [
                [math.cos(p * base_step), -math.sin(p * base_step), 0.0],
                [math.sin(p * base_step), math.cos(p * base_step), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        for u in range(asym_unit_size):
            idx = p * asym_unit_size + u
            expected_com = expected_r @ coms[u]
            torch.testing.assert_close(coms[idx], expected_com, atol=1e-12, rtol=0)


def test_compute_element_trace_alpha_helix() -> None:
    """Rolling 4-residue filter extracts central helical axis from CA atoms."""
    # Ideal alpha-helix along Z: radius r_1 = 2.3 A, 3.6 residues/turn, rise = 1.5 A
    n_res = 20
    r_helix = 2.3
    rise = 1.5
    omega = 2.0 * math.pi / 3.6

    angles = torch.arange(n_res, dtype=torch.float64) * omega
    z_coords = torch.arange(n_res, dtype=torch.float64) * rise
    ca_coords = torch.stack(
        [r_helix * torch.cos(angles), r_helix * torch.sin(angles), z_coords], dim=1
    )

    trace = compute_element_trace(ca_coords, secondary_structure="H")

    # In the interior of the helix (away from edges), central axis is at x ~ 0, y ~ 0
    interior_xy_dist = torch.linalg.vector_norm(trace[3:-3, :2], dim=1)
    # The 4-residue filter reduces 2.3 A radius to ~0.26 A residual oscillation
    assert (interior_xy_dist < 0.3).all()


def test_compute_element_trace_beta_strand() -> None:
    """Rolling 2-residue filter smooths pleating in a beta strand."""
    n_res = 12
    pleat = 0.5
    rise = 3.3
    signs = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_res)], dtype=torch.float64
    )
    z_coords = torch.arange(n_res, dtype=torch.float64) * rise
    ca_coords = torch.stack(
        [signs * pleat, torch.zeros(n_res, dtype=torch.float64), z_coords], dim=1
    )

    trace = compute_element_trace(ca_coords, secondary_structure="S")

    # The 2-residue mean of alternating +/- pleat is exactly 0.0 in the interior
    interior_pleat = torch.abs(trace[1:-1, 0])
    assert (interior_pleat < 1e-12).all()


def test_constrain_trace_to_cylinder_preserves_internal_geometry() -> None:
    """Radial translation places trace on cylinder preserving covalent bonds."""
    n_res = 15
    # Build a peptide with 4 atoms per residue (N, CA, C, O)
    # Put CA at a starting radius of 8.0 A, target radius is 14.0 A
    coords = torch.zeros((n_res * 4, 3), dtype=torch.float64)
    for r in range(n_res):
        ca = torch.tensor([8.0, float(r), float(r) * 1.5], dtype=torch.float64)
        n_atom = ca + torch.tensor([-0.5, -0.8, -0.4], dtype=torch.float64)
        c_atom = ca + torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)
        o_atom = c_atom + torch.tensor([0.2, 0.1, 1.2], dtype=torch.float64)
        coords[r * 4 + 0] = n_atom
        coords[r * 4 + 1] = ca
        coords[r * 4 + 2] = c_atom
        coords[r * 4 + 3] = o_atom

    # Compute original internal bond lengths
    orig_bonds = []
    for r in range(n_res):
        orig_bonds.append(
            torch.linalg.vector_norm(coords[r * 4 + 1] - coords[r * 4 + 0])
        )  # N-CA
        orig_bonds.append(
            torch.linalg.vector_norm(coords[r * 4 + 2] - coords[r * 4 + 1])
        )  # CA-C
        orig_bonds.append(
            torch.linalg.vector_norm(coords[r * 4 + 3] - coords[r * 4 + 2])
        )  # C-O

    target_radius = 14.0
    constrained = constrain_trace_to_cylinder(
        coords, radius=target_radius, secondary_structure="H"
    )

    # 1. Check all internal bond lengths are 100% preserved
    for idx, r in enumerate(range(n_res)):
        n_ca = torch.linalg.vector_norm(constrained[r * 4 + 1] - constrained[r * 4 + 0])
        ca_c = torch.linalg.vector_norm(constrained[r * 4 + 2] - constrained[r * 4 + 1])
        c_o = torch.linalg.vector_norm(constrained[r * 4 + 3] - constrained[r * 4 + 2])
        torch.testing.assert_close(n_ca, orig_bonds[idx * 3 + 0], atol=1e-12, rtol=0)
        torch.testing.assert_close(ca_c, orig_bonds[idx * 3 + 1], atol=1e-12, rtol=0)
        torch.testing.assert_close(c_o, orig_bonds[idx * 3 + 2], atol=1e-12, rtol=0)

    # 2. Check that the constrained trace lies at target_radius within 0.5 A
    ca_coords_out = constrained.reshape(n_res, 4, 3)[:, 1, :]
    trace_out = compute_element_trace(ca_coords_out, secondary_structure="H")
    trace_radii = torch.linalg.vector_norm(trace_out[:, :2], dim=1)
    torch.testing.assert_close(
        trace_radii,
        torch.full((n_res,), target_radius, dtype=torch.float64),
        atol=0.5,
        rtol=0,
    )


def test_align_consensus_to_cylinder() -> None:
    """Test align_consensus_to_cylinder aligns primary axis along +Z."""
    from boltzgen.model.modules.materials import (
        align_consensus_to_cylinder,
        strand_direction,
    )

    # Create a 20-residue peptide oriented along the X axis
    n_res = 20
    x = torch.linspace(-15.0, 15.0, n_res, dtype=torch.float64)
    # spiral slightly around X
    ang = torch.linspace(0, 4 * math.pi, n_res, dtype=torch.float64)
    y = 2.3 * torch.cos(ang)
    z = 2.3 * torch.sin(ang)
    ca = torch.stack([x, y, z], dim=1)

    aligned = align_consensus_to_cylinder(ca)
    # Check that primary axis is now aligned along Z
    v_dir = strand_direction(aligned)
    assert abs(v_dir[2].item()) > 0.95  # along the cylinder axis (sign ambiguous)
    assert abs(v_dir[0].item()) < 0.1
    assert abs(v_dir[1].item()) < 0.1


def test_align_consensus_to_cylinder_preserves_tangential_tilt() -> None:
    """Test align_consensus_to_cylinder preserves crossing angle / tilt in (Y, Z)."""
    from boltzgen.model.modules.materials import (
        align_consensus_to_cylinder,
        strand_direction,
    )

    n_res = 20
    alpha = math.radians(20.0)  # 20 degree tilt in the tangent plane
    # Ideal alpha helix with 20 deg tilt and a small out-of-plane radial error
    t = torch.arange(n_res, dtype=torch.float64)
    omega = 2.0 * math.pi / 3.6
    r_helix = 2.3
    x_local = r_helix * torch.cos(t * omega)
    y_local = r_helix * torch.sin(t * omega)
    z_local = (t - (n_res - 1) / 2.0) * 1.5
    ca_local = torch.stack([x_local, y_local, z_local], dim=1)

    r_tilt = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(alpha), math.sin(alpha)],
            [0.0, -math.sin(alpha), math.cos(alpha)],
        ],
        dtype=torch.float64,
    )
    r_radial_error = torch.tensor(
        [
            [math.cos(math.radians(3.0)), 0.0, math.sin(math.radians(3.0))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(3.0)), 0.0, math.cos(math.radians(3.0))],
        ],
        dtype=torch.float64,
    )
    ca_perturbed = ca_local @ r_tilt.T @ r_radial_error.T

    orig_bonds = torch.linalg.vector_norm(ca_perturbed[1:] - ca_perturbed[:-1], dim=1)
    v_orig = strand_direction(ca_perturbed)
    expected_tilt_deg = math.degrees(math.atan2(v_orig[1].item(), v_orig[2].item()))

    aligned = align_consensus_to_cylinder(ca_perturbed)
    aligned_bonds = torch.linalg.vector_norm(aligned[1:] - aligned[:-1], dim=1)

    # 1. Exact isometry: secondary structure bond lengths are 100% preserved
    torch.testing.assert_close(aligned_bonds, orig_bonds, atol=1e-12, rtol=0)

    # 2. Tangential tilt angle is preserved within 0.5 degrees
    v_dir = strand_direction(aligned)
    tilt_deg = math.degrees(math.atan2(v_dir[1].item(), v_dir[2].item()))
    assert abs(tilt_deg - expected_tilt_deg) < 0.5

    # 3. Out-of-plane radial tilt is eliminated
    assert abs(v_dir[0].item()) < 0.05


def test_cylindrical_antiparallel_consensus_slots_preserve_helical_geometry() -> None:
    """Test that separating antiparallel consensus slots prevents helical squashing."""
    from boltzgen.model.modules.materials import (
        constrain_trace_to_cylinder,
        generate_ideal_lattice,
        unfold_from_consensus_frame,
    )

    n_chains = 10
    asym_unit_size = 1
    target_radius = 13.0
    params = {"target_radius": target_radius, "target_dz": 0.0}
    coms, rots = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    n_res = 20
    t = torch.arange(n_res, dtype=torch.float64)
    omega = 2.0 * math.pi / 3.6
    r_helix = 2.3
    z_rise = (t - (n_res - 1) / 2.0) * 1.5
    ca_local = torch.stack(
        [r_helix * torch.cos(t * omega), r_helix * torch.sin(t * omega), z_rise],
        dim=1,
    )

    # Slot 0 for even (up) chains, Slot 1 for odd (down) chains
    r_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    slots = [ca_local.clone(), ca_local.clone()]

    # Unfold each chain according to its slot
    unfolded_chains = []
    for idx in range(n_chains):
        c_idx = idx % 2
        base_coords = slots[c_idx].clone()
        if idx % 2 == 1:
            base_coords = base_coords @ r_flip
        unfolded = unfold_from_consensus_frame(
            base_coords, coms[idx : idx + 1], rots[idx : idx + 1]
        )[0]
        constrained = constrain_trace_to_cylinder(
            unfolded, radius=target_radius, secondary_structure="H"
        )
        unfolded_chains.append(constrained)

    for c in unfolded_chains:
        dists = torch.linalg.vector_norm(c[1:] - c[:-1], dim=1)
        # All chains maintain ideal alpha-helical CA-CA bond lengths ~ 3.8 A
        assert math.isclose(dists.mean().item(), 3.8, abs_tol=0.15)


def test_cylinder_metrics_ideal_and_distorted() -> None:
    """Test cylinder_metrics calculation and RMSD reporting."""
    n_chains = 6
    radius = 15.0
    angles = torch.arange(n_chains, dtype=torch.float64) * (2.0 * math.pi / n_chains)
    chain_coms = torch.stack(
        [
            radius * torch.cos(angles),
            radius * torch.sin(angles),
            torch.zeros(n_chains, dtype=torch.float64),
        ],
        dim=1,
    )

    # 20 CA atoms exactly on cylinder
    ca_angles = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
    ca_coords = torch.stack(
        [
            radius * torch.cos(ca_angles),
            radius * torch.sin(ca_angles),
            torch.linspace(-10, 10, 20, dtype=torch.float64),
        ],
        dim=1,
    )

    metrics = cylinder_metrics(chain_coms, ca_coords, target_radius=radius)
    assert math.isclose(metrics["cylinder_radius_rmsd"], 0.0, abs_tol=1e-6)
    assert math.isclose(metrics["neg_cylinder_radius_rmsd"], 0.0, abs_tol=1e-6)
    assert math.isclose(metrics["cylinder_mean_radius"], radius, abs_tol=1e-6)

    # Add 2.0 A radial offset to CA atoms
    ca_coords_distorted = ca_coords.clone()
    ca_coords_distorted[:, :2] *= (radius + 2.0) / radius
    metrics_distorted = cylinder_metrics(
        chain_coms, ca_coords_distorted, target_radius=radius
    )
    assert math.isclose(metrics_distorted["cylinder_radius_rmsd"], 2.0, abs_tol=1e-5)
    assert math.isclose(
        metrics_distorted["neg_cylinder_radius_rmsd"], -2.0, abs_tol=1e-5
    )


def test_compute_material_metrics_dispatches_cylindrical() -> None:
    """Test dispatch of cylindrical metrics in compute_material_metrics."""
    chain_coms = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-10.0, 0.0, 0.0]], dtype=torch.float64
    )
    ca_coords = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-10.0, 0.0, 0.0]], dtype=torch.float64
    )
    backbone_coords = ca_coords

    res = compute_material_metrics(
        "cylindrical", chain_coms, backbone_coords, ca_coords
    )
    assert "cylinder_radius_rmsd" in res
    assert "neg_cylinder_radius_rmsd" in res
    assert "cylinder_pore_radius" in res


def test_heteromer_counter_screen_supports_cylindrical() -> None:
    """Test build_heteromer_counter_screen_specs supports cylindrical."""
    partners = {"A": "ACDEFGHIKL", "B": "KLMNPQRSTV"}
    jobs = build_heteromer_counter_screen_specs(
        partners,
        copies=4,
        topology="cylindrical",
        topology_params={"target_radius": 13.0, "target_dz": 2.0},
        parent_design_id="barrel_test",
    )
    assert "A" in jobs
    assert "B" in jobs
    assert jobs["A"]["topology"] == "cylindrical"
    assert jobs["B"]["topology"] == "cylindrical"
    assert jobs["A"]["topology_params"]["target_radius"] == 13.0


def test_wrap_consensus_to_cylinder_helical_axis_on_surface() -> None:
    """Test wrap_consensus_to_cylinder places helical axis strictly on rho = R."""
    from boltzgen.model.modules.materials import wrap_consensus_to_cylinder

    torch.manual_seed(42)
    n_res = 20
    atom_tokens = torch.arange(n_res).repeat_interleave(4)
    res_t = torch.arange(n_res, dtype=torch.float64)
    ca = torch.stack(
        [
            2.3 * torch.cos(res_t * 1.745),
            2.3 * torch.sin(res_t * 1.745),
            res_t * 1.5,
        ],
        dim=1,
    )
    n_offset = torch.tensor([-0.8, -1.2, -0.5], dtype=torch.float64)
    c_offset = torch.tensor([0.9, 1.1, 0.4], dtype=torch.float64)
    o_offset = torch.tensor([1.8, 0.8, 1.2], dtype=torch.float64)

    atoms = torch.zeros((n_res * 4, 3), dtype=torch.float64)
    for i in range(n_res):
        atoms[4 * i + 0] = ca[i] + n_offset
        atoms[4 * i + 1] = ca[i]
        atoms[4 * i + 2] = ca[i] + c_offset
        atoms[4 * i + 3] = ca[i] + o_offset

    atoms = atoms - atoms.mean(dim=0)
    alpha = math.radians(20.0)
    rot_tilt = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(alpha), -math.sin(alpha)],
            [0.0, math.sin(alpha), math.cos(alpha)],
        ],
        dtype=torch.float64,
    )
    atoms = atoms @ rot_tilt

    radius = 13.0
    ideal_com = torch.tensor(
        [radius * math.cos(0.7), radius * math.sin(0.7), 2.5], dtype=torch.float64
    )

    wrapped = wrap_consensus_to_cylinder(
        atoms, ideal_com, radius, atom_token_ids=atom_tokens
    )

    # 1. Internal residue geometry is strictly rigid (zero distortion)
    for i in range(n_res):
        orig_len = torch.norm(atoms[4 * i] - atoms[4 * i + 1]).item()
        wrapped_len = torch.norm(wrapped[4 * i] - wrapped[4 * i + 1]).item()
        assert math.isclose(orig_len, wrapped_len, abs_tol=1e-8)

    # 2. CA-CA distances are preserved
    ca_wrapped = wrapped[1::4]
    ca_diffs = torch.norm(torch.diff(ca_wrapped, dim=0), dim=1)
    assert math.isclose(ca_diffs.mean().item(), 3.8, abs_tol=0.15)

    # 3. 4-residue trace radius is centered on target radius with zero radial component
    trace = torch.stack(
        [ca_wrapped[j : j + 4].mean(0) for j in range(len(ca_wrapped) - 3)]
    )
    trace_rho = torch.hypot(trace[:, 0], trace[:, 1])
    assert math.isclose(trace_rho.mean().item(), radius, abs_tol=0.3)


def test_wrap_and_unwrap_cylinder_roundtrip() -> None:
    """Test wrapping and unwrapping invert accurately."""
    from boltzgen.model.modules.materials import (
        unwrap_cylinder_to_consensus,
        wrap_consensus_to_cylinder,
    )

    torch.manual_seed(42)
    n_res = 16
    atom_tokens = torch.arange(n_res).repeat_interleave(4)
    res_t = torch.arange(n_res, dtype=torch.float64)
    ca = torch.stack(
        [
            2.3 * torch.cos(res_t * 1.745),
            2.3 * torch.sin(res_t * 1.745),
            res_t * 1.5,
        ],
        dim=1,
    )
    atoms = ca.repeat_interleave(4, dim=0) - ca.mean(dim=0)
    alpha = math.radians(15.0)
    rot_tilt = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(alpha), -math.sin(alpha)],
            [0.0, math.sin(alpha), math.cos(alpha)],
        ],
        dtype=torch.float64,
    )
    atoms = atoms @ rot_tilt

    radius = 14.0
    ideal_com = torch.tensor([radius, 0.0, 0.0], dtype=torch.float64)

    wrapped = wrap_consensus_to_cylinder(
        atoms, ideal_com, radius, atom_token_ids=atom_tokens
    )
    unwrapped = unwrap_cylinder_to_consensus(
        wrapped, ideal_com, radius, atom_token_ids=atom_tokens
    )

    # Roundtrip recovers coordinates with sub-Angstrom agreement
    max_err = torch.norm(atoms - unwrapped, dim=1).max().item()
    assert max_err < 0.1


def test_generate_ideal_lattice_cylindrical_slide_vector_asym_unit_3() -> None:
    """Test slide vector per helix for asym_unit_size=3 with exact C2 symmetry."""
    n_chains = 6
    asym_unit_size = 3
    target_radius = 14.0
    slide_vec = [-2.5, 0.5, 2.0]  # Zero-sum slide vector
    params = {
        "target_radius": target_radius,
        "target_dz": 0.0,
        "slide_vector": slide_vec,
    }

    coms, rots = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # All chain COMs lie on cylinder surface rho = target_radius
    radii = torch.linalg.norm(coms[:, :2], dim=1)
    torch.testing.assert_close(
        radii,
        torch.full_like(radii, target_radius),
        atol=1e-12,
        rtol=0,
    )

    # In placement 0 (chains 0, 1, 2), z coordinates match slide vector
    assert math.isclose(coms[0, 2].item(), -2.5, abs_tol=1e-9)
    assert math.isclose(coms[1, 2].item(), 0.5, abs_tol=1e-9)
    assert math.isclose(coms[2, 2].item(), 2.0, abs_tol=1e-9)

    # In placement 1 (chains 3, 4, 5), z coordinates match placement 0
    assert math.isclose(coms[3, 2].item(), -2.5, abs_tol=1e-9)
    assert math.isclose(coms[4, 2].item(), 0.5, abs_tol=1e-9)
    assert math.isclose(coms[5, 2].item(), 2.0, abs_tol=1e-9)

    # Rotational symmetry C2: placement 1 is rotated by pi relative to placement 0
    r_c2 = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    for u in range(asym_unit_size):
        expected_com = r_c2 @ coms[u]
        torch.testing.assert_close(
            coms[u + asym_unit_size], expected_com, atol=1e-12, rtol=0
        )


def test_generate_ideal_lattice_cylindrical_slide_vector_with_tilt_angle() -> None:
    """Test helical geodesic slide displacement for tilted helices."""
    n_chains = 4
    asym_unit_size = 2
    target_radius = 12.0
    tilt_deg = 20.0
    slide_vec = [-3.0, 3.0]
    params = {
        "target_radius": target_radius,
        "target_dz": 0.0,
        "target_angle": tilt_deg,
        "slide_vector": slide_vec,
    }

    coms, _ = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # Radii must remain strictly target_radius
    radii = torch.linalg.norm(coms[:, :2], dim=1)
    torch.testing.assert_close(
        radii,
        torch.full_like(radii, target_radius),
        atol=1e-12,
        rtol=0,
    )

    tilt_rad = math.radians(tilt_deg)
    # Expected delta z = s_u * cos(alpha)
    assert math.isclose(coms[0, 2].item(), -3.0 * math.cos(tilt_rad), abs_tol=1e-9)
    assert math.isclose(coms[1, 2].item(), 3.0 * math.cos(tilt_rad), abs_tol=1e-9)

    # Expected delta phi = s_u * sin(alpha) / R added to base intra-unit azimuth
    phi_0 = math.atan2(coms[0, 1].item(), coms[0, 0].item())
    base_phi_0 = -0.5 * (math.pi / asym_unit_size)
    expected_phi_0 = base_phi_0 - 3.0 * math.sin(tilt_rad) / target_radius
    assert math.isclose(phi_0, expected_phi_0, abs_tol=1e-9)


def test_generate_ideal_lattice_cylindrical_antiparallel_dimer_slide() -> None:
    """Test alternating antiparallel chain sliding when asym_unit_size=1."""
    n_chains = 6
    asym_unit_size = 1
    target_radius = 10.0
    slide_vec = [-1.5, 1.5]
    params = {
        "target_radius": target_radius,
        "target_dz": 0.0,
        "slide_vector": slide_vec,
    }

    coms, _ = generate_ideal_lattice(
        "cylindrical",
        n_chains,
        asym_unit_size,
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    for i in range(n_chains):
        expected_z = -1.5 if i % 2 == 0 else 1.5
        assert math.isclose(coms[i, 2].item(), expected_z, abs_tol=1e-9)


