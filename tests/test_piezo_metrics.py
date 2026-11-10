"""Unit tests for piezoelectric and dipole response metrics."""

import math
import numpy as np
import pytest
import torch

from boltzgen.task.analyze.piezo_metrics import (
    compute_polar_order_parameter,
    compute_assembly_dipole,
    compute_tri_state_piezo_metrics,
    PEPTIDE_DIPOLE_DEBYE,
    DEBYE_PER_E_ANGSTROM,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure deterministic execution across test runs."""
    torch.manual_seed(42)
    np.random.seed(42)


def test_polar_order_parameter_parallel():
    """Verify parallel helices aligned along +z yield order parameter S ≈ 1.0."""
    # 4 parallel helices running from z=0 to z=30
    helices = [
        torch.tensor([[0.0, 0.0, float(z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[10.0, 0.0, float(z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[0.0, 10.0, float(z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[10.0, 10.0, float(z)] for z in range(20)], dtype=torch.float32),
    ]
    s_polar = compute_polar_order_parameter(helices)
    assert pytest.approx(1.0, abs=1e-5) == s_polar


def test_polar_order_parameter_antiparallel():
    """Verify alternating antiparallel helices yield order parameter S ≈ 0.0."""
    helices = [
        torch.tensor([[0.0, 0.0, float(z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[10.0, 0.0, float(19 - z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[0.0, 10.0, float(z)] for z in range(20)], dtype=torch.float32),
        torch.tensor([[10.0, 10.0, float(19 - z)] for z in range(20)], dtype=torch.float32),
    ]
    s_polar = compute_polar_order_parameter(helices)
    assert pytest.approx(0.0, abs=1e-5) == s_polar


def test_assembly_dipole_parallel_neutral():
    """Verify parallel uncharged helices sum peptide bond dipoles along +z."""
    n_chains = 4
    n_res = 20
    # Standard alpha helix pitch: 1.5 Å per residue rise along z
    chain_ca = [
        torch.tensor([[float(c * 10), 0.0, float(i * 1.5)] for i in range(n_res)], dtype=torch.float32)
        for c in range(n_chains)
    ]
    # Coordinates for all atoms (dummy ALA)
    all_coords = torch.cat(chain_ca, dim=0)
    all_res = ["ALA"] * len(all_coords)

    dipole_dict = compute_assembly_dipole(all_coords, all_res, chain_ca)

    expected_backbone_z = n_chains * (n_res - 1) * PEPTIDE_DIPOLE_DEBYE
    assert pytest.approx(expected_backbone_z, rel=1e-3) == dipole_dict["dipole_backbone_z_debye"]
    assert pytest.approx(0.0, abs=1e-5) == dipole_dict["dipole_charge_z_debye"]
    assert pytest.approx(expected_backbone_z, rel=1e-3) == dipole_dict["dipole_moment_z_debye"]


def test_assembly_dipole_formal_charges():
    """Verify formal charge displacement contributes correctly to dipole moment."""
    # Two atoms along z: ASP (-1.0) at z=-10, LYS (+1.0) at z=+10
    coords = torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]], dtype=torch.float32)
    res_names = ["ASP", "LYS"]
    dipole_dict = compute_assembly_dipole(coords, res_names, [])

    # Centroid is at z=0. ASP is at -10, LYS is at +10.
    # mu_charge_z = (-1.0 * -10.0 + 1.0 * 10.0) * DEBYE_PER_E_ANGSTROM = 20.0 * 4.8032
    expected_charge_z = 20.0 * DEBYE_PER_E_ANGSTROM
    assert pytest.approx(expected_charge_z, rel=1e-4) == dipole_dict["dipole_charge_z_debye"]
    assert pytest.approx(expected_charge_z, rel=1e-4) == dipole_dict["dipole_moment_z_debye"]


def test_tri_state_piezo_metrics():
    """Verify tri-state piezo coefficients and linearity/asymmetry."""
    mu_relaxed = 1000.0  # Debye
    # +10% strain increases dipole by 100 D
    mu_ten = 1100.0
    # -10% strain decreases dipole by 100 D (perfectly symmetric)
    mu_comp = 900.0
    vol = 10000.0  # Å³
    strain_ten = 0.10
    strain_comp = 0.10

    metrics = compute_tri_state_piezo_metrics(
        mu_z_relaxed=mu_relaxed,
        mu_z_tension=mu_ten,
        mu_z_compression=mu_comp,
        strain_tension=strain_ten,
        strain_compression=strain_comp,
        volume_angstrom3=vol,
    )

    # e33_ten = (100 / 10000) / 0.10 = 0.10 D / Å³
    # e33_comp = (-100 / 10000) / -0.10 = 0.10 D / Å³
    assert pytest.approx(0.10, rel=1e-4) == metrics["piezo_e33_tension"]
    assert pytest.approx(0.10, rel=1e-4) == metrics["piezo_e33_compression"]
    assert pytest.approx(0.10, rel=1e-4) == metrics["piezo_e33_mean"]
    assert pytest.approx(0.0, abs=1e-5) == metrics["piezo_asymmetry"]
