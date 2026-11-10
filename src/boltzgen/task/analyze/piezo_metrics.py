"""Piezoelectric and dipole response metrics for helical peptide assemblies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

DEBYE_PER_E_ANGSTROM: float = 4.80320427  # 1 e * Å in Debye (C*m conversion factor)
PEPTIDE_DIPOLE_DEBYE: float = 3.50  # Mean alpha-helix peptide bond dipole (Debye)

FORMAL_CHARGES: dict[str, float] = {
    "ARG": 1.0,
    "LYS": 1.0,
    "ASP": -1.0,
    "GLU": -1.0,
    "HIS": 0.1,
}


def compute_polar_order_parameter(
    chain_ca_coords: list[torch.Tensor],
) -> float:
    r"""Compute polar alignment order parameter along the assembly z-axis.

    .. math::
        S_{\text{polar}} = \frac{1}{K} \sum_{k=1}^K (\hat{\mathbf{u}}_k \cdot \hat{\mathbf{z}})

    where :math:`\hat{\mathbf{u}}_k = \frac{\mathbf{r}_{\text{C}\alpha, L_k} - \mathbf{r}_{\text{C}\alpha, 1}}{\|\mathbf{r}_{\text{C}\alpha, L_k} - \mathbf{r}_{\text{C}\alpha, 1}\|}`.

    Parameters
    ----------
    chain_ca_coords : list[torch.Tensor]
        List of C\alpha coordinate tensors, each of shape ``[L_k, 3]``.

    Returns
    -------
    float
        Order parameter in range [-1.0, 1.0]. A value of 1.0 indicates perfect
        parallel alignment along the +z axis; -1.0 indicates antiparallel alignment;
        0.0 indicates orthogonal or canceling antiparallel orientation.
    """
    if not chain_ca_coords:
        return 0.0

    device = chain_ca_coords[0].device
    dtype = chain_ca_coords[0].dtype
    z_hat = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    dot_products: list[float] = []
    for ca in chain_ca_coords:
        if ca.shape[0] < 2:
            continue
        u_k = ca[-1] - ca[0]
        u_norm = torch.linalg.norm(u_k)
        if u_norm > 1e-6:
            dot_products.append(float(torch.dot(u_k / u_norm, z_hat).item()))

    return float(np.mean(dot_products)) if dot_products else 0.0


def compute_assembly_dipole(
    atom_coords: torch.Tensor,
    atom_res_names: list[str],
    chain_ca_coords: list[torch.Tensor],
) -> dict[str, float]:
    r"""Compute total electric dipole moment vector and its components.

    Calculates:
    .. math::
        \mathbf{\mu}_{\text{total}} = \mathbf{\mu}_{\text{charge}} + \mathbf{\mu}_{\text{backbone}}

    where:
    .. math::
        \mathbf{\mu}_{\text{charge}} = \sum_{i=1}^{N_{\text{atoms}}} q_i (\mathbf{r}_i - \mathbf{r}_{\text{COM}}) \cdot C_{\text{Debye}}
        \mathbf{\mu}_{\text{backbone}} = \sum_{k=1}^K \sum_{j=1}^{L_k-1} \mu_{\text{pep}} \frac{\mathbf{r}_{j+1} - \mathbf{r}_j}{\|\mathbf{r}_{j+1} - \mathbf{r}_j\|}

    Parameters
    ----------
    atom_coords : torch.Tensor
        Coordinates of all atoms in the assembly, shape ``[N_atoms, 3]``.
    atom_res_names : list[str]
        Residue name (e.g. 'ARG', 'ALA') for each atom, length ``N_atoms``.
    chain_ca_coords : list[torch.Tensor]
        List of C\alpha coordinates per chain, each shape ``[L_k, 3]``.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - ``dipole_moment_total_debye``: Euclidean norm of total dipole vector.
        - ``dipole_moment_z_debye``: Z-component of total dipole vector.
        - ``dipole_charge_z_debye``: Z-component of formal charge contribution.
        - ``dipole_backbone_z_debye``: Z-component of backbone peptide dipole contribution.
    """
    if atom_coords.shape[0] == 0:
        return {
            "dipole_moment_total_debye": 0.0,
            "dipole_moment_z_debye": 0.0,
            "dipole_charge_z_debye": 0.0,
            "dipole_backbone_z_debye": 0.0,
        }

    device = atom_coords.device
    dtype = atom_coords.dtype

    # Vectorized formal charge calculation
    if atom_res_names and len(atom_res_names) == atom_coords.shape[0]:
        charges_np = np.array(
            [FORMAL_CHARGES.get(res.upper(), 0.0) for res in atom_res_names],
            dtype=np.float32,
        )
        charges = torch.from_numpy(charges_np).to(device=device, dtype=dtype)  # [N_atoms]
    else:
        charges = torch.zeros(atom_coords.shape[0], device=device, dtype=dtype)

    centroid = atom_coords.mean(dim=0, keepdim=True)  # [1, 3]
    centered_coords = atom_coords - centroid  # [N_atoms, 3]
    mu_charge = torch.sum(charges.unsqueeze(-1) * centered_coords, dim=0) * DEBYE_PER_E_ANGSTROM  # [3]

    # Vectorized peptide backbone dipole calculation
    mu_backbone = torch.zeros(3, device=device, dtype=dtype)
    for ca in chain_ca_coords:
        if ca.shape[0] < 2:
            continue
        tangents = ca[1:] - ca[:-1]  # [L-1, 3]
        norms = torch.linalg.norm(tangents, dim=-1, keepdim=True).clamp(min=1e-6)
        unit_tangents = tangents / norms
        mu_backbone += torch.sum(unit_tangents * PEPTIDE_DIPOLE_DEBYE, dim=0)

    mu_total = mu_charge + mu_backbone
    return {
        "dipole_moment_total_debye": float(torch.linalg.norm(mu_total).item()),
        "dipole_moment_z_debye": float(mu_total[2].item()),
        "dipole_charge_z_debye": float(mu_charge[2].item()),
        "dipole_backbone_z_debye": float(mu_backbone[2].item()),
    }


def compute_tri_state_piezo_metrics(
    mu_z_relaxed: float,
    mu_z_tension: float,
    mu_z_compression: float,
    strain_tension: float,
    strain_compression: float,
    volume_angstrom3: float,
) -> dict[str, float]:
    r"""Evaluate linear piezoelectric response, asymmetry, and compliance from tri-state screen.

    Calculates:
    .. math::
        e_{33}^{(\text{ten})} = \frac{1}{V_0} \frac{\mu_z(S_{\text{ten}}) - \mu_z(S_0)}{\epsilon_{\text{ten}}}
        e_{33}^{(\text{comp})} = \frac{1}{V_0} \frac{\mu_z(S_{\text{comp}}) - \mu_z(S_0)}{-\epsilon_{\text{comp}}}
        e_{33} = \frac{1}{2} (e_{33}^{(\text{ten})} + e_{33}^{(\text{comp})})
        \eta_{\text{asym}} = \frac{|e_{33}^{(\text{ten})} - e_{33}^{(\text{comp})}|}{\max(|e_{33}^{(\text{ten})}|, |e_{33}^{(\text{comp})}|, 10^{-4})}

    Parameters
    ----------
    mu_z_relaxed : float
        Z-dipole of relaxed state S_0 (Debye).
    mu_z_tension : float
        Z-dipole of tension state S_ten (Debye).
    mu_z_compression : float
        Z-dipole of compression state S_comp (Debye).
    strain_tension : float
        Tensile strain magnitude \epsilon > 0 (e.g. 0.10).
    strain_compression : float
        Magnitude of compressive strain \epsilon > 0 (e.g. 0.10).
    volume_angstrom3 : float
        Assembly volume in Å³.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - ``delta_mu_z_tension``: Change in z-dipole under tension (Debye).
        - ``delta_mu_z_compression``: Change in z-dipole under compression (Debye).
        - ``piezo_e33_tension``: Piezoelectric stress coefficient under tension (D / Å³).
        - ``piezo_e33_compression``: Piezoelectric stress coefficient under compression (D / Å³).
        - ``piezo_e33_mean``: Mean linear piezoelectric coefficient (D / Å³).
        - ``piezo_asymmetry``: Dimensionless electromechanical asymmetry ratio.
    """
    vol = max(volume_angstrom3, 1.0)
    eps_ten = max(abs(strain_tension), 1e-6)
    eps_comp = max(abs(strain_compression), 1e-6)

    delta_mu_ten = mu_z_tension - mu_z_relaxed
    delta_mu_comp = mu_z_compression - mu_z_relaxed

    e33_ten = (delta_mu_ten / vol) / eps_ten
    # Note: compressive strain represents negative strain (-eps_comp)
    e33_comp = (delta_mu_comp / vol) / (-eps_comp)
    e33_mean = 0.5 * (e33_ten + e33_comp)

    denom = max(abs(e33_ten), abs(e33_comp), 1e-6)
    asymmetry = abs(e33_ten - e33_comp) / denom

    return {
        "delta_mu_z_tension": float(delta_mu_ten),
        "delta_mu_z_compression": float(delta_mu_comp),
        "piezo_e33_tension": float(e33_ten),
        "piezo_e33_compression": float(e33_comp),
        "piezo_e33_mean": float(e33_mean),
        "piezo_asymmetry": float(asymmetry),
    }
