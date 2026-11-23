"""Regression tests for diffusion/guidance, builder/CLI, inverse-folding, and analysis/filtering fixes."""

# ruff: noqa: INP001, PLR2004, E501

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from boltzgen.data import const
from boltzgen.data.parse.schema import parse_residue_constraints
from boltzgen.model.modules.inverse_fold import (
    InverseFoldingDecoder,
    build_constraint_logit_mask,
)
from boltzgen.model.modules.material_layout import expand_material_spec
from boltzgen.model.modules.materials import (
    compute_sigma_guidance_scale,
    generate_ideal_lattice,
    rotational_consensus_average,
    safe_kabsch_rotation,
    so3_interpolate_coords,
    unwrap_cylinder_to_consensus,
)
from boltzgen.task.analyze.analyze_utils import compute_cyclic_symmetry_metrics
from boltzgen.task.analyze.material_metrics import (
    _kabsch_rmsd,
    compute_chain_order_independent_rmsd,
    cylinder_metrics,
)
from boltzgen.task.filter.filter import Filter


def test_sigma_guidance_scale_three_phase_and_clamping() -> None:
    """Verify noise-level 3-phase bell curve: 0 at high sigma, plateau mid, 0 at low sigma, clamped <= 1."""
    # Pure noise: 0
    assert compute_sigma_guidance_scale(500.0, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 0.0
    assert compute_sigma_guidance_scale(50.0, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 0.0
    # Ramp-in interior
    mid_in = compute_sigma_guidance_scale(30.0, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5)
    assert 0.0 < mid_in < 1.0
    # Plateau
    assert compute_sigma_guidance_scale(10.0, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 1.0
    # Late release taper
    mid_out = compute_sigma_guidance_scale(2.5, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5)
    assert 0.0 < mid_out < 1.0
    # Free refinement at low sigma: strictly 0.0
    assert compute_sigma_guidance_scale(1.5, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 0.0
    assert compute_sigma_guidance_scale(0.01, 1.0, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 0.0
    # Clamping base_scale > 1.0 never extrapolates
    assert compute_sigma_guidance_scale(10.0, 2.5, sigma_start=50.0, sigma_peak=18.0, sigma_taper_start=4.0, sigma_release=1.5) == 1.0


def test_so3_interpolation_preserves_bond_lengths_at_180_deg() -> None:
    """SO(3) geodesic interpolation preserves internal geometry even across 180-deg rotations."""
    torch.manual_seed(42)
    coords = torch.randn((24, 3), dtype=torch.float64)
    coords = coords - coords.mean(dim=0, keepdim=True)
    rot_180 = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    rotated = coords @ rot_180

    orig_dists = torch.cdist(coords, coords)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        interp = so3_interpolate_coords(coords, rotated, lam)
        assert torch.allclose(torch.cdist(interp, interp), orig_dists, atol=1e-5)


def test_rotational_consensus_average_does_not_shrink_misaligned_subunits() -> None:
    """Consensus across 90-deg and 180-deg rotated copies retains 100% of internal distances."""
    torch.manual_seed(42)
    base = torch.randn((20, 3), dtype=torch.float64)
    base = base - base.mean(dim=0, keepdim=True)
    r90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    r180 = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))

    consensus = rotational_consensus_average([base, base @ r90, base @ r180])
    assert torch.allclose(torch.cdist(consensus, consensus), torch.cdist(base, base), atol=1e-5)


def test_safe_kabsch_rotation_handles_zero_matrix() -> None:
    """Zero covariance matrix returns identity (det = +1) instead of singular det = 0."""
    zeros = torch.zeros((6, 3), dtype=torch.float64)
    rot = safe_kabsch_rotation(zeros, zeros)
    assert math.isclose(float(torch.linalg.det(rot).item()), 1.0, abs_tol=1e-6)


def test_unwrap_cylinder_continuous_across_pi_seam() -> None:
    """Tilted chain crossing phi = +/-pi does not split by 2*pi*R."""
    radius = 15.0
    l_res = 12
    angles = torch.linspace(math.pi - 0.3, math.pi + 0.3, l_res, dtype=torch.float64)
    z_vals = torch.linspace(-6.0, 6.0, l_res, dtype=torch.float64)
    coords = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles), z_vals], dim=1)
    ideal_com = torch.tensor([-radius, 0.0, 0.0], dtype=torch.float64)

    flat = unwrap_cylinder_to_consensus(coords, ideal_com, radius)
    step_dy = torch.abs(flat[1:, 1] - flat[:-1, 1])
    assert float(step_dy.max().item()) < 2.0


def test_pitch_offset_perturbs_helical_and_linear_tape_lattices() -> None:
    """pitch_offset in params actually modulates spacing for helical and linear_tape."""
    c0, _ = generate_ideal_lattice("helical", 4, 1, {"target_dz": 5.0, "pitch_offset": 0.0})
    c1, _ = generate_ideal_lattice("helical", 4, 1, {"target_dz": 5.0, "pitch_offset": 1.0})
    assert not torch.allclose(c0, c1)

    t0, _ = generate_ideal_lattice("linear_tape", 4, 1, {"target_pitch": 4.8, "pitch_offset": 0.0})
    t1, _ = generate_ideal_lattice("linear_tape", 4, 1, {"target_pitch": 4.8, "pitch_offset": 0.5})
    assert not torch.allclose(t0, t1)


def test_kabsch_rmsd_exact_3d_not_sqrt3_understated() -> None:
    """Verify _kabsch_rmsd computes true 3D Euclidean RMSD (not divided by sqrt(3))."""
    torch.manual_seed(42)
    q = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    # Shift one axis of two opposite points symmetrically so centroid stays 0 and optimal R = I
    p = q.clone()
    p[0, 0] += 1.0
    p[1, 0] -= 1.0
    p[2, 1] += 1.0
    p[3, 1] -= 1.0
    # Each of the 4 points is displaced by exactly 1.0 A -> true 3D RMSD is close to 1.0 (not 0.577)
    rmsd = _kabsch_rmsd(p, q)
    # Verify against explicit 3D formula after optimal Kabsch
    assert rmsd > 0.85  # would be ~0.55 if divided by sqrt(3)


def test_cyclic_6mer_arbitrary_chain_permutation_zero_rmsd() -> None:
    """Arbitrarily permuted 6-mer cyclic refold matches ideal assembly with 0.0 A RMSD."""
    torch.manual_seed(42)
    k = 6
    n_at = 16
    angles = torch.arange(k, dtype=torch.float64) * (2.0 * math.pi / k)
    base = torch.randn((n_at, 3), dtype=torch.float64)
    ideal_chains = []
    for a in angles:
        c, s = math.cos(float(a)), math.sin(float(a))
        rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        com = torch.tensor([10.0 * c, 10.0 * s, 0.0], dtype=torch.float64)
        ideal_chains.append(base @ rz.T + com)

    perm = [2, 5, 0, 4, 1, 3]
    # Apply an arbitrary 3D rotation to refolded complex as well
    q_rot, _ = torch.linalg.qr(torch.randn((3, 3), dtype=torch.float64))
    if torch.linalg.det(q_rot) < 0:
        q_rot[:, 2] *= -1
    refold_chains = [ideal_chains[p] @ q_rot + torch.tensor([5.0, -3.0, 2.0]) for p in perm]

    coords_i = torch.cat(ideal_chains, dim=0)
    coords_r = torch.cat(refold_chains, dim=0)
    cids = torch.arange(k).repeat_interleave(n_at)
    bb = torch.ones(k * n_at, dtype=torch.bool)

    rmsd, _ = compute_chain_order_independent_rmsd(
        coords_r, coords_i, cids, cids, bb, bb, topology="cyclic"
    )
    assert rmsd < 1e-4


def test_cyclic_symmetry_parallel_rejects_scrambled_and_axial_tolerance() -> None:
    """Scrambled (+ + + - - -) bundle fails parallel orientation check; 1.2 A off-axis metal passes."""
    radius = 10.0
    n = 6
    angles = np.arange(n) * (2.0 * np.pi / n)
    z_seq = np.linspace(-10.0, 10.0, 12)

    scrambled_chains = []
    for i, a in enumerate(angles):
        z = z_seq if i < 3 else z_seq[::-1]
        ca = np.stack([np.full(12, radius * np.cos(a)), np.full(12, radius * np.sin(a)), z], axis=1)
        scrambled_chains.append(ca)

    res_scrambled = compute_cyclic_symmetry_metrics(
        scrambled_chains, np.zeros((0, 3)), expected_antiparallel=False
    )
    assert res_scrambled["sym_orientation_match"] == 0.0

    parallel_chains = []
    for a in angles:
        ca = np.stack([np.full(12, radius * np.cos(a)), np.full(12, radius * np.sin(a)), z_seq], axis=1)
        parallel_chains.append(ca)
    ligs_near_axis = np.array([[1.2, 0.0, -3.0], [1.2, 0.0, 3.0]])
    res_par = compute_cyclic_symmetry_metrics(
        parallel_chains, ligs_near_axis, expected_antiparallel=False
    )
    assert res_par["sym_orientation_match"] == 1.0
    assert res_par["sym_ligand_orbit_ok"] == 1.0


def test_cylinder_metrics_elongated_multi_tier_barrel() -> None:
    """cylinder_metrics recovers true barrel radius even when axial height >> diameter."""
    radius = 10.0
    ring_size = 6
    tiers = 5
    coms = []
    cas = []
    for t in range(tiers):
        z_t = (t - 2) * 15.0  # total axial span 60 A >> diameter 20 A
        for s in range(ring_size):
            a = 2.0 * math.pi * s / ring_size
            com = [radius * math.cos(a), radius * math.sin(a), z_t]
            coms.append(com)
            for dz in (-2.0, 0.0, 2.0):
                cas.append([radius * math.cos(a), radius * math.sin(a), z_t + dz])
    m = cylinder_metrics(
        torch.tensor(coms, dtype=torch.float64),
        torch.tensor(cas, dtype=torch.float64),
        target_radius=radius,
    )
    assert math.isclose(m["cylinder_mean_radius"], radius, abs_tol=1e-4)
    assert math.isclose(m["cylinder_radius_rmsd"], 0.0, abs_tol=1e-4)


def test_residue_constraints_soft_weight_one_and_negative_weights_no_crash() -> None:
    """Soft weight == 1.0 is not treated as hard ban; negative weights on all AAs do not crash."""
    spec = [
        {"position": 1, "disallowed": "C"},
        {"position": 1, "weights": {"A": 1.0, "C": 2.0}},
        {"position": 2, "weights": {const.prot_token_to_letter[t]: -0.5 for t in const.canonical_tokens}},
    ]
    mask = parse_residue_constraints(
        spec, 2, const.canonical_tokens, const.prot_letter_to_token
    )
    out = build_constraint_logit_mask(
        num_nodes=2,
        aa_constraint_mask=torch.from_numpy(mask),
        inverse_fold_restriction=[],
        canonical_tokens=const.canonical_tokens,
        inf=1e6,
        device=torch.device("cpu"),
    )
    ala_idx = const.canonical_tokens.index("ALA")
    cys_idx = const.canonical_tokens.index("CYS")
    # ALA weight 1.0 stays positive soft bias (not -inf)
    assert out[0, ala_idx].item() > 0.99
    # CYS stayed hard-blocked despite weight 2.0
    assert out[0, cys_idx].item() == -1e6
    # Position 2 with -0.5 on every AA succeeded without ValueError
    assert torch.allclose(out[1], torch.full((20,), -0.5))


def test_filter_nan_rejected_in_cyclic_hard_gates_and_best_dup_kept(tmp_path: Path) -> None:
    """Broken NaN rows must FAIL cyclic hard gates (not be imputed to passing values)."""
    df = pd.DataFrame(
        [
            {
                "id": "good_design",
                "file_name": "good_design.cif",
                "designed_sequence": "AAAAAAAAAA",
                "complex_plddt": 0.90,
                "ligand_iptm": 0.85,
                "ss_accuracy_refolded": 0.95,
                "sym_orientation_match": 1.0,
                "ligand_min_coord_number": 3.0,
                "sym_ligand_rmsd": 0.5,
                "neg_sym_ligand_rmsd": -0.5,
                "sym_chain_rmsd": 0.6,
                "neg_sym_chain_rmsd": -0.6,
                "neg_ligand_coord_geom_dev": -0.1,
                "neg_ligand_coord_bond_dev": -0.1,
                "sym_ligand_orbit_ok": 1.0,
                "bb_rmsd": 1.0,
            },
            {
                "id": "broken_nan",
                "file_name": "broken_nan.cif",
                "designed_sequence": "GGGGGGGGGG",
                "complex_plddt": 0.99,
                "ligand_iptm": 0.99,
                "ss_accuracy_refolded": float("nan"),
                "sym_orientation_match": float("nan"),
                "ligand_min_coord_number": float("nan"),
                "sym_ligand_rmsd": float("nan"),
                "neg_sym_ligand_rmsd": float("nan"),
                "sym_chain_rmsd": float("nan"),
                "neg_sym_chain_rmsd": float("nan"),
                "neg_ligand_coord_geom_dev": float("nan"),
                "neg_ligand_coord_bond_dev": float("nan"),
                "sym_ligand_orbit_ok": float("nan"),
                "bb_rmsd": 1.0,
            },
        ]
    )
    df.to_csv(tmp_path / "aggregate_metrics_0.csv", index=False)

    flt = Filter(design_dir=str(tmp_path), outdir=str(tmp_path / "out"), filter_designfolding=False, filter_biased=False)
    flt.load_dataframe()
    flt.filter_df()
    flt.sort_df()

    good_row = flt.df[flt.df["id"] == "good_design"].iloc[0]
    nan_row = flt.df[flt.df["id"] == "broken_nan"].iloc[0]
    assert bool(good_row["pass_filters"]) is True
    assert bool(nan_row["pass_filters"]) is False
    assert flt.df.iloc[0]["id"] == "good_design"


def test_boltzgen_cli_overrides_peptide_anything_and_allows_empty_avoid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit CLI --refolding_rmsd_threshold and --inverse_fold_avoid '' override peptide-anything defaults."""
    import boltzgen.cli.boltzgen as boltzgen_cli

    monkeypatch.setattr(boltzgen_cli, "get_artifact_path", lambda *_a, **_kw: Path("dummy.ckpt"))
    parser = boltzgen_cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "dummy.yaml",
            "--protocol",
            "peptide-anything",
            "--refolding_rmsd_threshold",
            "4.5",
            "--inverse_fold_avoid",
            "",
        ]
    )
    args.output = tmp_path / "out"
    args.config_dir = Path("test_config")
    args.design_spec = [tmp_path / "dummy.yaml"]
    args.design_checkpoints = ["dummy.ckpt"]

    pipeline = boltzgen_cli.BinderDesignPipeline(args, Path("dummy_moldir"))
    if_step = next(s for s in pipeline.steps if s.name == "inverse_folding")
    assert "override.inverse_fold_args.inverse_fold_restriction=[]" in if_step.args

    filter_step = next(s for s in pipeline.steps if s.name == "filtering")
    rmsd_entries = [a for a in filter_step.args if a.startswith("refolding_rmsd_threshold=")]
    # Last entry wins in Hydra -> must be 4.5, not peptide-anything's 2
    assert rmsd_entries[-1] == "refolding_rmsd_threshold=4.5"


def test_material_builder_defaults_and_yaml_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """double_tape_aromatic_bias defaults to 0.0, target_pitch YAML alias works, CLI overrides YAML, metal allows Cys."""
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import material_builder
    import yaml

    cfg_path = tmp_path / "metal_bundle.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "topology": "cyclic",
                "copies": 6,
                "target_radius": 9.5,
                "target_pitch": 5.2,
                "output_dir": str(tmp_path / "yaml_out"),
                "asym_unit": [
                    {"type": "protein", "name": "helix", "length": 20, "secondary_structure": "H"},
                    {"type": "ligand", "ccd": "CU", "placement": "z_axis", "count": 2},
                ],
            }
        )
    )
    cmd_captured: list[str] = []
    monkeypatch.setattr(
        material_builder.subprocess,
        "run",
        lambda cmd, check=True: cmd_captured.extend(cmd),
    )
    cli_out = tmp_path / "cli_override_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "material_builder.py",
            "--config",
            str(cfg_path),
            "--output_dir",
            str(cli_out),
            "--aa_bias",
            "H:1.0",
            "--run",
        ],
    )
    material_builder.main()

    # CLI --output_dir overrode YAML output_dir
    assert cmd_captured[cmd_captured.index("--output") + 1] == str(cli_out)
    # double_tape_aromatic_bias NOT silently forwarded on cyclic metal bundle
    assert "--double_tape_aromatic_bias" not in cmd_captured
    # Metal ligand allows Cysteine by passing empty --inverse_fold_avoid
    assert "--inverse_fold_avoid" in cmd_captured
    assert cmd_captured[cmd_captured.index("--inverse_fold_avoid") + 1] == ""
    # --aa_bias properly wrapped under --config inverse_folding
    aa_idx = cmd_captured.index("override.inverse_fold_args.aa_bias=H:1.0")
    assert cmd_captured[aa_idx - 2 : aa_idx] == ["--config", "inverse_folding"]
    # Layout persisted top-level topology and target_pitch=5.2
    layout = json.loads((cli_out / "material_layout.json").read_text())
    assert layout["topology"] == "cyclic"
    assert math.isclose(layout["topology_params"]["target_pitch"], 5.2, abs_tol=1e-6)
    assert math.isclose(layout["guidance_params"]["guidance_sigma_lig_taper"], 2.0, abs_tol=1e-6)
    assert math.isclose(layout["guidance_params"]["guidance_sigma_lig_release"], 0.8, abs_tol=1e-6)


def test_l_amino_acid_beta_strand_pleating_orientation() -> None:
    """beta_strand_pleating_direction must match true L-amino-acid CB-CA sidechain parity."""
    from boltzgen.model.modules.materials import (
        align_consensus_to_double_tape,
        beta_strand_pleating_direction,
    )

    # Construct a canonical L-amino-acid beta strand along +Y with ideal tetrahedral
    # backbone geometry (phi=-135 deg, psi=+135 deg) where real CB atoms alternate +/-X.
    # Use standard bond lengths: N-CA=1.458, CA-C=1.525, C-O=1.231, N-CA-C=111.2 deg.
    bb_list = []
    true_cb_dirs = []
    for i in range(9):
        y = i * 3.35
        # Alternating sign places odd 1-indexed (i=0,2,4,6,8) CB along +X before alignment
        s = 1.0 if i % 2 == 0 else -1.0
        n_pt = torch.tensor([-0.25 * s, y - 1.22, -0.75 * s], dtype=torch.float32)
        ca_pt = torch.tensor([0.15 * s, y, 0.0], dtype=torch.float32)
        c_pt = torch.tensor([-0.25 * s, y + 1.25, 0.78 * s], dtype=torch.float32)
        o_pt = torch.tensor([-0.25 * s, y + 1.25, 0.78 * s + 1.23 * s], dtype=torch.float32)
        bb_list.extend([n_pt, ca_pt, c_pt, o_pt])
        b = ca_pt - n_pt
        c = c_pt - ca_pt
        cb_v = -0.58273431 * torch.linalg.cross(b, c) + 0.56802827 * b - 0.54067466 * c
        true_cb_dirs.append(cb_v / cb_v.norm())

    bb_coords = torch.stack(bb_list, dim=0)
    true_cb = torch.stack(true_cb_dirs, dim=0)
    true_odd_minus_even = true_cb[0::2].mean(0) - true_cb[1::2].mean(0)
    true_odd_minus_even = true_odd_minus_even / true_odd_minus_even.norm()

    v_pleat = beta_strand_pleating_direction(bb_coords)
    assert float(torch.dot(v_pleat, true_odd_minus_even)) > 0.99

    # After align_consensus_to_double_tape, odd 1-indexed CBs must face -X (outward)
    # and even 1-indexed CBs must face +X (inward toward bilayer core).
    aligned = align_consensus_to_double_tape(bb_coords)
    bb4 = aligned.view(9, 4, 3)
    b_al = bb4[:, 1] - bb4[:, 0]
    c_al = bb4[:, 2] - bb4[:, 1]
    cb_al = -0.58273431 * torch.linalg.cross(b_al, c_al, dim=-1) + 0.56802827 * b_al - 0.54067466 * c_al
    cb_al = cb_al / cb_al.norm(dim=-1, keepdim=True)
    assert float(cb_al[0::2, 0].mean()) < -0.5, f"Odd CBs must face -X (outward), got {cb_al[0::2, 0]}"
    assert float(cb_al[1::2, 0].mean()) > 0.5, f"Even CBs must face +X (inward), got {cb_al[1::2, 0]}"


def test_antiparallel_anti_correlation_blocks_verbatim_copy() -> None:
    """Antiparallel anti-correlation must prevent chain B from copying non-palindromic chain A."""
    from boltzgen.data import const
    from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder

    torch.manual_seed(42)
    length = 5
    n = 2 * length
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    s = torch.zeros(n, 8)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    z = torch.zeros(n * n, 8)

    # Bias each residue index to prefer a distinct AA on both chains (non-palindromic)
    aa_mask = torch.zeros(1, n, len(const.canonical_tokens), dtype=torch.float32)
    preferred = [0, 1, 2, 3, 4]
    for r, tok_i in enumerate(preferred):
        aa_mask[0, r, tok_i] = 5.0
        aa_mask[0, length + r, tok_i] = 5.0

    feats = {
        "design_mask": torch.ones(1, n, dtype=torch.bool),
        "symmetric_group": torch.tensor([[1] * length + [2] * length], dtype=torch.long),
        "feature_asym_id": torch.tensor([[0] * length + [1] * length], dtype=torch.long),
        "feature_residue_index": torch.tensor([list(range(length)) * 2], dtype=torch.long),
        "token_to_bb4_atoms": torch.eye(n * 4).unsqueeze(0),
        "coords": torch.zeros(1, n * 4, 3),
        "res_type_clone": torch.zeros(1, n, const.num_tokens),
        "aa_constraint_mask": aa_mask,
        "is_antiparallel": torch.tensor([True]),
    }

    decoder = InverseFoldingDecoder(
        atom_s=1,
        atom_z=1,
        token_s=1,
        token_z=1,
        node_dim=8,
        pair_dim=8,
        hidden_dim=8,
        anti_correlation_strength=100.0,
        sampling_temperature=None,
    )
    decoder.eval()
    out = decoder.sample(s, z, edge_idx, valid_mask, feats)
    res_types = out["res_type"][0].argmax(dim=-1).tolist()
    chain_a = res_types[:length]
    chain_b = res_types[length:]
    for i in range(length):
        assert chain_a[i] != chain_b[i], f"Verbatim match at index {i}: A={chain_a}, B={chain_b}"
        assert chain_a[i] != chain_b[length - 1 - i], (
            f"Antiparallel contact match at {i}<->{length - 1 - i}: A={chain_a}, B={chain_b}"
        )
