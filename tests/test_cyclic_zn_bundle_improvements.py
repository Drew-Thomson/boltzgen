"""Tests for cyclic Zn-binding peptide bundle guidance, folding, D3 flagging, and counter-screen fixes."""

# ruff: noqa: INP001, PLR2004, E501

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from boltzgen.model.modules.materials import (
    compute_anchored_ligand_position,
    enforce_minimum_ligand_separation,
    generate_ideal_lattice,
    symmetrize_cyclic_ligand_positions,
)
from boltzgen.task.analyze.analyze_utils import (
    compute_cyclic_symmetry_metrics,
    compute_ligand_symmetry_metrics,
    get_best_folding_sample,
)
from boltzgen.task.analyze.merge_counter_screen_metrics import _load_prediction_metrics


def _rot_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]
    )


def _antiparallel_ring_ca(n: int = 6, radius: float = 9.0, length: int = 20) -> list[np.ndarray]:
    z = np.linspace(-14.0, 14.0, length)
    base = np.stack([np.full(length, radius), np.zeros(length), z], axis=1)
    chains: list[np.ndarray] = []
    for i in range(n):
        c = base[::-1] if (i % 2 == 1) else base
        chains.append(c @ _rot_z(360.0 * i / n).T)
    return chains


def test_symmetrize_cyclic_ligand_positions_c3_antiparallel() -> None:
    """3 asymmetric Zn ions on antiparallel 6-mer (parents B, D, F) project to exact C3 symmetry."""
    torch.manual_seed(42)
    ideal_coms, ideal_rots = generate_ideal_lattice(
        "cyclic", 6, 1, {"target_radius": 9.0}, dtype=torch.float64
    )
    p_mean = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    q_mean = ideal_coms.mean(dim=0)
    r_kabsch = torch.eye(3, dtype=torch.float64)

    # Three asymmetric Zn coordinates in global frame
    ligs = [
        torch.tensor([[5.0, 8.0, 7.0]], dtype=torch.float64),
        torch.tensor([[-10.0, 1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[4.0, -6.0, 11.0]], dtype=torch.float64),
    ]
    sym_ligs = symmetrize_cyclic_ligand_positions(
        ligs,
        parent_indices=[1, 3, 5],
        ideal_coms=ideal_coms,
        ideal_rotations=ideal_rots,
        p_mean=p_mean,
        q_mean=q_mean,
        r_kabsch=r_kabsch,
        is_antiparallel=True,
        asym_unit_size=1,
    )
    assert len(sym_ligs) == 3
    centers = np.stack([(x[0] - p_mean).numpy() for x in sym_ligs], axis=0)
    # All 3 must share the exact same radial distance and axial height z
    radii = np.linalg.norm(centers[:, :2], axis=1)
    assert np.allclose(radii, radii[0], atol=1e-5)
    assert np.allclose(centers[:, 2], centers[0, 2], atol=1e-5)
    # 120-deg rotation must map the 3 sites onto themselves
    r120 = _rot_z(120.0)
    rot_c = centers @ r120.T
    dists = np.linalg.norm(rot_c[:, None, :] - centers[None, :, :], axis=-1)
    assert float(dists.min(axis=1).max()) < 1e-5


def test_symmetrize_cyclic_ligand_positions_d3_six_metals() -> None:
    """6 Zn ions on antiparallel 6-mer (parents A..F) with antiparallel flips recover D3 (+z/-z) symmetry."""
    ideal_coms, ideal_rots = generate_ideal_lattice(
        "cyclic", 6, 1, {"target_radius": 9.0}, dtype=torch.float64
    )
    zero = torch.zeros(3, dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    # Place even parents near +5 A and odd parents near -5 A with slight noise
    ligs = []
    for i in range(6):
        ang = 2.0 * math.pi * i / 6.0
        z_i = 5.0 if i % 2 == 0 else -5.0
        ligs.append(
            torch.tensor(
                [[8.0 * math.cos(ang), 8.0 * math.sin(ang), z_i + 0.3 * (i % 3)]],
                dtype=torch.float64,
            )
        )
    sym_ligs = symmetrize_cyclic_ligand_positions(
        ligs,
        parent_indices=list(range(6)),
        ideal_coms=ideal_coms,
        ideal_rotations=ideal_rots,
        p_mean=zero,
        q_mean=zero,
        r_kabsch=eye,
        is_antiparallel=True,
    )
    zs = [float(x[0, 2].item()) for x in sym_ligs]
    assert math.isclose(zs[0], -zs[1], abs_tol=1e-5)
    assert math.isclose(sum(zs), 0.0, abs_tol=1e-5)


def test_compute_anchored_ligand_position_interface_and_inward() -> None:
    """'interface' places ligand between adjacent helices without overshooting; 'inward' moves toward lumen."""
    ca = torch.tensor([9.0, 0.0, 4.0], dtype=torch.float32)
    nbr_ca = torch.tensor([4.5, 7.7942, 4.0], dtype=torch.float32)  # 60 deg away on R=9 ring
    com = torch.zeros(3, dtype=torch.float32)

    pos_iface = compute_anchored_ligand_position(
        ca_coords=ca,
        assembly_com=com,
        topology="cyclic",
        anchor_distance=3.5,
        direction_mode="interface",
        neighbor_ca_coords=nbr_ca,
    )
    # Should lie on segment between ca and nbr_ca at distance 3.5 A from ca (midpoint is 4.5 A)
    assert math.isclose(float(torch.norm(pos_iface - ca).item()), 3.5, abs_tol=1e-4)
    assert float(torch.norm(pos_iface - nbr_ca).item()) < float(torch.norm(ca - nbr_ca).item())

    # If anchor_distance exceeds half the inter-helix gap, clamp at midpoint
    pos_clamped = compute_anchored_ligand_position(
        ca_coords=ca,
        assembly_com=com,
        topology="cyclic",
        anchor_distance=10.0,
        direction_mode="interface",
        neighbor_ca_coords=nbr_ca,
    )
    midpoint = 0.5 * (ca + nbr_ca)
    assert torch.allclose(pos_clamped, midpoint, atol=1e-4)

    # 'inward' moves radially toward the z-axis (r = 9 - 3.5 = 5.5)
    pos_in = compute_anchored_ligand_position(
        ca_coords=ca,
        assembly_com=com,
        topology="cyclic",
        anchor_distance=3.5,
        direction_mode="inward",
    )
    assert torch.allclose(pos_in, torch.tensor([5.5, 0.0, 4.0]), atol=1e-4)


def test_enforce_minimum_ligand_separation_3d() -> None:
    """Clashing non-axial ligands are pushed apart in 3D to >= min_distance."""
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]], dtype=torch.float32
    )
    sep = enforce_minimum_ligand_separation(pts, min_distance=3.5)
    dmat = torch.cdist(sep, sep)
    mask = ~torch.eye(3, dtype=torch.bool)
    assert float(dmat[mask].min().item()) >= 3.49


def test_best_sample_selection_skips_zero_design_iptm(tmp_path: Path) -> None:
    """When design_iptm is identically 0 (all-designed apo), pick sample with highest protein_iptm."""
    npz_path = tmp_path / "cand_apo.npz"
    coords = np.zeros((5, 4, 3), dtype=np.float32)
    for idx in range(5):
        coords[idx] = float(idx)
    design_iptm = np.zeros(5, dtype=np.float32)
    protein_iptm = np.array([0.15, 0.25, 0.30, 0.88, 0.40], dtype=np.float32)
    complex_plddt = np.array([0.40, 0.50, 0.55, 0.91, 0.60], dtype=np.float32)
    ptm = np.array([0.20, 0.30, 0.35, 0.85, 0.45], dtype=np.float32)

    np.savez_compressed(
        npz_path,
        coords=coords,
        design_iptm=design_iptm,
        protein_iptm=protein_iptm,
        complex_plddt=complex_plddt,
        ptm=ptm,
        min_interaction_pae=np.array([15.0, 12.0, 11.0, 2.5, 9.0], dtype=np.float32),
    )

    loaded = _load_prediction_metrics(npz_path)
    assert loaded["protein_iptm"] == pytest.approx(0.88, abs=1e-5)
    assert loaded["complex_plddt"] == pytest.approx(0.91, abs=1e-5)
    assert loaded["min_interaction_pae"] == pytest.approx(2.5, abs=1e-5)

    with np.load(npz_path) as data:
        best = get_best_folding_sample(data)
    assert float(best["coords"][0, 0]) == pytest.approx(3.0, abs=1e-5)
    assert float(best["protein_iptm"]) == pytest.approx(0.88, abs=1e-5)


def test_de_novo_material_disables_target_templates_and_expands_cs_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Material designs disable target_templates in folding and save full keys in counter-screen folding."""
    import boltzgen.cli.boltzgen as boltzgen_cli

    spec = tmp_path / "material_spec.yaml"
    spec.write_text("entities: []\n")
    (tmp_path / "material_layout.json").write_text(
        json.dumps({"copies": 6, "topology": "cyclic"})
    )
    monkeypatch.setattr(
        boltzgen_cli.torch.cuda, "get_device_capability", lambda: (8, 0)
    )
    monkeypatch.setattr(
        boltzgen_cli,
        "get_artifact_path",
        lambda *_a, **_kw: Path("dummy.ckpt"),
    )

    parser = boltzgen_cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            str(spec),
            "--protocol",
            "peptide-anything",
            "--ligand_counter_screen",
            "--use_kernels",
            "false",
        ]
    )
    args.output = tmp_path / "out"
    args.config_dir = Path("test_config")
    args.design_checkpoints = ["dummy.ckpt"]

    pipeline = boltzgen_cli.BinderDesignPipeline(args, Path("dummy_moldir"))
    fold_step = next(s for s in pipeline.steps if s.name == "folding")
    assert "data.target_templates=false" in fold_step.args

    cs_fold_step = next(s for s in pipeline.steps if s.name == "counter_screen_folding")
    assert "data.target_templates=false" in cs_fold_step.args
    keys_arg = next(a for a in cs_fold_step.args if a.startswith("keys_dict_out="))
    for k in ("protein_iptm", "iptm", "ptm", "ligand_iptm", "complex_plddt"):
        assert k in keys_arg


def _write_antiparallel_zn_bundle_cif(
    path: Path,
    zn_positions: list[np.ndarray],
    donor_z_levels: tuple[float, ...] = (7.0, -7.0),
) -> None:
    """Write a synthetic D3-symmetric antiparallel 6-helix bundle with Cys/His donors and Zn ions."""
    import gemmi

    st = gemmi.Structure()
    model = gemmi.Model("1")
    chains_ca = _antiparallel_ring_ca(n=6, radius=9.0, length=10)
    for i, ca_pts in enumerate(chains_ca):
        ch = gemmi.Chain(chr(ord("A") + i))
        for r_idx, ca in enumerate(ca_pts):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(r_idx + 1, " ")
            a_ca = gemmi.Atom()
            a_ca.name, a_ca.element = "CA", gemmi.Element("C")
            a_ca.pos = gemmi.Position(*ca)
            res.add_atom(a_ca)
            ch.add_residue(res)
        # Add coordinating CYS SG atoms near each specified z-level at the inter-helical site
        ang_mid = np.deg2rad(60.0 * (i // 2 * 2) + 30.0)
        for d_idx, z_lvl in enumerate(donor_z_levels):
            res = gemmi.Residue()
            res.name = "CYS"
            res.seqid = gemmi.SeqId(20 + d_idx, " ")
            offset = 1.5 if (i % 2 == 0) else -1.5
            site = np.array([7.5 * np.cos(ang_mid), 7.5 * np.sin(ang_mid), z_lvl])
            tang = np.array([-np.sin(ang_mid), np.cos(ang_mid), 0.0])
            sg = gemmi.Atom()
            sg.name, sg.element = "SG", gemmi.Element("S")
            sg.pos = gemmi.Position(*(site + offset * tang))
            res.add_atom(sg)
            ch.add_residue(res)
        model.add_chain(ch)

    zch = gemmi.Chain("Z")
    for idx, zp in enumerate(zn_positions):
        r = gemmi.Residue()
        r.name, r.seqid = "ZN", gemmi.SeqId(idx + 1, " ")
        r.het_flag = "H"
        a = gemmi.Atom()
        a.name, a.element = "ZN", gemmi.Element("Zn")
        a.pos = gemmi.Position(*zp)
        r.add_atom(a)
        zch.add_residue(r)
    model.add_chain(zch)
    st.add_model(model)
    st.make_mmcif_document().write_file(str(path))


def test_d3_broken_polar_ligands_flag_and_double_suggestion(tmp_path: Path) -> None:
    """Polar single-end 3-Zn cluster flags D3 breaking & suggests 6 metals; equatorial 3-Zn and balanced 6-Zn do not."""
    chains = _antiparallel_ring_ca(n=6, radius=9.0, length=20)
    angles = [np.deg2rad(120.0 * i + 30.0) for i in range(3)]

    # 1. Equatorial 3-Zn (z=0): preserves D3 symmetry
    eq_zn = np.array([[7.5 * np.cos(a), 7.5 * np.sin(a), 0.0] for a in angles])
    res_eq = compute_cyclic_symmetry_metrics(chains, eq_zn, expected_antiparallel=True)
    assert res_eq["sym_d3_chain_rmsd"] < 1e-4
    assert res_eq["sym_d3_ligand_rmsd"] < 1e-4
    assert res_eq["sym_d3_broken_polar_ligands"] == 0.0
    assert res_eq["sym_suggest_double_ligands"] == 0.0

    # 2. Polar single-end 3-Zn (all at z=+7 A): breaks D3, flags double-ligand recovery
    polar_zn = [np.array([7.5 * np.cos(a), 7.5 * np.sin(a), 7.0]) for a in angles]
    polar_cif = tmp_path / "polar_3zn.cif"
    _write_antiparallel_zn_bundle_cif(polar_cif, polar_zn, donor_z_levels=(7.0, -7.0))
    res_polar = compute_ligand_symmetry_metrics(polar_cif, expected_antiparallel=True)
    assert res_polar["sym_ligand_rmsd"] < 1e-3
    assert res_polar["sym_d3_chain_rmsd"] < 1e-3
    assert res_polar["sym_d3_ligand_rmsd"] > 10.0
    assert res_polar["sym_ligand_axial_centroid"] == pytest.approx(7.0, abs=0.1)
    assert res_polar["sym_d3_broken_polar_ligands"] == 1.0
    assert res_polar["sym_suggest_double_ligands"] == 1.0
    assert res_polar["sym_d3_vacant_site_count"] == 3.0
    assert res_polar["sym_d3_vacant_site_min_donors"] >= 2.0

    # 3. Balanced 6-Zn (3 at +7 A, 3 at -7 A): recovers full D3 symmetry
    six_zn = polar_zn + [np.array([7.5 * np.cos(a), 7.5 * np.sin(a), -7.0]) for a in angles]
    six_cif = tmp_path / "balanced_6zn.cif"
    _write_antiparallel_zn_bundle_cif(six_cif, six_zn, donor_z_levels=(7.0, -7.0))
    res_six = compute_ligand_symmetry_metrics(six_cif, expected_antiparallel=True)
    assert res_six["sym_d3_ligand_rmsd"] < 1e-3
    assert res_six["sym_d3_broken_polar_ligands"] == 0.0
    assert res_six["sym_suggest_double_ligands"] == 0.0
    assert res_six["sym_d3_vacant_site_count"] == 0.0


def test_coordination_bridging_and_residue_counts(tmp_path: Path) -> None:
    """Distinguish single-chain bidentate Asp + backbone N from 2-chain sidechain bridging."""
    import gemmi

    # Case A: Single-chain bidentate ASP (2 O atoms, 1 residue, 1 chain) + nearby backbone N
    st_single = gemmi.Structure()
    model = gemmi.Model("1")
    zn_pos = [np.array([7.0, 0.0, 0.0]) @ _rot_z(120.0 * i).T for i in range(3)]
    for i in range(3):
        ch = gemmi.Chain("ABC"[i])
        res = gemmi.Residue()
        res.name, res.seqid = "ASP", gemmi.SeqId(1, " ")
        for aname, el, pos in (
            ("CA", "C", np.array([10.0, 0.0, 0.0]) @ _rot_z(120.0 * i).T),
            ("N", "N", zn_pos[i] + np.array([0.0, 0.0, 2.1])),  # backbone amide N
            ("OD1", "O", zn_pos[i] + np.array([2.0, 0.0, 0.0])),
            ("OD2", "O", zn_pos[i] + np.array([0.0, 2.0, 0.0])),
        ):
            a = gemmi.Atom()
            a.name, a.element = aname, gemmi.Element(el)
            a.pos = gemmi.Position(*pos)
            res.add_atom(a)
        ch.add_residue(res)
        model.add_chain(ch)
    zch = gemmi.Chain("Z")
    for i, zp in enumerate(zn_pos):
        r = gemmi.Residue()
        r.name, r.seqid = "ZN", gemmi.SeqId(i + 1, " ")
        r.het_flag = "H"
        a = gemmi.Atom()
        a.name, a.element = "ZN", gemmi.Element("Zn")
        a.pos = gemmi.Position(*zp)
        r.add_atom(a)
        zch.add_residue(r)
    model.add_chain(zch)
    st_single.add_model(model)
    single_cif = tmp_path / "single_chain_asp.cif"
    st_single.make_mmcif_document().write_file(str(single_cif))

    m_single = compute_ligand_symmetry_metrics(single_cif)
    assert m_single["ligand_min_coord_number"] == 3.0  # N + OD1 + OD2
    assert m_single["ligand_min_sidechain_coord_number"] == 2.0  # excludes backbone N
    assert m_single["ligand_min_coord_residues"] == 1.0
    assert m_single["ligand_min_coord_n_chains"] == 1.0

    # Case B: Two-chain bridging CYS SG donors (from polar_3zn fixture)
    polar_zn = [
        np.array([7.5 * np.cos(np.deg2rad(120.0 * i + 30.0)), 7.5 * np.sin(np.deg2rad(120.0 * i + 30.0)), 7.0])
        for i in range(3)
    ]
    bridged_cif = tmp_path / "bridged.cif"
    _write_antiparallel_zn_bundle_cif(bridged_cif, polar_zn, donor_z_levels=(7.0,))
    m_bridged = compute_ligand_symmetry_metrics(bridged_cif)
    assert m_bridged["ligand_min_coord_n_chains"] == 2.0
    assert m_bridged["ligand_min_coord_residues"] == 2.0
    assert m_bridged["ligand_min_sidechain_coord_number"] == 2.0


def test_cyclic_zn_6mer_yaml_material_builder_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cyclic_AP_Zn_6mer.yaml configures 50x5=250 designs, max_target_candidates=30, Cys allowed, interface anchor."""
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import material_builder

    cfg_path = root_dir / "cyclic_AP_Zn_6mer.yaml"
    out_dir = tmp_path / "zn_dry_run"
    cmd_captured: list[str] = []
    monkeypatch.setattr(
        material_builder.subprocess,
        "run",
        lambda cmd, check=True: cmd_captured.extend(cmd),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "material_builder.py",
            "--config",
            str(cfg_path),
            "--output_dir",
            str(out_dir),
            "--run",
        ],
    )
    material_builder.main()

    assert cmd_captured[cmd_captured.index("--num_designs") + 1] == "50"
    assert cmd_captured[cmd_captured.index("--inverse_fold_num_sequences") + 1] == "5"
    assert "--ligand_counter_screen" in cmd_captured
    assert cmd_captured[cmd_captured.index("--max_target_candidates") + 1] == "30"
    assert cmd_captured[cmd_captured.index("--min_ligand_distance") + 1] == "3.5"
    assert cmd_captured[cmd_captured.index("--inverse_fold_avoid") + 1] == "M"
    assert "--target_max_rmsd" not in cmd_captured

    layout = json.loads((out_dir / "material_layout.json").read_text())
    assert layout["topology"] == "cyclic"
    assert layout["copies"] == 6
    lig_chains = [c for c in layout["chains"] if c["role"] == "ligand"]
    assert len(lig_chains) == 3
    assert [c["placement_index"] for c in lig_chains] == [1, 3, 5]
    for c in lig_chains:
        assert c["anchor_residue"] == "random"
        assert c["anchor_direction"] == "interface"
        assert math.isclose(c["anchor_distance"], 3.5, abs_tol=1e-6)
