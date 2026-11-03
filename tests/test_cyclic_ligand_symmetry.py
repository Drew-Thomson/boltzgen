"""Tests for cyclic metal-ligand coordination and fitted-axis symmetry metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boltzgen.task.analyze.analyze_utils import (
    compute_cyclic_symmetry_metrics,
    compute_ligand_symmetry_metrics,
    coordination_geometry_deviation,
)
from boltzgen.task.filter.material_presets import (
    cyclic_ligand_symmetry_filtering_overrides,
)

TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) * 2.0


def _rot_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _ring_chains(n: int, alternate: bool = False) -> list[np.ndarray]:
    z = np.linspace(-15.0, 15.0, 20)
    base = np.stack([np.full(20, 10.0), np.zeros(20), z], axis=1)
    chains = []
    for i in range(n):
        c = base[::-1] if (alternate and i % 2) else base
        chains.append(c @ _rot_z(360.0 * i / n).T)
    return chains


def test_geometry_deviation_tetrahedron_is_zero_and_agnostic():
    assert coordination_geometry_deviation(TETRA) < 1e-6
    octa = np.vstack([np.eye(3), -np.eye(3)]) * 2.4
    assert coordination_geometry_deviation(octa) < 1e-6
    assert coordination_geometry_deviation(TETRA[:3]) < 20  # vacancy variant
    assert coordination_geometry_deviation(TETRA[:1]) == 90.0


def test_symmetry_perfect_c6_parallel():
    chains = _ring_chains(6)
    ligands = np.array([[0.0, 0, 0]] + [[6.0, 0, 3.0] @ _rot_z(120 * i).T for i in range(3)])
    ligands = ligands[1:]
    out = compute_cyclic_symmetry_metrics(chains, ligands)
    assert out["sym_order"] == 6
    assert out["sym_chain_rmsd"] < 1e-6
    assert out["sym_ligand_rmsd"] < 1e-6
    assert out["sym_ligand_orbit_ok"] == 1.0


def test_symmetry_antiparallel_ring_uses_half_order_and_detects_asymmetry():
    chains = _ring_chains(6, alternate=True)
    ligands = np.array([[6.0, 0, 0] @ _rot_z(120 * i).T for i in range(3)])
    out = compute_cyclic_symmetry_metrics(chains, ligands)
    assert out["sym_order"] == 3
    assert out["sym_chain_rmsd"] < 1e-6
    assert out["sym_ligand_rmsd"] < 1e-6

    ligands_bad = ligands.copy()
    ligands_bad[0] += [2.0, 1.0, 0.0]
    assert compute_cyclic_symmetry_metrics(chains, ligands_bad)["sym_ligand_rmsd"] > 0.5

    missing = compute_cyclic_symmetry_metrics(chains, ligands[:2])
    assert missing["sym_ligand_orbit_ok"] == 0.0


def test_symmetry_too_few_chains():
    assert compute_cyclic_symmetry_metrics(_ring_chains(2), np.zeros((1, 3))) == {}


def _write_pdb_like_cif(path: Path, n_chains: int = 3, perturb: float = 0.0) -> None:
    """Minimal ring of 3 chains; each Zn sits in a tetrahedron of 4 Cys-S/His-N/backbone-O donors."""
    import gemmi

    st = gemmi.Structure()
    model = gemmi.Model("1")
    donor_sets = []
    zn_pos = [np.array([8.0, 0, 0]) @ _rot_z(360.0 / 3 * i).T for i in range(3)]
    for i in range(n_chains):
        ch = gemmi.Chain("ABC"[i])
        res = gemmi.Residue()
        res.name, res.seqid = "CYS", gemmi.SeqId(1, " ")
        for name, el, pos in (
            ("N", "N", [0, 0, 0]),
            ("CA", "C", [0, 0, 1.5]),
            ("SG", "S", None),
        ):
            a = gemmi.Atom()
            a.name, a.element = name, gemmi.Element(el)
            p = np.array(pos if pos is not None else (zn_pos[i] + TETRA[0] / 2 * 2.3 / 1.732 * 1.0))
            if name == "N":
                p = zn_pos[i] + TETRA[1] / np.linalg.norm(TETRA[1]) * 2.05 + perturb
            if name == "CA":
                p = np.array([10.0, 0, 0.0]) @ _rot_z(120 * i).T
            if name == "SG":
                p = zn_pos[i] + TETRA[0] / np.linalg.norm(TETRA[0]) * 2.3
            a.pos = gemmi.Position(*p)
            res.add_atom(a)
        ch.add_residue(res)
        model.add_chain(ch)
    zch = gemmi.Chain("Z")
    for i in range(3):
        r = gemmi.Residue()
        r.name, r.seqid = "ZN", gemmi.SeqId(i + 1, " ")
        r.het_flag = "H"
        a = gemmi.Atom()
        a.name, a.element = "ZN", gemmi.Element("Zn")
        a.pos = gemmi.Position(*zn_pos[i])
        r.add_atom(a)
        zch.add_residue(r)
    model.add_chain(zch)
    st.add_model(model)
    st.make_mmcif_document().write_file(str(path))


def test_ligand_symmetry_metrics_from_cif(tmp_path):
    cif = tmp_path / "x.cif"
    _write_pdb_like_cif(cif)
    out = compute_ligand_symmetry_metrics(cif)
    assert out["ligand_min_coord_number"] == 2.0  # S + N per Zn in this toy model
    assert out["ligand_coord_cn_std"] == 0.0
    assert out["sym_ligand_rmsd"] < 1e-3
    assert out["neg_ligand_coord_bond_dev"] <= 0.0


def test_preset_disables_template_rmsd_and_prioritises_symmetry():
    ov = cyclic_ligand_symmetry_filtering_overrides()
    assert ov["neg_local_rmsd"] is None and ov["neg_global_rmsd"] is None
    assert ov["neg_sym_ligand_rmsd"] == 1 and ov["neg_ligand_coord_geom_dev"] == 1
