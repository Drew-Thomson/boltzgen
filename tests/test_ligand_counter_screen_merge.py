"""Tests for ligand-free counter-screen metrics merging and contrast calculations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from boltzgen.task.analyze.merge_counter_screen_metrics import MergeCounterScreenMetrics


def test_merge_counter_screen_metrics_apo(tmp_path, monkeypatch):
    design_dir = tmp_path / "design"
    screen_dir = tmp_path / "screens"
    design_dir.mkdir(parents=True)
    screen_dir.mkdir(parents=True)

    # Set up mapping with apo
    mapping = {
        "candidate_01": {"apo": "candidate_01_apo"},
        "candidate_02": {"apo": "candidate_02_apo"},
    }
    (screen_dir / "counter_screen_mapping.json").write_text(json.dumps(mapping))

    # Set up apo folding prediction NPZs
    # Candidate 01: good holo (plddt=85, iptm=0.8, rmsd=1.5), poor apo (plddt=45, iptm=0.2)
    # Candidate 02: folds equally well in apo (plddt=80, iptm=0.75)
    np.savez_compressed(
        screen_dir / "candidate_01_apo.npz",
        complex_plddt=np.asarray([45.0]),
        protein_iptm=np.asarray([0.20]),
        design_iptm=np.asarray([0.18]),
        min_interaction_pae=np.asarray([18.0]),
    )
    np.savez_compressed(
        screen_dir / "candidate_02_apo.npz",
        complex_plddt=np.asarray([80.0]),
        protein_iptm=np.asarray([0.75]),
        design_iptm=np.asarray([0.70]),
        min_interaction_pae=np.asarray([5.0]),
    )

    # Candidate 01 has apo_rmsd = 12.0, Candidate 02 has apo_rmsd = 1.6
    apo_rmsds = {
        ("candidate_01_apo", "candidate_01"): 12.0,
        ("candidate_02_apo", "candidate_02"): 1.6,
        ("candidate_01", "candidate_01_apo"): 11.5,
        ("candidate_02", "candidate_02_apo"): 0.5,
    }
    monkeypatch.setattr(
        "boltzgen.task.analyze.merge_counter_screen_metrics._compute_cif_backbone_rmsd",
        lambda refold, design, *args, **kwargs: apo_rmsds.get(
            (refold.stem, design.stem), 99.0
        ),
    )

    # Holo metrics in CSV
    csv_path = design_dir / "aggregate_metrics_analyze.csv"
    pd.DataFrame([
        {
            "id": "candidate_01",
            "complex_plddt": 85.0,
            "protein_iptm": 0.80,
            "design_iptm": 0.78,
            "min_interaction_pae": 4.0,
            "rmsd": 1.5,
            "neg_lattice_rmsd_refolded": -1.5,
        },
        {
            "id": "candidate_02",
            "complex_plddt": 82.0,
            "protein_iptm": 0.76,
            "design_iptm": 0.72,
            "min_interaction_pae": 4.5,
            "rmsd": 1.6,
            "neg_lattice_rmsd_refolded": -1.6,
        },
    ]).to_csv(csv_path, index=False)

    task = MergeCounterScreenMetrics(str(design_dir), str(screen_dir))
    task.run()

    df = pd.read_csv(csv_path)
    assert len(df) == 2

    row1 = df[df["id"] == "candidate_01"].iloc[0]
    assert row1["apo_complex_plddt"] == pytest.approx(45.0)
    assert row1["apo_protein_iptm"] == pytest.approx(0.20)
    assert row1["apo_rmsd"] == pytest.approx(12.0)
    assert row1["delta_complex_plddt_vs_apo"] == pytest.approx(40.0)  # 85 - 45
    assert row1["delta_protein_iptm_vs_apo"] == pytest.approx(0.60)   # 0.80 - 0.20
    assert row1["delta_rmsd_vs_apo"] == pytest.approx(10.5)           # 12.0 - 1.5
    assert row1["delta_neg_min_interaction_pae_vs_apo"] == pytest.approx(14.0)  # 18 - 4

    row2 = df[df["id"] == "candidate_02"].iloc[0]
    assert row2["apo_complex_plddt"] == pytest.approx(80.0)
    assert row2["apo_protein_iptm"] == pytest.approx(0.75)
    assert row2["apo_rmsd"] == pytest.approx(1.6)
    assert row2["delta_complex_plddt_vs_apo"] == pytest.approx(2.0)   # 82 - 80
    assert row2["delta_protein_iptm_vs_apo"] == pytest.approx(0.01)   # 0.76 - 0.75
    assert row2["delta_rmsd_vs_apo"] == pytest.approx(0.0)            # 1.6 - 1.6


def test_ligand_counter_screen_ranking_priority():
    """Verify that ligand counter screen rankings prioritize apo divergence over copper-independent folding."""
    from boltzgen.task.filter.filter import Filter
    from boltzgen.task.filter.material_presets import ligand_counter_screen_filtering_overrides

    overrides = ligand_counter_screen_filtering_overrides()

    filter_task = Filter(
        design_dir=".",
        budget=10,
        metrics_override=overrides,
        refolding_rmsd_threshold=5.0,
    )

    # candidate_divergent: folds well with metal (rmsd=1.5), completely falls apart without (apo_rmsd=12.0)
    # candidate_independent: folds equally well without metal (rmsd=1.2, apo_rmsd=1.0, delta_rmsd=-0.2)
    df = pd.DataFrame([
        {
            "id": "candidate_divergent",
            "delta_rmsd_vs_apo": 10.5,
            "holo_apo_rmsd": 11.5,
            "apo_rmsd": 12.0,
            "delta_protein_iptm_vs_apo": 0.60,
            "delta_complex_plddt_vs_apo": 40.0,
            "neg_lattice_rmsd_refolded": -1.5,
            "lattice_rmsd_refolded": 1.5,
            "local_rmsd": 1.5,
            "global_rmsd": 1.5,
            "complex_plddt": 85.0,
            "num_filters_passed": 2,
            "pass_local_rmsd_filter": True,
            "pass_global_rmsd_filter": True,
        },
        {
            "id": "candidate_independent",
            "delta_rmsd_vs_apo": -0.2,
            "holo_apo_rmsd": 0.5,
            "apo_rmsd": 1.0,
            "delta_protein_iptm_vs_apo": 0.01,
            "delta_complex_plddt_vs_apo": 1.0,
            "neg_lattice_rmsd_refolded": -1.2,
            "lattice_rmsd_refolded": 1.2,
            "local_rmsd": 1.2,
            "global_rmsd": 1.2,
            "complex_plddt": 88.0,
            "num_filters_passed": 2,
            "pass_local_rmsd_filter": True,
            "pass_global_rmsd_filter": True,
        },
    ])

    filter_task.df = df
    filter_task.sort_df()

    sorted_ids = filter_task.df.sort_values("max_rank")["id"].tolist()
    assert sorted_ids[0] == "candidate_divergent"
    assert sorted_ids[1] == "candidate_independent"


def _write_backbone_bundle_cif(path: Path, chain_coords: list[np.ndarray]) -> None:
    """Write a multi-chain protein backbone-only MMCIF for RMSD testing."""
    import gemmi

    path.parent.mkdir(parents=True, exist_ok=True)
    st = gemmi.Structure()
    model = gemmi.Model("1")
    for idx, res_coords in enumerate(chain_coords):
        chain = gemmi.Chain(chr(ord("A") + idx))
        for r_idx, xyz in enumerate(res_coords):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(str(r_idx + 1))
            for a_idx, (atom_name, elem) in enumerate(
                [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
            ):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(elem)
                atom.pos = gemmi.Position(*(xyz + np.array([0.3 * a_idx, 0.0, 0.0])))
                res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)
    st.add_model(model)
    st.make_mmcif_document().write_file(str(path))


def _make_ring_bundle(radius: float, n_chains: int = 6, n_res: int = 4) -> list[np.ndarray]:
    """Generate a synthetic cyclic helical bundle of ``n_chains`` chains."""
    chains: list[np.ndarray] = []
    for k in range(n_chains):
        theta = 2.0 * np.pi * k / n_chains
        center = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        pts = np.stack(
            [center + np.array([0.0, 0.0, 1.5 * r]) for r in range(n_res)], axis=0
        )
        chains.append(pts)
    return chains


def test_merge_reads_refold_cif_and_uses_permutation_invariance(tmp_path):
    """Ensure MergeCounterScreenMetrics uses refold_cif/ (not unrefolded input) and full S_K invariance."""
    design_dir = tmp_path / "design"
    screen_dir = tmp_path / "screens"
    (design_dir / "refold_cif").mkdir(parents=True)
    (screen_dir / "refold_cif").mkdir(parents=True)
    (screen_dir / "fold_out_npz").mkdir(parents=True)

    holo_chains = _make_ring_bundle(radius=9.0, n_chains=6, n_res=4)
    # cand_switch: apo expands to radius 15.0 A (genuine conformational change)
    apo_switch_chains = _make_ring_bundle(radius=15.0, n_chains=6, n_res=4)
    # cand_swapped: apo has identical 3D geometry to holo, but chains 0 & 1 are swapped (non-dihedral)
    apo_swapped_chains = [
        holo_chains[1],
        holo_chains[0],
        holo_chains[2],
        holo_chains[3],
        holo_chains[4],
        holo_chains[5],
    ]

    for cid, apo_chains in [
        ("cand_switch", apo_switch_chains),
        ("cand_swapped", apo_swapped_chains),
    ]:
        # Unrefolded input CIFs (identical to holo) sit directly in screen_dir/
        _write_backbone_bundle_cif(design_dir / f"{cid}.cif", holo_chains)
        _write_backbone_bundle_cif(design_dir / "refold_cif" / f"{cid}.cif", holo_chains)
        _write_backbone_bundle_cif(screen_dir / f"{cid}_apo.cif", holo_chains)
        _write_backbone_bundle_cif(
            screen_dir / "refold_cif" / f"{cid}_apo.cif", apo_chains
        )
        np.savez_compressed(
            screen_dir / "fold_out_npz" / f"{cid}_apo.npz",
            complex_plddt=np.asarray([0.82]),
            protein_iptm=np.asarray([0.75]),
            design_iptm=np.asarray([0.0]),
            min_interaction_pae=np.asarray([100000.25]),
        )

    mapping = {
        "cand_switch": {"apo": "cand_switch_apo"},
        "cand_swapped": {"apo": "cand_swapped_apo"},
    }
    (screen_dir / "counter_screen_mapping.json").write_text(json.dumps(mapping))

    csv_path = design_dir / "aggregate_metrics_analyze.csv"
    pd.DataFrame(
        [
            {
                "id": "cand_switch",
                "complex_plddt": 0.85,
                "protein_iptm": 0.78,
                "design_iptm": 0.72,
                "min_interaction_pae": 5.0,
                "neg_lattice_rmsd_refolded": -0.5,
            },
            {
                "id": "cand_swapped",
                "complex_plddt": 0.85,
                "protein_iptm": 0.78,
                "design_iptm": 0.72,
                "min_interaction_pae": 5.0,
                "neg_lattice_rmsd_refolded": -0.5,
            },
        ]
    ).to_csv(csv_path, index=False)

    MergeCounterScreenMetrics(str(design_dir), str(screen_dir)).run()
    df = pd.read_csv(csv_path).set_index("id")

    # Must read refold_cif/cand_switch_apo.cif (expanded 15A), not unrefolded input (0.0)
    assert df.loc["cand_switch", "holo_apo_rmsd"] > 5.0
    assert df.loc["cand_switch", "apo_rmsd"] > 5.0
    # Non-dihedral chain swap with identical 3D shape must be invariant (~0 A)
    assert df.loc["cand_swapped", "holo_apo_rmsd"] == pytest.approx(0.0, abs=1e-4)
    # Sentinel apo_min_interaction_pae >= 1e4 must not produce bogus ~100000 delta
    assert (
        "delta_neg_min_interaction_pae_vs_apo" not in df.columns
        or pd.isna(df.loc["cand_switch", "delta_neg_min_interaction_pae_vs_apo"])
    )


def test_ligand_counter_screen_composes_with_cyclic_symmetry_and_soft_ranks(tmp_path):
    """Filter(ligand_counter_screen=True) soft-ranks folded apo switches and composes cyclic preset."""
    from boltzgen.task.filter.filter import Filter

    df = pd.DataFrame(
        [
            {
                "id": "folded_switch",
                "designed_sequence": "AAAA",
                "file_name": "folded_switch.cif",
                "bb_rmsd": 8.0,  # > 5A template drift, but < 15A cyclic ligand threshold
                "neg_lattice_rmsd_refolded": -8.0,
                "holo_apo_rmsd": 7.5,
                "apo_complex_plddt": 0.85,  # well-folded alternative apo state
                "complex_plddt": 0.82,
                "ligand_iptm": 0.75,
                "mean_ligand_coordination": 4.0,
                "mean_ligand_plddt": 0.80,
                "min_ligand_distance": 12.0,
                "neg_num_ligand_clashes": 0,
                "neg_sym_chain_rmsd": -0.5,
                "sym_chain_rmsd": 0.5,
                "neg_sym_ligand_rmsd": -0.2,
                "sym_ligand_rmsd": 0.2,
                "neg_ligand_coord_geom_dev": -5.0,
                "neg_ligand_coord_bond_dev": -0.1,
                "ligand_coord_bond_dev": 0.1,
                "ligand_min_coord_number": 4.0,
                "sym_ligand_orbit_ok": 1.0,
            },
            {
                "id": "static_complex",
                "designed_sequence": "BBBB",
                "file_name": "static_complex.cif",
                "bb_rmsd": 1.0,
                "neg_lattice_rmsd_refolded": -1.0,
                "holo_apo_rmsd": 0.4,
                "apo_complex_plddt": 0.85,
                "complex_plddt": 0.85,
                "ligand_iptm": 0.78,
                "mean_ligand_coordination": 4.0,
                "mean_ligand_plddt": 0.82,
                "min_ligand_distance": 12.0,
                "neg_num_ligand_clashes": 0,
                "neg_sym_chain_rmsd": -0.5,
                "sym_chain_rmsd": 0.5,
                "neg_sym_ligand_rmsd": -0.2,
                "sym_ligand_rmsd": 0.2,
                "neg_ligand_coord_geom_dev": -5.0,
                "neg_ligand_coord_bond_dev": -0.1,
                "ligand_coord_bond_dev": 0.1,
                "ligand_min_coord_number": 4.0,
                "sym_ligand_orbit_ok": 1.0,
            },
        ]
    )
    csv_path = tmp_path / "aggregate_metrics_analyze.csv"
    df.to_csv(csv_path, index=False)

    filter_task = Filter(
        design_dir=str(tmp_path),
        budget=2,
        from_inverse_folded=True,
        ligand_counter_screen=True,
        filter_designfolding=False,
        filter_biased=False,
        refolding_rmsd_threshold=2.0,
    )
    filter_task.load_dataframe()
    filter_task.filter_df()
    filter_task.sort_df()

    assert filter_task.cyclic_ligand_preset is True
    assert "holo_apo_rmsd" in filter_task.metrics
    assert "neg_sym_chain_rmsd" in filter_task.metrics
    # Both pass relaxed 15A RMSD + cyclic filters (soft-rank only on holo_apo_rmsd)
    assert filter_task.df["pass_filters"].all()
    ranked = filter_task.df.sort_values("final_rank")["id"].tolist()
    assert ranked[0] == "folded_switch"
