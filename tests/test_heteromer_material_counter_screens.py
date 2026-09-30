"""Tests for deterministic homomer counter-screen construction helpers."""

from __future__ import annotations

import pytest

from boltzgen.model.modules.material_layout import (
    build_heteromer_counter_screen_specs,
    expand_material_spec,
)


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        ("cyclic", ["A", "B", "A", "B"]),
        ("linear_tape", ["A", "B", "A", "B"]),
        ("double_tape", ["A", "A", "B", "B", "A", "A", "B", "B"]),
        ("open_arc", ["A", "B", "A", "B"]),
        ("helical", ["A", "B", "A", "B"]),
    ],
)
def test_layout_partner_pattern(topology: str, expected: list[str]) -> None:
    _, layout = expand_material_spec(
        len(expected),
        [
            {"type": "protein", "name": "A", "length": 12},
            {"type": "protein", "name": "B", "length": 12},
        ],
        heteromer_screening=True,
        topology=topology,
    )
    assert [record["partner_label"] for record in layout["chains"]] == expected
    if topology == "double_tape":
        side_labels: dict[int, list[str]] = {}
        for record in layout["chains"]:
            side_labels.setdefault(record["double_tape_side"], []).append(
                record["partner_label"]
            )
        assert all(labels in (["A", "B", "A", "B"], ["B", "A", "B", "A"]) for labels in side_labels.values())


def test_counter_screen_jobs_keep_topology_and_parent_link() -> None:
    jobs = build_heteromer_counter_screen_specs(
        {"A": "ACDE", "B": "FGHI"},
        10,
        "cyclic",
        {"target_radius": 18.0},
        "candidate_0001",
    )
    assert jobs["A"]["id"] == "candidate_0001__homomer_a"
    assert jobs["B"]["id"] == "candidate_0001__homomer_b"
    assert jobs["A"]["parent_design_id"] == jobs["B"]["parent_design_id"] == "candidate_0001"
    assert len(jobs["A"]["entities"]) == len(jobs["B"]["entities"]) == 10
    assert jobs["A"]["entities"][0]["protein"]["sequence"] == "ACDE"
    assert jobs["B"]["entities"][0]["protein"]["sequence"] == "FGHI"
    assert jobs["A"]["topology_params"] == {"target_radius": 18.0}


def test_counter_screen_jobs_reject_odd_copies() -> None:
    with pytest.raises(ValueError, match="even"):
        build_heteromer_counter_screen_specs(
            {"A": "ACDE", "B": "FGHI"}, 9, "linear_tape", {}, "candidate"
        )


def test_ligand_bearing_heteromer_is_not_eligible_for_screening():
    with pytest.raises(ValueError, match="unsuitable.*ligands"):
        expand_material_spec(
            4,
            [
                {"type": "protein", "name": "A", "length": 12},
                {"type": "protein", "name": "B", "length": 12},
                {"type": "ligand", "ccd": "ZN"},
            ],
            heteromer_screening=True,
            topology="linear_tape",
        )
