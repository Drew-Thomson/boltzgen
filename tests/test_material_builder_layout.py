import json

import pytest

from boltzgen.model.modules.material_layout import (
    build_homomer_counter_screen_entities,
    build_heteromer_counter_screen_specs,
    expand_material_spec,
    write_layout,
)


def test_ligand_multiplicity_expansion_and_parent_mapping():
    entities, layout = expand_material_spec(
        2,
        [
            {"type": "protein", "name": "p", "length": 12},
            {
                "type": "ligand",
                "ccd": "ZN",
                "ligands_per_peptide": 2,
                "attach_to": "p",
            },
            {"type": "ligand", "ccd": "CA", "ligands_per_asym_unit": 3},
        ],
    )

    assert len(entities) == 12
    assert layout["guided_proteins_per_unit"] == 1
    assert [record["role"] for record in layout["chains"][:6]] == [
        "protein",
        "ligand",
        "ligand",
        "ligand",
        "ligand",
        "ligand",
    ]
    for placement in layout["placements"]:
        assert len(placement["protein_asym_indices"]) == 1
        assert len(placement["ligand_asym_indices"]) == 5
    per_peptide = [
        record for record in layout["chains"] if record.get("attachment_scope") == "per_peptide"
    ]
    assert len(per_peptide) == 4
    assert all(
        record["parent_asym_index"] == layout["placements"][record["placement_index"]]["protein_asym_indices"][0]
        for record in per_peptide
    )
    assert all("name" not in item[next(iter(item))] for item in entities)


def test_legacy_ligand_defaults_to_one_per_placement():
    entities, layout = expand_material_spec(
        3,
        [
            {"type": "protein", "length": 10},
            {"type": "ligand", "ccd": "ZN"},
        ],
    )
    assert len(entities) == 6
    assert layout["guided_proteins_per_unit"] == 1
    assert all(len(placement["ligand_asym_indices"]) == 1 for placement in layout["placements"])


def test_fractional_ligands_per_peptide_distributes_ratio_across_placements():
    entities, layout = expand_material_spec(
        8,
        [
            {"type": "protein", "name": "p", "length": 20},
            {
                "type": "ligand",
                "ccd": "ZN",
                "ligands_per_peptide": 0.5,
                "attach_to": "p",
            },
        ],
    )
    assert len(entities) == 12
    ligand_records = [record for record in layout["chains"] if record["role"] == "ligand"]
    assert len(ligand_records) == 4
    assert [record["placement_index"] for record in ligand_records] == [1, 3, 5, 7]
    assert all(
        record["parent_asym_index"]
        == layout["placements"][record["placement_index"]]["protein_asym_indices"][0]
        for record in ligand_records
    )


def test_fractional_ligand_ratio_requires_integral_total():
    with pytest.raises(ValueError, match="integer ligand count"):
        expand_material_spec(
            3,
            [
                {"type": "protein", "name": "p", "length": 12},
                {
                    "type": "ligand",
                    "ccd": "ZN",
                    "ligands_per_peptide": 0.5,
                    "attach_to": "p",
                },
            ],
        )


def test_attach_to_can_select_a_named_protein_template():
    _, layout = expand_material_spec(
        1,
        [
            {"type": "protein", "name": "a", "length": 10},
            {"type": "protein", "name": "b", "length": 12},
            {"type": "ligand", "ccd": "ZN", "ligands_per_peptide": 1, "attach_to": "b"},
        ],
    )
    ligand = next(record for record in layout["chains"] if record["role"] == "ligand")
    assert ligand["parent_protein_slot_index"] == 1
    assert ligand["parent_asym_index"] == layout["placements"][0]["protein_asym_indices"][1]


@pytest.mark.parametrize(
    "unit, message",
    [
        (
            [
                {"type": "protein"},
                {"type": "ligand", "ccd": "ZN", "ligands_per_peptide": 1, "ligands_per_asym_unit": 2},
            ],
            "both",
        ),
        ([{"type": "protein"}, {"type": "ligand", "ccd": "ZN", "ligands_per_peptide": 0}], "positive integer"),
        (
            [
                {"type": "protein", "name": "a"},
                {"type": "protein", "name": "b"},
                {"type": "ligand", "ccd": "ZN", "ligands_per_peptide": 1},
            ],
            "attach_to is required",
        ),
        (
            [
                {"type": "protein", "name": "a"},
                {"type": "ligand", "ccd": "ZN", "ligands_per_peptide": 1, "attach_to": "missing"},
            ],
            "does not name",
        ),
        ([{"type": "protein"}, {"type": "ligand", "ccd": "ZN", "attach_to": "protein"}], "only valid"),
    ],
)
def test_invalid_ligand_multiplicity_is_rejected(unit, message):
    with pytest.raises(ValueError, match=message):
        expand_material_spec(2, unit)


def test_layout_sidecar_round_trips(tmp_path):
    _, layout = expand_material_spec(1, [{"type": "protein", "length": 10}])
    path = write_layout(layout, tmp_path / "material_layout.json")
    loaded = json.loads((tmp_path / "material_layout.json").read_text())
    assert path.endswith("material_layout.json")
    assert loaded == layout


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        ("cyclic", ["A", "B", "A", "B"]),
        ("linear_tape", ["A", "B", "A", "B"]),
        ("open_arc", ["A", "B", "A", "B"]),
        ("helical", ["A", "B", "A", "B"]),
        ("double_tape", ["A", "A", "B", "B", "A", "A", "B", "B"]),
    ],
)
def test_heteromer_layout_alternates_partner_identity(topology, expected):
    copies = len(expected)
    entities, layout = expand_material_spec(
        copies,
        [
            {"type": "protein", "name": "partner_a", "length": 8},
            {"type": "protein", "name": "partner_b", "length": 9},
        ],
        heteromer_screening=True,
        topology=topology,
    )
    assert len(entities) == copies
    assert [record["partner_label"] for record in layout["chains"]] == expected
    assert layout["guided_proteins_per_unit"] == 1
    assert layout["heteromer_screening"]["enabled"] is True


@pytest.mark.parametrize(
    ("copies", "topology", "protein_count", "message"),
    [
        (4, "cyclic", 1, "exactly two"),
        (3, "linear_tape", 2, "even number"),
        (4, "nanotube", 2, "supports only"),
    ],
)
def test_heteromer_layout_rejects_unsupported_inputs(
    copies, topology, protein_count, message
):
    with pytest.raises(ValueError, match=message):
        expand_material_spec(
            copies,
            [{"type": "protein", "length": 8} for _ in range(protein_count)],
            heteromer_screening=True,
            topology=topology,
        )


def test_heteromer_screening_rejects_ligands_as_ternary_systems():
    with pytest.raises(ValueError, match="unsuitable.*ligands"):
        expand_material_spec(
            4,
            [
                {"type": "protein", "name": "A", "length": 8},
                {"type": "protein", "name": "B", "length": 9},
                {"type": "ligand", "ccd": "ZN", "ligands_per_asym_unit": 1},
            ],
            heteromer_screening=True,
            topology="cyclic",
        )


def test_build_homomer_counter_screen_entities():
    entities, mapping = build_homomer_counter_screen_entities(
        "ACDE", 4, partner_label="B"
    )
    assert [entity["protein"]["id"] for entity in entities] == ["A", "B", "C", "D"]
    assert all(entity["protein"]["sequence"] == "ACDE" for entity in entities)
    assert mapping["partner_label"] == "B"
    assert mapping["copies"] == 4


def test_build_heteromer_counter_screen_specs_preserves_parent_and_params():
    jobs = build_heteromer_counter_screen_specs(
        {"A": "ACDE", "B": "FGHI"},
        6,
        "helical",
        {"target_radius": 12.0, "target_dz": 5.0},
        "design_001",
    )
    assert jobs["A"]["id"] == "design_001__homomer_a"
    assert jobs["B"]["id"] == "design_001__homomer_b"
    assert jobs["A"]["parent_design_id"] == "design_001"
    assert len(jobs["A"]["entities"]) == len(jobs["B"]["entities"]) == 6
    assert jobs["A"]["entities"][0]["protein"]["sequence"] == "ACDE"
    assert jobs["B"]["entities"][0]["protein"]["sequence"] == "FGHI"
    assert jobs["A"]["topology_params"] == {"target_radius": 12.0, "target_dz": 5.0}


@pytest.mark.parametrize(
    ("sequences", "copies", "topology", "error"),
    [
        ({"A": "AC", "B": "FG", "C": "HI"}, 4, "cyclic", "exactly keys"),
        ({"A": "AC", "B": "FG"}, 3, "cyclic", "even integer"),
        ({"A": "AC", "B": "FG"}, 4, "nanotube", "Unsupported"),
    ],
)
def test_build_heteromer_counter_screen_specs_rejects_invalid_inputs(
    sequences, copies, topology, error
):
    with pytest.raises(ValueError, match=error):
        build_heteromer_counter_screen_specs(
            sequences, copies, topology, {}, "parent"
        )
