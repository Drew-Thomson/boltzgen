"""Tests for opt-in heteromer contrast filtering presets."""

from __future__ import annotations

from boltzgen.task.filter.material_presets import heteromer_filtering_overrides


def test_heteromer_preset_ranks_assembly_contrasts():
    overrides = heteromer_filtering_overrides()
    assert "delta_design_iptm_vs_homomer_max" in overrides
    assert "delta_complex_plddt_vs_homomer_max" in overrides
    assert all(weight > 0 for weight in overrides.values())
