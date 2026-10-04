"""Integration tests for inverse-folding constraint mask composition."""

import pytest

torch = pytest.importorskip("torch")

from boltzgen.data import const
from boltzgen.model.modules.inverse_fold import build_constraint_logit_mask


INF = 10**6


def _allowed_only_mask(allowed_tokens: list[str]) -> torch.Tensor:
    """Build a single-row mask where only `allowed_tokens` are permitted."""
    num_aa = len(const.canonical_tokens)
    mask = torch.ones((1, num_aa), dtype=torch.float32)
    for token in allowed_tokens:
        mask[0, const.canonical_tokens.index(token)] = 0.0
    return mask


def test_conflict_allowed_and_global_avoid_keeps_global_restriction() -> None:
    # If per-residue mask only allows CYS, but global avoid blocks CYS,
    # the position has no valid amino acids, which now raises a ValueError.
    aa_constraint_mask = _allowed_only_mask(["CYS"])

    with pytest.raises(ValueError, match="no valid amino acids at token positions"):
        build_constraint_logit_mask(
            num_nodes=1,
            aa_constraint_mask=aa_constraint_mask,
            inverse_fold_restriction=["CYS"],
            canonical_tokens=const.canonical_tokens,
            inf=INF,
            device=torch.device("cpu"),
        )


def test_non_conflicting_constraints_compose_correctly() -> None:
    ala_idx = const.canonical_tokens.index("ALA")
    cys_idx = const.canonical_tokens.index("CYS")
    aa_constraint_mask = _allowed_only_mask(["ALA"])

    out = build_constraint_logit_mask(
        num_nodes=1,
        aa_constraint_mask=aa_constraint_mask,
        inverse_fold_restriction=["CYS"],
        canonical_tokens=const.canonical_tokens,
        inf=INF,
        device=torch.device("cpu"),
    )

    # Only ALA should remain available.
    assert out[0, ala_idx].item() == 0.0
    assert out[0, cys_idx].item() == -INF
    assert (out[0] == 0).sum().item() == 1


def test_global_restrictions_that_block_all_raise() -> None:
    with pytest.raises(ValueError, match="no valid amino acids"):
        build_constraint_logit_mask(
            num_nodes=1,
            aa_constraint_mask=None,
            inverse_fold_restriction=const.canonical_tokens,
            canonical_tokens=const.canonical_tokens,
            inf=INF,
            device=torch.device("cpu"),
        )


def test_shape_mismatch_ignores_per_residue_mask() -> None:
    bad_shape = torch.zeros((2, 20), dtype=torch.float32)

    # Note: Currently it silently ignores the bad shape.
    out = build_constraint_logit_mask(
        num_nodes=1,
        aa_constraint_mask=bad_shape,
        inverse_fold_restriction=[],
        canonical_tokens=const.canonical_tokens,
        inf=INF,
        device=torch.device("cpu"),
    )

    # No restrictions should remain after ignoring mismatched input.
    assert out.shape == (1, len(const.canonical_tokens))
    assert torch.all(out == 0)
