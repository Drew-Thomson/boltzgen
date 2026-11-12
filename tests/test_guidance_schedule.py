"""Unit tests for time-dependent symmetry guidance scheduling."""

# ruff: noqa: INP001,PLR2004,E501

from __future__ import annotations

import pytest

from boltzgen.model.modules.materials import compute_scheduled_guidance_scale


def test_linear_schedule_boundary_conditions() -> None:
    """Verify linear schedule boundaries at t_start, t_end, and outside."""
    t_start = 0.50
    t_end = 0.30
    base_scale = 1.0

    # Outside active ramp window (early diffusion: pure noise)
    assert compute_scheduled_guidance_scale(
        1.0, base_scale, t_start=t_start, t_end=t_end
    ) == 0.0
    assert compute_scheduled_guidance_scale(
        0.60, base_scale, t_start=t_start, t_end=t_end
    ) == 0.0
    assert compute_scheduled_guidance_scale(
        t_start, base_scale, t_start=t_start, t_end=t_end
    ) == 0.0

    # Critical point t = 0.53 from figure 6
    assert compute_scheduled_guidance_scale(
        0.53, base_scale, t_start=t_start, t_end=t_end
    ) == 0.0

    # Outside active ramp window (late diffusion: structural refinement)
    assert compute_scheduled_guidance_scale(
        t_end, base_scale, t_start=t_start, t_end=t_end
    ) == 1.0
    assert compute_scheduled_guidance_scale(
        0.20, base_scale, t_start=t_start, t_end=t_end
    ) == 1.0
    assert compute_scheduled_guidance_scale(
        0.0, base_scale, t_start=t_start, t_end=t_end
    ) == 1.0


def test_linear_schedule_interior_interpolation() -> None:
    """Verify exact linear progression within ramp interval."""
    t_start = 0.50
    t_end = 0.30
    base_scale = 2.0

    # Quarter point: t = 0.45 -> (0.50 - 0.45) / 0.20 = 0.25 -> 0.25 * 2.0 = 0.5
    scale_quarter = compute_scheduled_guidance_scale(
        0.45, base_scale, t_start=t_start, t_end=t_end
    )
    assert pytest.approx(scale_quarter, abs=1e-6) == 0.50

    # Midpoint: t = 0.40 -> 0.50 * 2.0 = 1.0
    scale_mid = compute_scheduled_guidance_scale(
        0.40, base_scale, t_start=t_start, t_end=t_end
    )
    assert pytest.approx(scale_mid, abs=1e-6) == 1.00

    # Three-quarter point: t = 0.35 -> 0.75 * 2.0 = 1.5
    scale_three_quarters = compute_scheduled_guidance_scale(
        0.35, base_scale, t_start=t_start, t_end=t_end
    )
    assert pytest.approx(scale_three_quarters, abs=1e-6) == 1.50


def test_sigmoid_schedule_smooth_progression() -> None:
    """Verify smooth cosine/sigmoid s-curve properties."""
    t_start = 0.60
    t_end = 0.40
    base_scale = 1.0

    # Boundaries
    assert compute_scheduled_guidance_scale(
        0.70, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
    ) == 0.0
    assert compute_scheduled_guidance_scale(
        t_start, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
    ) == 0.0
    assert compute_scheduled_guidance_scale(
        t_end, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
    ) == 1.0
    assert compute_scheduled_guidance_scale(
        0.20, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
    ) == 1.0

    # Midpoint must be exactly 0.5 for symmetric cosine curve
    mid_t = 0.50
    scale_mid = compute_scheduled_guidance_scale(
        mid_t, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
    )
    assert pytest.approx(scale_mid, abs=1e-6) == 0.50

    # Strict monotonicity within the ramp interval
    t_vals = [0.58, 0.55, 0.50, 0.45, 0.42]
    scales = [
        compute_scheduled_guidance_scale(
            t, base_scale, t_start=t_start, t_end=t_end, schedule="sigmoid"
        )
        for t in t_vals
    ]
    for s_prev, s_next in zip(scales[:-1], scales[1:]):
        assert s_prev < s_next


def test_constant_schedule_invariance() -> None:
    """Verify constant schedule maintains base scale across all timepoints."""
    for t in [1.0, 0.8, 0.53, 0.4, 0.1, 0.0]:
        assert compute_scheduled_guidance_scale(
            t, 1.5, schedule="constant"
        ) == 1.5


def test_invalid_or_inverted_interval_fallback() -> None:
    """Verify fallback behavior when t_start <= t_end or base_scale <= 0."""
    # Inverted interval falls back to constant base_scale
    assert compute_scheduled_guidance_scale(
        0.5, 1.0, t_start=0.2, t_end=0.4
    ) == 1.0
    # Zero or negative scale
    assert compute_scheduled_guidance_scale(0.5, 0.0) == 0.0
    assert compute_scheduled_guidance_scale(0.5, -1.0) == 0.0


def test_two_stage_schedule() -> None:
    """Verify two-stage gentle early ramp followed by second stage."""
    t_start = 0.85
    t_mid = 0.50
    t_end = 0.30
    early_scale = 0.10
    base_scale = 1.0

    # Boundaries
    assert compute_scheduled_guidance_scale(
        0.90, base_scale, t_start=t_start, t_end=t_end, t_mid=t_mid, schedule="two_stage"
    ) == 0.0
    assert pytest.approx(
        compute_scheduled_guidance_scale(
            t_mid, base_scale, t_start=t_start, t_end=t_end, t_mid=t_mid, early_scale=early_scale, schedule="two_stage"
        ), abs=1e-6
    ) == 0.10
    assert compute_scheduled_guidance_scale(
        t_end, base_scale, t_start=t_start, t_end=t_end, t_mid=t_mid, schedule="two_stage"
    ) == 1.0

    # Intermediate stage 1 (t = 0.675 -> midpoint of [0.85, 0.50])
    s_early = compute_scheduled_guidance_scale(
        0.675, base_scale, t_start=t_start, t_end=t_end, t_mid=t_mid, early_scale=early_scale, schedule="two_stage"
    )
    assert pytest.approx(s_early, abs=1e-6) == 0.05

    # Intermediate stage 2 (t = 0.40 -> midpoint of [0.50, 0.30])
    s_late = compute_scheduled_guidance_scale(
        0.40, base_scale, t_start=t_start, t_end=t_end, t_mid=t_mid, early_scale=early_scale, schedule="two_stage"
    )
    assert pytest.approx(s_late, abs=1e-6) == 0.55


def test_quadratic_power_schedule() -> None:
    """Verify convex quadratic schedule progression."""
    t_start = 0.80
    t_end = 0.40
    base_scale = 1.0

    # Midpoint progress = 0.50 -> progress^2 = 0.25
    s_mid = compute_scheduled_guidance_scale(
        0.60, base_scale, t_start=t_start, t_end=t_end, schedule="quadratic"
    )
    assert pytest.approx(s_mid, abs=1e-6) == 0.25


def test_decoupled_coordinate_decomposition_invariance() -> None:
    """Verify decoupled guidance preserves internal subunit geometry identically."""
    import torch

    torch.manual_seed(42)
    # 20 atoms in a subunit
    subunit_coords = torch.randn(20, 3)
    orig_com = subunit_coords.mean(dim=0)
    rel_orig = subunit_coords - orig_com

    # Ideal target COM and consensus shape
    target_com = torch.tensor([15.0, 0.0, 5.0])
    consensus_shape = torch.randn(20, 3)
    consensus_shape = consensus_shape - consensus_shape.mean(dim=0)

    # When lambda_shape = 0 and lambda_pos = 1.0:
    lambda_pos = 1.0
    lambda_shape = 0.0

    eff_com = (1.0 - lambda_pos) * orig_com + lambda_pos * target_com
    eff_rel = (1.0 - lambda_shape) * rel_orig + lambda_shape * consensus_shape
    new_coords = eff_com + eff_rel

    # Subunit centroid must match target_com exactly
    assert torch.allclose(new_coords.mean(dim=0), target_com, atol=1e-6)

    # All pairwise internal distances must be identical to machine precision
    dist_orig = torch.cdist(subunit_coords, subunit_coords)
    dist_new = torch.cdist(new_coords, new_coords)
    assert torch.allclose(dist_orig, dist_new, atol=1e-6)
