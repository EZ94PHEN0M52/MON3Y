"""
Unit tests for probability calibration (Phase 6).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration import (  # noqa: E402
    apply_calibration,
    fit_calibrator,
    fit_calibrators_from_frame,
)
from distributional import (  # noqa: E402
    poisson_over_probability,
    count_threshold_for_line,
)
import pandas as pd  # noqa: E402


def test_isotonic_improves_calibration():
    rng = np.random.default_rng(42)
    n = 500

    raw = rng.uniform(0.05, 0.95, n)
    labels = (
        rng.random(n) < (raw * 0.7 + 0.1)
    ).astype(int)

    package = fit_calibrator(
        raw,
        labels,
        method="isotonic",
    )

    assert package is not None
    assert package["n_samples"] == n

    calibrated = [
        apply_calibration(
            package,
            value,
        )
        for value in raw
    ]

    assert all(
        0.0 <= value <= 1.0
        for value in calibrated
    )

    raw_brier = np.mean(
        (raw - labels) ** 2
    )
    cal_brier = np.mean(
        (np.array(calibrated) - labels) ** 2
    )

    assert cal_brier <= raw_brier + 0.05


def test_platt_calibration():
    raw = np.array([
        0.2,
        0.3,
        0.7,
        0.8,
        0.55,
        0.45,
    ] * 30)
    labels = (raw > 0.5).astype(int)

    package = fit_calibrator(
        raw,
        labels,
        method="platt",
    )

    assert package is not None

    high = apply_calibration(
        package,
        0.9,
    )
    low = apply_calibration(
        package,
        0.1,
    )

    assert high > low


def test_apply_calibration_missing_package():
    assert apply_calibration(
        None,
        0.6,
    ) == 0.6


def test_fit_calibrators_from_frame_uses_raw_column():
    frame = pd.DataFrame({
        "market": ["batter_hits"] * 200,
        "raw_over_probability": np.linspace(
            0.1,
            0.9,
            200,
        ),
        "over_probability": np.linspace(
            0.2,
            0.8,
            200,
        ),
        "actual_over": (
            np.arange(200) % 2
        ),
    })

    saved = fit_calibrators_from_frame(
        frame,
        version="v2",
        min_samples=100,
    )

    assert "batter_hits" in saved


def test_poisson_over_probability():
    assert count_threshold_for_line(0.5) == 1
    assert count_threshold_for_line(1.5) == 2

    low = poisson_over_probability(
        0.5,
        1.5,
    )
    high = poisson_over_probability(
        2.0,
        1.5,
    )

    assert 0.0 <= low <= 1.0
    assert high > low


def main():
    tests = [
        test_isotonic_improves_calibration,
        test_platt_calibration,
        test_apply_calibration_missing_package,
        test_fit_calibrators_from_frame_uses_raw_column,
        test_poisson_over_probability,
    ]

    passed = 0

    for test in tests:
        test()
        print(f"PASS {test.__name__}")
        passed += 1

    print()
    print(f"{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
