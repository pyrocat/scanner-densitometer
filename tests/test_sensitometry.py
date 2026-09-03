from __future__ import annotations

import math
import unittest
from decimal import Decimal

import numpy as np

from densitometer.logic.analysis import analyze_selection, calibrate_from_wedge
from densitometer.logic.models import AnalysisSettings, T2115_DENSITIES
from densitometer.logic.sensitometry import (
    iso_range,
    iso_range_class,
    iso_range_from_analysis,
    round_ler,
)
from test_analysis import gamma_scanner, make_strip, whole_strip

STEPS = np.arange(21) * 0.15
FINE = np.linspace(0.0, 3.0, 30001)
# (dmin, dmax, s, a) from the ADR 0001 benchmark: R50 through R167.
CURVES = [(0.07, 2.10, s, 1.0) for s in (0.15, 0.26, 0.32, 0.41, 0.50)]


def paper_curve(x: np.ndarray, dmin: float, dmax: float, s: float, a: float) -> np.ndarray:
    phi = 0.5 * (1.0 + np.vectorize(math.erf)((np.asarray(x, float) - 1.5) / (s * math.sqrt(2.0))))
    return dmin + (dmax - dmin) * phi**a


def true_r(dmin: float, dmax: float, s: float, a: float) -> float:
    d = paper_curve(FINE, dmin, dmax, s, a)
    return 100.0 * (np.interp(dmin + 0.9 * (dmax - dmin), d, FINE) - np.interp(dmin + 0.04, d, FINE))


class IsoRangeTests(unittest.TestCase):
    def test_supplied_dmin_recovers_benchmark_curves_within_two_r(self) -> None:
        for dmin, dmax, s, a in CURVES:
            estimate = iso_range(STEPS, paper_curve(STEPS, dmin, dmax, s, a), dmin=dmin)
            self.assertAlmostEqual(estimate.raw_r, true_r(dmin, dmax, s, a), delta=2.0, msg=f"s={s}")
            self.assertEqual(estimate.dmin_source, "supplied")
            self.assertAlmostEqual(estimate.dmax, dmax, delta=0.01)

    def test_estimated_dmin_is_labelled_and_order_does_not_matter(self) -> None:
        dmin, dmax, s, a = CURVES[1]
        densities = paper_curve(STEPS, dmin, dmax, s, a)
        forward = iso_range(STEPS, densities)
        reversed_ = iso_range(STEPS[::-1], densities[::-1])
        self.assertEqual(forward, reversed_)
        self.assertEqual(forward.dmin_source, "estimated")
        self.assertGreaterEqual(forward.light_plateau_steps, 2)
        self.assertAlmostEqual(forward.raw_r, true_r(dmin, dmax, s, a), delta=2.0)
        self.assertTrue(any("unexposed patch" in w for w in forward.warnings))

    def test_cut_off_curves_are_refused(self) -> None:
        # Exposure ends before Dmax: no dark plateau.
        with self.assertRaisesRegex(ValueError, "Dmax"):
            iso_range(STEPS, paper_curve(STEPS - 1.3, 0.07, 2.1, 0.26, 1.0))
        # Starts on the toe: HT is bracketed but there is no light plateau.
        with self.assertRaisesRegex(ValueError, "paper white"):
            iso_range(STEPS, paper_curve(STEPS + 0.85, 0.07, 2.1, 0.26, 1.0))
        # The same curve is fine once Dmin comes from an unexposed patch.
        estimate = iso_range(STEPS, paper_curve(STEPS + 0.85, 0.07, 2.1, 0.26, 1.0), dmin=0.07)
        self.assertAlmostEqual(estimate.raw_r, true_r(0.07, 2.1, 0.26, 1.0), delta=2.0)
        # Starts past HT: the toe endpoint is not bracketed at all.
        with self.assertRaisesRegex(ValueError, "bracketed"):
            iso_range(STEPS, paper_curve(STEPS + 1.3, 0.07, 2.1, 0.26, 1.0), dmin=0.07)

    def test_table_two_band_edges_and_tie(self) -> None:
        for value, expected in ((0.345, 40), (0.344, None), (0.445, 50), (1.944, 190), (1.945, None), (1.07, 110)):
            self.assertEqual(iso_range_class(round_ler(value)), expected, msg=str(value))
        self.assertEqual(round_ler(0.345), Decimal("0.35"))


class IsoRangeFromAnalysisTests(unittest.TestCase):
    def test_reversed_strip_gives_same_estimate(self) -> None:
        dmin, dmax, s, a = CURVES[2]
        densities = paper_curve(STEPS, dmin, dmax, s, a)
        signals = 60000.0 * 10.0 ** -(densities - dmin)  # relative mode: paper white is the brightest step
        results = [
            iso_range_from_analysis(analyze_selection(image, whole_strip(image)))
            for image in (make_strip(signals[::-1]), make_strip(signals))
        ]
        self.assertEqual(results[0], results[1])
        self.assertAlmostEqual(results[0].raw_r, true_r(dmin, dmax, s, a), delta=2.0)

    def test_clamped_calibration_is_refused(self) -> None:
        wedge_signals = gamma_scanner(np.array(T2115_DENSITIES))
        wedge_signals[-8:] = wedge_signals[-9]  # calibration only resolves up to D 1.85
        wedge_image = make_strip(wedge_signals)
        calibration = calibrate_from_wedge(wedge_image, whole_strip(wedge_image))
        sample = make_strip(gamma_scanner(paper_curve(STEPS, 0.07, 2.1, 0.32, 1.0))[::-1])
        result = analyze_selection(
            sample, whole_strip(sample), AnalysisSettings(reference_mode="calibrated", calibration=calibration)
        )
        self.assertTrue(any(m.clamped for m in result.measurements))
        with self.assertRaisesRegex(ValueError, "calibration range"):
            iso_range_from_analysis(result)

    def test_rising_end_is_not_pooled_into_a_plateau(self) -> None:
        # Truncated curve: the last two raw steps still differ by 0.03 D.
        densities = paper_curve(STEPS - 0.9, 0.07, 2.1, 0.32, 1.0)
        self.assertGreater(densities[-1] - densities[-2], 0.02)
        with self.assertRaisesRegex(ValueError, "Dmax"):
            iso_range(STEPS, densities, dmin=0.07)
        # A tiny reversal at the dark end must not be refused as "rising" either.
        densities = paper_curve(STEPS, 0.07, 2.1, 0.32, 1.0)
        densities[-1] -= 0.004
        self.assertGreaterEqual(iso_range(STEPS, densities, dmin=0.07).dark_plateau_steps, 2)

    def test_black_clipped_steps_are_flagged_and_refused(self) -> None:
        signals = 60000.0 * 10.0 ** -paper_curve(STEPS, 0.0, 2.1, 0.32, 1.0)
        signals[-3:] = 0.0
        image = make_strip(signals[::-1])
        result = analyze_selection(image, whole_strip(image))
        self.assertEqual(sum(m.clipped for m in result.measurements), 3)
        with self.assertRaisesRegex(ValueError, "clipped"):
            iso_range_from_analysis(result)

    def test_degenerate_calibration_is_rejected(self) -> None:
        image = make_strip(np.full(21, 65535.0))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            calibrate_from_wedge(image, whole_strip(image))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            calibrate_from_wedge(image, whole_strip(image), (0.1, 0.1, 0.2))

    def test_clipped_step_is_refused(self) -> None:
        signals = 60000.0 * 10.0 ** -paper_curve(STEPS, 0.0, 2.1, 0.32, 1.0)
        signals[:3] = 65535.0
        image = make_strip(signals[::-1])
        result = analyze_selection(image, whole_strip(image))
        with self.assertRaisesRegex(ValueError, "clipped"):
            iso_range_from_analysis(result)


if __name__ == "__main__":
    unittest.main()
