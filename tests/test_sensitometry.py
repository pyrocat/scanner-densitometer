from __future__ import annotations

import math
import unittest
from decimal import Decimal

import numpy as np

from densitometer.logic.analysis import analyze_selection, calibrate_from_wedge
from densitometer.logic.models import AnalysisSettings, T2115_DENSITIES
from densitometer.logic.sensitometry import (
    DMAX_PLATEAU_SPAN,
    FIT_ADJUSTMENT_WARNING,
    MAX_FIT_ADJUSTMENT,
    MIN_NET_DENSITY,
    SUPPLIED_DMIN_TOLERANCE,
    DarkPatch,
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


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, float) / math.sqrt(2.0)))


def compound_shoulder(x: np.ndarray) -> np.ndarray:
    """Issue 5 synthetic: a fast component plus a slow shoulder that keeps rising beyond the wedge."""
    return 0.07 + 2.03 * (0.8 * normal_cdf((x - 1.5) / 0.26) + 0.2 * normal_cdf((x - 2.4) / 1.2))


def strip_image(densities: np.ndarray, dmin: float) -> np.ndarray:
    """Relative-mode strip whose brightest step is paper white."""
    return make_strip(60000.0 * 10.0 ** -(densities - dmin))


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


class IsoRangeValidityTests(unittest.TestCase):
    def test_direct_inputs_are_checked(self) -> None:
        densities = paper_curve(STEPS, 0.07, 2.1, 0.32, 1.0)
        with self.assertRaisesRegex(ValueError, "matching one-dimensional"):
            iso_range(STEPS[:-1], densities)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            iso_range(np.stack([STEPS, STEPS]), np.stack([densities, densities]))
        with self.assertRaisesRegex(ValueError, "at least three"):
            iso_range(STEPS[:2], densities[:2])
        broken = densities.copy()
        broken[5] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            iso_range(STEPS, broken)
        duplicated = STEPS.copy()
        duplicated[3] = duplicated[2]
        with self.assertRaisesRegex(ValueError, "distinct"):
            iso_range(duplicated, densities)
        with self.assertRaisesRegex(ValueError, "supplied Dmin is not a finite"):
            iso_range(STEPS, densities, dmin=float("nan"))

    def test_curves_without_usable_net_density_are_refused(self) -> None:
        # Issue 7: 0 to 0.042 D used to return a negative LER with no warning.
        with self.assertRaisesRegex(ValueError, "net density"):
            iso_range(STEPS, np.linspace(0.0, 0.042, 21), dmin=0.0)
        # Linear ramps of 0.10 and 0.20 D used to pass every check.
        for total in (0.10, 0.20):
            with self.assertRaisesRegex(ValueError, "net density"):
                iso_range(STEPS, np.linspace(0.0, total, 21), dmin=0.0)
        # Just below the minimum net density.
        with self.assertRaisesRegex(ValueError, "net density"):
            iso_range(STEPS, paper_curve(STEPS, 0.07, 0.07 + MIN_NET_DENSITY - 0.01, 0.32, 1.0), dmin=0.07)

    def test_low_contrast_curve_above_minimum_net_density_is_accepted(self) -> None:
        dmin, dmax, s, a = 0.07, 0.07 + 0.50, 0.32, 1.0
        estimate = iso_range(STEPS, paper_curve(STEPS, dmin, dmax, s, a), dmin=dmin)
        self.assertGreaterEqual(estimate.net_density, MIN_NET_DENSITY)
        self.assertAlmostEqual(estimate.raw_r, true_r(dmin, dmax, s, a), delta=3.0)
        self.assertGreater(estimate.log_hs, estimate.log_ht)

    def test_supplied_dmin_above_the_light_end_is_refused(self) -> None:
        dmin, dmax, s, a = 0.07, 2.10, 0.32, 1.0
        densities = paper_curve(STEPS, dmin, dmax, s, a)
        # Issue 7: 0.20 against a light end of 0.07 returned R 85.9 silently.
        with self.assertRaisesRegex(ValueError, "supplied Dmin 0.200 exceeds"):
            iso_range(STEPS, densities, dmin=0.20)
        # The tolerance is measured from the light plateau mean, which sits a
        # little above the true Dmin because the toe is still falling there.
        light_end = iso_range(STEPS, densities).dmin
        self.assertAlmostEqual(light_end, 0.075, delta=0.002)
        with self.assertRaisesRegex(ValueError, "supplied Dmin"):
            iso_range(STEPS, densities, dmin=light_end + SUPPLIED_DMIN_TOLERANCE + 0.001)
        # Within the tolerance, and below the light end (soft paper), it is used as given.
        accepted = iso_range(STEPS, densities, dmin=light_end + SUPPLIED_DMIN_TOLERANCE - 0.001)
        self.assertEqual(accepted.dmin_source, "supplied")
        below = iso_range(STEPS, densities, dmin=dmin - 0.02)
        self.assertAlmostEqual(below.dmin, dmin - 0.02)
        self.assertEqual(below.dmin_source, "supplied")

    def test_large_reversals_are_refused_and_moderate_ones_warned(self) -> None:
        # Issue 6: this sequence returned R 169.4 with no warning.
        sequence = np.array(
            [0.07, 0.07, 0.07, 0.10, 0.25, 1.60, 0.40, 1.80, 0.50, 0.70, 0.90,
             1.20, 1.40, 1.60, 1.80, 2.00, 2.10, 2.10, 2.10, 2.10, 2.10]
        )
        with self.assertRaisesRegex(ValueError, "monotone fit changes a step by 0.8"):
            iso_range(STEPS, sequence, dmin=0.07)
        densities = paper_curve(STEPS, 0.07, 2.1, 0.32, 1.0)
        dented = densities.copy()
        dented[18] -= 0.12  # a 0.06 D adjustment on two steps: beyond the limit
        with self.assertRaisesRegex(ValueError, "monotone fit"):
            iso_range(STEPS, dented, dmin=0.07)
        dented = densities.copy()
        dented[18] -= 0.05  # pools three shoulder steps, largest change 0.03 D: warned, not refused
        estimate = iso_range(STEPS, dented, dmin=0.07)
        self.assertGreater(estimate.fit_adjustment, FIT_ADJUSTMENT_WARNING)
        self.assertLess(estimate.fit_adjustment, MAX_FIT_ADJUSTMENT)
        self.assertEqual(estimate.adjusted_steps, 3)
        self.assertTrue(any("Monotone fit adjusted 3 step(s)" in w for w in estimate.warnings))
        clean = iso_range(STEPS, densities, dmin=0.07)
        self.assertEqual(clean.fit_adjustment, 0.0)
        self.assertFalse(any("Monotone fit" in w for w in clean.warnings))

    def test_compound_shoulder_is_accepted_only_as_provisional_without_patches(self) -> None:
        estimate = iso_range(STEPS, compound_shoulder(STEPS), dmin=0.07)
        self.assertEqual(estimate.dmax_source, "provisional")
        self.assertAlmostEqual(estimate.dmax, 1.9655, delta=0.001)
        self.assertAlmostEqual(estimate.raw_r, 118.5, delta=0.3)
        self.assertEqual(estimate.dark_plateau_steps, 2)
        self.assertLess(estimate.dmax_span, DMAX_PLATEAU_SPAN)
        self.assertTrue(any("provisional" in w and "lower bound" in w for w in estimate.warnings))

    def test_overexposed_patches_expose_a_rising_shoulder(self) -> None:
        densities = compound_shoulder(STEPS)
        # One patch reading above the darkest step: no plateau at all.
        with self.assertRaisesRegex(ValueError, "still rises by 0.088 D"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=[(4.0, float(compound_shoulder(4.0)))])
        # Two patches that still differ by 0.031 D: Dmax is not reached.
        patches = [(4.0, float(compound_shoulder(4.0))), (5.0, float(compound_shoulder(5.0)))]
        with self.assertRaisesRegex(ValueError, "still rises by 0.031 D"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=patches)

    def test_overexposed_patch_establishes_dmax_of_a_saturating_curve(self) -> None:
        dmin, dmax, s, a = 0.07, 2.10, 0.32, 1.0
        densities = paper_curve(STEPS, dmin, dmax, s, a)
        wedge_only = iso_range(STEPS, densities, dmin=dmin)
        self.assertEqual(wedge_only.dmax_source, "provisional")
        self.assertAlmostEqual(wedge_only.dmax_span, 0.75, delta=1e-9)
        with_patch = iso_range(STEPS, densities, dmin=dmin, dark_patches=[(4.0, dmax)])
        self.assertEqual(with_patch.dmax_source, "established")
        self.assertEqual(with_patch.dark_patches_used, 1)
        self.assertGreaterEqual(with_patch.dmax_span, DMAX_PLATEAU_SPAN)
        self.assertAlmostEqual(with_patch.dmax, dmax, delta=0.01)
        self.assertAlmostEqual(with_patch.raw_r, true_r(dmin, dmax, s, a), delta=2.0)
        self.assertFalse(any("provisional" in w for w in with_patch.warnings))
        # A contrasty paper saturates early enough for the wedge alone to establish Dmax.
        contrasty = iso_range(STEPS, paper_curve(STEPS, dmin, dmax, 0.15, a), dmin=dmin)
        self.assertEqual(contrasty.dmax_source, "established")
        self.assertEqual(contrasty.dark_patches_used, 0)

    def test_patch_inputs_are_checked(self) -> None:
        densities = paper_curve(STEPS, 0.07, 2.1, 0.32, 1.0)
        with self.assertRaisesRegex(ValueError, "beyond the most exposed"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=[(3.0, 2.1)])
        with self.assertRaisesRegex(ValueError, "distinct exposures"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=[(4.0, 2.1), (4.0, 2.1)])
        with self.assertRaisesRegex(ValueError, "finite"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=[(4.0, float("nan"))])
        with self.assertRaisesRegex(ValueError, "pairs"):
            iso_range(STEPS, densities, dmin=0.07, dark_patches=[(4.0,)])

    def test_analysis_path_converts_patch_offsets(self) -> None:
        dmin, dmax, s, a = 0.07, 2.10, 0.32, 1.0
        image = strip_image(paper_curve(STEPS, dmin, dmax, s, a), dmin)
        result = analyze_selection(image, whole_strip(image))
        wedge_only = iso_range_from_analysis(result)
        self.assertEqual(wedge_only.dmax_source, "provisional")
        established = iso_range_from_analysis(result, dark_patches=[DarkPatch(1.0, dmax - dmin)])
        self.assertEqual(established.dmax_source, "established")
        self.assertEqual(established.dark_patches_used, 1)
        with self.assertRaisesRegex(ValueError, "beyond the most exposed"):
            iso_range_from_analysis(result, dark_patches=[DarkPatch(0.0, dmax - dmin)])


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
