from __future__ import annotations

import unittest

import numpy as np

from densitometer.logic.analysis import (
    T2115_DENSITIES,
    analyze_selection,
    calibrate_from_wedge,
    equal_boundaries,
    flag_signals,
    measure_patch,
    signal_to_density,
)
from densitometer.logic.models import AnalysisSettings, Calibration, Selection, validate_wedge_densities

CELL = 18
HEIGHT = 40


def make_strip(values: np.ndarray) -> np.ndarray:
    """Build a horizontal strip with one constant cell per value."""
    strip = np.concatenate([np.full(CELL, value, dtype=np.float32) for value in values])
    return np.repeat(strip[np.newaxis, :], HEIGHT, axis=0)


def whole_strip(image: np.ndarray) -> Selection:
    """Centreline through the middle of a horizontal strip image, full height."""
    return Selection(0.0, image.shape[0] / 2, float(image.shape[1]), image.shape[0] / 2, float(image.shape[0]))


def make_rotated_strip(values: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, Selection]:
    """Paint the cells along a centreline at ``angle_degrees`` inside a zero background."""
    length = CELL * len(values)
    ux, uy = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    pad = HEIGHT
    size = int(length + 2 * pad) + 2 * HEIGHT
    x0 = pad + (length if ux < 0 else 0) + HEIGHT
    y0 = pad + (length if uy < 0 else 0) + HEIGHT
    ys, xs = np.mgrid[0:size, 0:size]
    px, py = xs + 0.5 - x0, ys + 0.5 - y0
    along = px * ux + py * uy
    across = -px * uy + py * ux
    inside = (along >= 0) & (along < length) & (np.abs(across) <= HEIGHT / 2)
    cells = np.clip(np.floor(along / CELL).astype(int), 0, len(values) - 1)
    image = np.where(inside, np.asarray(values, dtype=np.float32)[cells], 0.0).astype(np.float32)
    selection = Selection(x0, y0, x0 + ux * length, y0 + uy * length, HEIGHT * 0.8)
    return image, selection


def gamma_scanner(densities: np.ndarray) -> np.ndarray:
    """Signal of a gamma-encoded scanner with a black offset, for a given density."""
    return 300.0 + 60000.0 * (10.0 ** -densities) ** (1.0 / 2.2)


class AnalysisTests(unittest.TestCase):
    def test_relative_density_calculation(self) -> None:
        densities = signal_to_density(np.array([65535.0, 6553.5]), AnalysisSettings(), 65535.0)
        self.assertAlmostEqual(densities[0], 0.0, places=4)
        self.assertAlmostEqual(densities[1], 1.0, places=3)

    def test_equal_boundaries_cover_entire_length(self) -> None:
        boundaries = equal_boundaries(420, 21)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], 420)
        self.assertEqual(len(boundaries), 22)
        self.assertTrue(np.all(np.diff(boundaries) > 0))

    def test_steps_are_assigned_by_position_in_either_direction(self) -> None:
        values = np.linspace(54000, 1800, 21)
        for image in (make_strip(values), make_strip(values[::-1])):
            result = analyze_selection(image, whole_strip(image))
            steps = [m.step for m in result.measurements]
            self.assertEqual(steps, list(range(1, 22)))
            # Step 1 is the most exposed, hence the darkest sample cell.
            self.assertAlmostEqual(result.measurements[0].signal, 1800.0, places=1)
            self.assertAlmostEqual(result.measurements[-1].density, 0.0, places=6)
            self.assertTrue(np.allclose(np.diff(result.x_values), -0.15, atol=1e-6))
            self.assertEqual(result.warnings, ())

    def test_rotated_strips_measure_the_same_values(self) -> None:
        values = np.linspace(54000, 1800, 21)
        for angle in (0.0, 33.0, 90.0, 200.0):
            image, selection = make_rotated_strip(values, angle)
            result = analyze_selection(image, selection)
            signals = [m.signal for m in result.measurements]
            self.assertTrue(np.allclose(signals, values[::-1], atol=1.0), msg=f"angle {angle}")
            self.assertEqual(result.warnings, (), msg=f"angle {angle}")

    def test_misaligned_selection_still_measures_step_centres(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        # Off by a quarter of a cell at both ends and a little off-axis.
        selection = Selection(4.0, HEIGHT / 2 + 3, image.shape[1] - 4.0, HEIGHT / 2 - 3, HEIGHT * 0.6)
        result = analyze_selection(image, selection)
        signals = [m.signal for m in result.measurements]
        self.assertTrue(np.allclose(signals, np.linspace(1800, 54000, 21), atol=1.0))

    def test_too_thin_or_short_strip_is_rejected(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        with self.assertRaises(ValueError):
            analyze_selection(image, Selection(0.0, 20.0, float(image.shape[1]), 20.0, 4.0))
        with self.assertRaises(ValueError):
            analyze_selection(image, Selection(0.0, 20.0, 100.0, 20.0, 40.0))

    def test_calibration_recovers_known_densities(self) -> None:
        wedge_image = make_strip(gamma_scanner(np.array(T2115_DENSITIES)))
        calibration = calibrate_from_wedge(wedge_image, whole_strip(wedge_image))
        self.assertEqual(calibration.usable_steps, 21)

        true_densities = np.linspace(2.4, 0.2, 21)
        sample = make_strip(gamma_scanner(true_densities))
        settings = AnalysisSettings(reference_mode="calibrated", calibration=calibration)
        result = analyze_selection(sample, whole_strip(sample), settings)
        self.assertTrue(np.allclose(result.y_values, true_densities, atol=0.02))
        self.assertIsNone(result.reference_value)

    def test_calibrated_mode_requires_calibration(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        with self.assertRaises(ValueError):
            analyze_selection(image, whole_strip(image), AnalysisSettings(reference_mode="calibrated"))

    def test_clipped_steps_are_flagged(self) -> None:
        values = np.linspace(54000, 1800, 21)
        values[:2] = 65535.0
        image = make_strip(values)
        result = analyze_selection(image, whole_strip(image))
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("2 step(s) clipped", result.warnings[0])

    def test_calibration_drops_flattened_and_clipped_steps(self) -> None:
        signals = gamma_scanner(np.array(T2115_DENSITIES))
        signals[:2] = 65535.0  # clipped light end
        signals[-3:] = signals[-4]  # scanner flare floor reached
        wedge_image = make_strip(signals)
        calibration = calibrate_from_wedge(wedge_image, whole_strip(wedge_image))
        self.assertEqual(calibration.usable_steps, 16)
        self.assertAlmostEqual(calibration.max_density, T2115_DENSITIES[17], places=6)


class CalibrationTests(unittest.TestCase):
    def test_black_clipped_anchors_are_not_usable(self) -> None:
        calibration = Calibration(signals=(1000.0, 0.0, 0.0), densities=(0.05, 0.20, 0.35))
        self.assertEqual(calibration.usable_steps, 1)
        image = make_strip(np.array([1000.0] + [0.0] * 20))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            calibrate_from_wedge(image, whole_strip(image))

    def test_sub_one_count_anchors_do_not_duplicate_coordinates(self) -> None:
        self.assertEqual(Calibration(signals=(1000.0, 0.5, 0.2), densities=(0.05, 0.20, 0.35)).usable_steps, 1)
        calibration = Calibration(signals=(1000.0, 100.0, 1.0, 0.5), densities=(0.05, 0.20, 0.35, 0.50))
        log_signals, densities = calibration.table()
        self.assertEqual(calibration.usable_steps, 3)
        self.assertTrue(np.all(np.diff(log_signals) > 0))
        self.assertAlmostEqual(calibration.max_density, 0.35)
        self.assertEqual(calibration.min_signal, 1.0)

    def test_nonfinite_anchors_are_skipped_without_blocking_darker_steps(self) -> None:
        calibration = Calibration(signals=(1000.0, float("nan"), 100.0), densities=(0.05, 0.20, 0.35))
        self.assertEqual(calibration.usable_steps, 2)
        self.assertAlmostEqual(calibration.max_density, 0.35)
        self.assertEqual(Calibration(signals=(1000.0, 100.0), densities=(0.05, float("inf"))).usable_steps, 1)

    def test_measured_anchor_bounds_are_exact(self) -> None:
        calibration = Calibration(signals=(50000.0, 5000.0, 500.0), densities=(0.1, 1.1, 2.1))
        self.assertEqual(calibration.max_signal, 50000.0)
        self.assertEqual(calibration.min_signal, 500.0)
        settings = AnalysisSettings(reference_mode="calibrated", calibration=calibration)
        clipped, clamped = flag_signals(np.array([50000.0, 500.0, 50000.5, 499.9]), settings)
        self.assertEqual(clamped.tolist(), [False, False, True, True])
        self.assertFalse(clipped.any())
        self.assertTrue(np.allclose(calibration.apply(np.array([50000.0, 500.0])), [0.1, 2.1]))

    def test_sample_equal_to_the_brightest_anchor_is_not_clamped(self) -> None:
        wedge_signals = gamma_scanner(np.array(T2115_DENSITIES))
        wedge_image = make_strip(wedge_signals)
        calibration = calibrate_from_wedge(wedge_image, whole_strip(wedge_image))
        settings = AnalysisSettings(reference_mode="calibrated", calibration=calibration)
        result = analyze_selection(wedge_image, whole_strip(wedge_image), settings)
        self.assertFalse(any(m.clamped for m in result.measurements))
        self.assertTrue(np.allclose(result.y_values, np.array(T2115_DENSITIES)[::-1], atol=1e-9))
        self.assertEqual(result.warnings, ())


class CoverageAndOrientationTests(unittest.TestCase):
    def test_selection_reaching_outside_the_image_is_refused(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        width = float(image.shape[1])
        # Issue 4: 36 pixels beyond the edge used to become three repeated dark cells.
        with self.assertRaisesRegex(ValueError, r"Cell\(s\) \[20, 21\] .* outside the image"):
            analyze_selection(image, Selection(0.0, HEIGHT / 2, width + 36.0, HEIGHT / 2, float(HEIGHT)))
        # Only the unmeasured margin of the last cell is outside: still measurable.
        result = analyze_selection(image, Selection(0.0, HEIGHT / 2, width + 3.0, HEIGHT / 2, float(HEIGHT)))
        self.assertEqual(len(result.measurements), 21)
        # A strip wider than the image reaches outside along its whole length.
        with self.assertRaisesRegex(ValueError, "outside the image"):
            analyze_selection(image, Selection(0.0, HEIGHT / 2, width, HEIGHT / 2, HEIGHT + 4.0))
        with self.assertRaisesRegex(ValueError, "outside the image"):
            calibrate_from_wedge(image, Selection(0.0, HEIGHT / 2, width + 36.0, HEIGHT / 2, float(HEIGHT)))

    def test_patch_selection_must_be_inside_the_image(self) -> None:
        image = make_strip(np.full(21, 30000.0))
        self.assertAlmostEqual(measure_patch(image, Selection(2.0, HEIGHT / 2, 40.0, HEIGHT / 2, 20.0)), 30000.0)
        with self.assertRaisesRegex(ValueError, "outside the image"):
            measure_patch(image, Selection(-10.0, HEIGHT / 2, 40.0, HEIGHT / 2, 20.0))

    def test_three_step_targets_orient_by_their_ends(self) -> None:
        # Issue 10: three-cell windows overlapped completely for three-step targets.
        values = np.array([54000.0, 20000.0, 5000.0])
        wedge = (0.05, 0.50, 1.00)
        for image in (make_strip(values), make_strip(values[::-1])):
            result = analyze_selection(image, whole_strip(image), AnalysisSettings(wedge_densities=wedge))
            self.assertEqual([m.step for m in result.measurements], [1, 2, 3])
            self.assertAlmostEqual(result.measurements[0].signal, 5000.0)
            self.assertAlmostEqual(result.measurements[-1].density, 0.0)
        calibrations = [calibrate_from_wedge(img, whole_strip(img), wedge) for img in (make_strip(values), make_strip(values[::-1]))]
        self.assertEqual(calibrations[0], calibrations[1])
        self.assertEqual(calibrations[0].usable_steps, 3)
        self.assertEqual(calibrations[0].signals[0], 54000.0)

    def test_undetermined_direction_is_refused_or_flagged(self) -> None:
        symmetric = make_strip(np.array([1000.0] * 3 + [5000.0] * 15 + [1000.0] * 3))
        with self.assertRaisesRegex(ValueError, "Cannot tell which end"):
            analyze_selection(symmetric, whole_strip(symmetric))
        with self.assertRaisesRegex(ValueError, "Cannot tell which end"):
            calibrate_from_wedge(symmetric, whole_strip(symmetric))
        uniform = make_strip(np.full(21, 30000.0))
        result = analyze_selection(uniform, whole_strip(uniform))
        self.assertTrue(any("same signal" in w for w in result.warnings))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            calibrate_from_wedge(uniform, whole_strip(uniform))

    def test_wedge_definition_is_validated_on_every_path(self) -> None:
        # Issue 11: analysis accepted what calibration rejected, and NaN slipped through.
        image = make_strip(np.linspace(54000, 1800, 21))
        for wedge in ((0.1, 0.2), (0.1, 0.2, 0.2), (0.1, float("nan"), 0.3), (0.3, 0.2, 0.1)):
            with self.assertRaisesRegex(ValueError, "strictly increasing", msg=str(wedge)):
                analyze_selection(image, whole_strip(image), AnalysisSettings(wedge_densities=wedge))
            with self.assertRaisesRegex(ValueError, "strictly increasing", msg=str(wedge)):
                calibrate_from_wedge(image, whole_strip(image), wedge)
            with self.assertRaises(ValueError, msg=str(wedge)):
                validate_wedge_densities(wedge)
        with self.assertRaisesRegex(ValueError, "numbers"):
            validate_wedge_densities(("a", "b", "c"))
        self.assertEqual(validate_wedge_densities(["0.1", 0.2, 0.3]), (0.1, 0.2, 0.3))


class SelectionTests(unittest.TestCase):
    def test_local_and_point_at_are_inverse(self) -> None:
        selection = Selection(10.0, 20.0, 110.0, 70.0, 12.0)
        for along, across in ((0.0, 0.0), (50.0, -6.0), (111.8, 6.0)):
            x, y = selection.point_at(along, across)
            back = selection.local(x, y)
            self.assertAlmostEqual(back[0], along, places=9)
            self.assertAlmostEqual(back[1], across, places=9)
        self.assertTrue(selection.contains(*selection.point_at(50.0, 5.9)))
        self.assertFalse(selection.contains(*selection.point_at(50.0, 6.1)))
        self.assertFalse(selection.contains(*selection.point_at(-1.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
