from __future__ import annotations

import unittest

import numpy as np

from densitometer.logic.analysis import (
    T2115_DENSITIES,
    analyze_selection,
    calibrate_from_wedge,
    equal_boundaries,
    signal_to_density,
)
from densitometer.logic.models import AnalysisSettings, Selection


def make_strip(values: np.ndarray, width: int = 18, height: int = 40) -> np.ndarray:
    """Build a horizontal strip with one constant cell per value."""
    strip = np.concatenate([np.full(width, value, dtype=np.float32) for value in values])
    return np.repeat(strip[np.newaxis, :], height, axis=0)


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
            result = analyze_selection(image, Selection(0, 0, image.shape[1], image.shape[0]))
            self.assertEqual(result.orientation, "horizontal")
            steps = [m.step for m in result.measurements]
            self.assertEqual(steps, list(range(1, 22)))
            # Step 1 is the most exposed, hence the darkest sample cell.
            self.assertAlmostEqual(result.measurements[0].signal, 1800.0, places=1)
            self.assertAlmostEqual(result.measurements[-1].density, 0.0, places=6)
            exposures = result.x_values
            self.assertTrue(np.allclose(np.diff(exposures), -0.15, atol=1e-6))
            self.assertEqual(result.warnings, ())

    def test_analysis_supports_vertical_strip(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21)).T
        result = analyze_selection(image, Selection(0, 0, image.shape[1], image.shape[0]))
        self.assertEqual(result.orientation, "vertical")
        self.assertEqual(len(result.measurements), 21)

    def test_misaligned_selection_still_measures_step_centres(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        # Off by a quarter of a cell at both ends.
        result = analyze_selection(image, Selection(4, 0, image.shape[1] - 4, image.shape[0]))
        signals = [m.signal for m in result.measurements]
        self.assertTrue(np.allclose(signals, np.linspace(1800, 54000, 21), atol=1.0))

    def test_calibration_recovers_known_densities(self) -> None:
        wedge_image = make_strip(gamma_scanner(np.array(T2115_DENSITIES)))
        calibration = calibrate_from_wedge(wedge_image, Selection(0, 0, wedge_image.shape[1], wedge_image.shape[0]))
        self.assertEqual(calibration.usable_steps, 21)

        true_densities = np.linspace(2.4, 0.2, 21)
        sample = make_strip(gamma_scanner(true_densities))
        settings = AnalysisSettings(reference_mode="calibrated", calibration=calibration)
        result = analyze_selection(sample, Selection(0, 0, sample.shape[1], sample.shape[0]), settings)
        self.assertTrue(np.allclose(result.y_values, true_densities, atol=0.02))
        self.assertIsNone(result.reference_value)

    def test_calibrated_mode_requires_calibration(self) -> None:
        image = make_strip(np.linspace(54000, 1800, 21))
        with self.assertRaises(ValueError):
            analyze_selection(image, Selection(0, 0, image.shape[1], image.shape[0]), AnalysisSettings(reference_mode="calibrated"))

    def test_clipped_steps_are_flagged(self) -> None:
        values = np.linspace(54000, 1800, 21)
        values[:2] = 65535.0
        image = make_strip(values)
        result = analyze_selection(image, Selection(0, 0, image.shape[1], image.shape[0]))
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("2 step(s) clipped", result.warnings[0])

    def test_calibration_drops_flattened_and_clipped_steps(self) -> None:
        signals = gamma_scanner(np.array(T2115_DENSITIES))
        signals[:2] = 65535.0  # clipped light end
        signals[-3:] = signals[-4]  # scanner flare floor reached
        wedge_image = make_strip(signals)
        calibration = calibrate_from_wedge(wedge_image, Selection(0, 0, wedge_image.shape[1], wedge_image.shape[0]))
        self.assertEqual(calibration.usable_steps, 16)
        self.assertAlmostEqual(calibration.max_density, T2115_DENSITIES[17], places=6)


if __name__ == "__main__":
    unittest.main()
