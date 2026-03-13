from __future__ import annotations

import unittest

import numpy as np

from densitometer.logic.analysis import (
    T2115_DENSITIES,
    analyze_selection,
    calculate_densities,
    equal_boundaries,
)
from densitometer.logic.models import AnalysisSettings, Selection


def make_step_strip(width: int = 18, height: int = 40) -> np.ndarray:
    values = np.linspace(54000, 1800, 21)
    strip = np.concatenate([np.full(width, value, dtype=np.float32) for value in values], axis=0)
    image = np.repeat(strip[np.newaxis, :], height, axis=0)
    return image


class AnalysisTests(unittest.TestCase):
    def test_full_scale_density_calculation(self) -> None:
        densities, reference = calculate_densities(np.array([65535.0, 6553.5]), "full_scale")
        self.assertAlmostEqual(reference, 65535.0)
        self.assertAlmostEqual(densities[0], 0.0, places=4)
        self.assertAlmostEqual(densities[1], 1.0, places=3)

    def test_equal_boundaries_cover_entire_length(self) -> None:
        boundaries = equal_boundaries(420, 21)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], 420)
        self.assertEqual(len(boundaries), 22)
        self.assertTrue(np.all(np.diff(boundaries) > 0))

    def test_analysis_detects_twenty_one_steps(self) -> None:
        image = make_step_strip()
        selection = Selection(0, 0, image.shape[1], image.shape[0])
        result = analyze_selection(image, selection, AnalysisSettings(reference_mode="selection_max"))

        self.assertEqual(len(result.measurements), 21)
        self.assertEqual(result.orientation, "horizontal")

        densities = np.array([measurement.density for measurement in result.measurements])
        self.assertTrue(np.all(np.diff(densities) >= -1e-6))

        exposures = np.array([measurement.relative_log_exposure for measurement in result.measurements])
        self.assertTrue(np.allclose(np.diff(exposures), 0.15, atol=1e-6))

    def test_analysis_supports_vertical_strip(self) -> None:
        image = make_step_strip().T
        selection = Selection(0, 0, image.shape[1], image.shape[0])
        result = analyze_selection(image, selection)

        self.assertEqual(result.orientation, "vertical")
        self.assertEqual(len(result.measurements), 21)


if __name__ == "__main__":
    unittest.main()
