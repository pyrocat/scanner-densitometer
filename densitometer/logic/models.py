"""Shared data models for loaded images, selections, calibration, and results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL.Image import Image as PilImage


# Nominal Stouffer T2115 densities, step 1 (0.05) through step 21 (3.05), 0.15
# apart (one half stop). Replace with the certificate values of a calibrated
# T2115C wedge when available.
T2115_DENSITIES: tuple[float, ...] = tuple(round(0.05 + 0.15 * index, 2) for index in range(21))

# Signals within 2% of 16-bit full scale are treated as clipped by the scanner.
CLIP_LEVEL = 0.98 * 65535.0
# Signals below one count carry no density information (clipped at black).
BLACK_LEVEL = 1.0

# ``relative`` reports density above the brightest step (base + fog or paper
# white); ``calibrated`` maps signal through a scanned target of known density.
ReferenceMode = Literal["relative", "calibrated"]

# Fewest steps a wedge or calibration target may have: the exposure axis and
# the orientation heuristic need distinct ends and at least one step between.
MIN_WEDGE_STEPS = 3


def validate_wedge_densities(values) -> tuple[float, ...]:
    """Check a wedge or target definition shared by analysis, calibration and UI.

    Args:
        values: Density of each step, step 1 (lightest) first.

    Returns:
        The densities as a tuple of floats.

    Raises:
        ValueError: If there are fewer than ``MIN_WEDGE_STEPS`` values, any is
            not a finite number, or they are not strictly increasing. A
            strictly increasing finite wedge makes every exposure axis derived
            from it finite with distinct coordinates.
    """
    try:
        densities = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError("Wedge densities must be numbers.") from None
    if (
        len(densities) < MIN_WEDGE_STEPS
        or not all(math.isfinite(density) for density in densities)
        or any(later <= earlier for earlier, later in zip(densities, densities[1:]))
    ):
        raise ValueError(
            f"Wedge densities must be at least {MIN_WEDGE_STEPS} finite values, strictly increasing."
        )
    return densities

@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Hold source metadata and the two prepared image representations.

    Attributes:
        path: Filesystem path from which the image was loaded.
        analysis_image: Full-resolution, two-dimensional luminance samples used
            for density measurements.
        preview_image: Contrast-adjusted RGB image used by the canvas, possibly
            downscaled for display.
        mode: Pillow mode reported by the source image.
        source_dtype: String form of the source NumPy data type.
        width: Width of ``analysis_image`` in pixels.
        height: Height of ``analysis_image`` in pixels.
        full_scale: Nominal maximum signal for the normalized image data.
        bits_per_sample: Declared bit depth of the source samples; the
            analysis data was scaled from this range, never from content.
        photometric: Name of the photometric interpretation applied while
            decoding (``"MinIsBlack"``, ``"MinIsWhite"`` or ``"RGB"``).
    """

    path: Path
    analysis_image: np.ndarray
    preview_image: PilImage
    mode: str
    source_dtype: str
    width: int
    height: int
    full_scale: float = 65535.0
    bits_per_sample: int = 16
    photometric: str = "MinIsBlack"


@dataclass(frozen=True, slots=True)
class Selection:
    """A rotatable strip defined by its centreline and width, in image pixels.

    The centreline runs from ``(x0, y0)`` to ``(x1, y1)`` along the wedge, so
    moving an endpoint rotates and resizes the strip in one gesture. ``width``
    is the extent across the wedge, centred on the line. Coordinates are
    continuous: pixel ``(row, column)`` covers ``[column, column + 1)`` by
    ``[row, row + 1)``.

    Attributes:
        x0: Horizontal coordinate of the first end of the centreline.
        y0: Vertical coordinate of the first end of the centreline.
        x1: Horizontal coordinate of the second end of the centreline.
        y1: Vertical coordinate of the second end of the centreline.
        width: Extent across the wedge.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    width: float

    @property
    def length(self) -> float:
        """Return the centreline length in pixels."""
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def axis(self) -> tuple[float, float]:
        """Return the unit vector along the centreline (x-axis for a point)."""
        length = self.length
        if length == 0.0:
            return 1.0, 0.0
        return (self.x1 - self.x0) / length, (self.y1 - self.y0) / length

    def point_at(self, along: float, across: float) -> tuple[float, float]:
        """Return image coordinates for strip-local offsets.

        Args:
            along: Distance from the first end along the centreline.
            across: Signed distance from the centreline; positive is clockwise
                from the axis in image coordinates (y down).
        """
        ux, uy = self.axis
        return self.x0 + ux * along - uy * across, self.y0 + uy * along + ux * across

    def local(self, x: float, y: float) -> tuple[float, float]:
        """Return ``(along, across)`` strip-local offsets of an image point."""
        ux, uy = self.axis
        dx, dy = x - self.x0, y - self.y0
        return dx * ux + dy * uy, -dx * uy + dy * ux

    def corners(self) -> tuple[tuple[float, float], ...]:
        """Return the four corners in drawing order."""
        half = self.width / 2.0
        return (
            self.point_at(0.0, -half),
            self.point_at(self.length, -half),
            self.point_at(self.length, half),
            self.point_at(0.0, half),
        )

    def contains(self, x: float, y: float) -> bool:
        """Check whether an image point lies inside the strip."""
        along, across = self.local(x, y)
        return 0.0 <= along <= self.length and abs(across) <= self.width / 2.0


@dataclass(frozen=True, slots=True)
class Calibration:
    """Signal-to-density map built from a scan of a target with known densities.

    Valid only for scans made with the same scanner, light path (transmission
    or reflection), and locked exposure settings as the target scan.

    Attributes:
        signals: Median scanner signal of each target step, step 1 first.
        densities: Known density of each target step, in the same order.
    """

    signals: tuple[float, ...]
    densities: tuple[float, ...]

    def usable(self) -> tuple[np.ndarray, np.ndarray]:
        """Return usable ``(signal, density)`` pairs, ascending in signal.

        A step is usable when its reading is finite, unclipped at scanner
        white and black, and below every lighter usable step: beyond the
        scanner's flare floor the signal flattens and cannot be inverted into
        density, and a black-clipped reading carries no density information.
        Usable signals are therefore distinct and at least one count, so their
        logarithms are distinct interpolation coordinates.
        """
        signals = np.asarray(self.signals, dtype=np.float64)
        densities = np.asarray(self.densities, dtype=np.float64)
        finite = np.isfinite(signals) & np.isfinite(densities)
        # Unreadable steps must not block darker ones, so they do not enter
        # the running minimum of the lighter steps.
        lighter = np.where(finite, signals, np.inf)
        lighter_minimum = np.minimum.accumulate(np.concatenate(([np.inf], lighter[:-1])))
        keep = finite & (signals >= BLACK_LEVEL) & (signals < CLIP_LEVEL) & (signals < lighter_minimum)
        return signals[keep][::-1], densities[keep][::-1]

    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """Return usable ``(log10 signal, density)`` pairs, ascending in signal."""
        signals, densities = self.usable()
        return np.log10(signals), densities

    @property
    def usable_steps(self) -> int:
        """Number of target steps that contribute to the map."""
        return len(self.usable()[0])

    @property
    def min_signal(self) -> float:
        """Darkest signal the map can still resolve, as measured."""
        return float(self.usable()[0][0])

    @property
    def max_signal(self) -> float:
        """Brightest signal the map can still resolve, as measured; brighter samples clamp."""
        return float(self.usable()[0][-1])

    @property
    def max_density(self) -> float:
        """Highest density the map can report; darker samples clamp to it."""
        return float(self.usable()[1][0])

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Map scanner signals to density by interpolating in log signal.

        Signals outside the usable range take the density of the nearest
        end; ``flag_signals`` marks such samples as clamped.
        """
        log_signals, densities = self.table()
        return np.interp(np.log10(np.clip(signals, BLACK_LEVEL, None)), log_signals, densities)


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Configure the wedge definition and density reference behavior.

    Attributes:
        wedge_densities: Density of each wedge step, step 1 first. Its length
            sets the number of cells the selection is divided into.
        reference_mode: Signal-to-density strategy.
        calibration: Required when ``reference_mode`` is ``"calibrated"``.
    """

    wedge_densities: tuple[float, ...] = T2115_DENSITIES
    reference_mode: ReferenceMode = "relative"
    calibration: Calibration | None = None


@dataclass(frozen=True, slots=True)
class StepMeasurement:
    """Describe one measured wedge step and its plotting metadata.

    Attributes:
        step: One-based wedge step number; step 1 is the thinnest wedge step
            and therefore the most exposed (darkest) sample cell.
        spatial_index: One-based cell position along the selected image strip.
        signal: Median luminance measured inside the cell.
        density: Density calculated from ``signal``.
        density_spread: One-sigma density variation inside the cell. Large
            values indicate dust, texture, or a cell straddling a step edge.
        relative_log_exposure: Exposure-axis value assigned from wedge spacing.
        wedge_density: Nominal optical density of the corresponding wedge step.
        clipped: The scanner saturated in this cell, so the signal is a floor.
        clamped: The signal lies outside the calibration's range, so the
            density is the nearest end of the calibration table.
    """

    step: int
    spatial_index: int
    signal: float
    density: float
    density_spread: float
    relative_log_exposure: float
    wedge_density: float
    clipped: bool = False
    clamped: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Collect segmentation data and measurements for one selection.

    Attributes:
        selection: Strip that was analyzed.
        boundaries: Cell endpoints in pixels along the strip centreline,
            including the first and final endpoints.
        measurements: Step records in wedge step order.
        reference_mode: Density reference strategy used for the analysis.
        reference_value: Signal treated as zero density in relative mode.
        warnings: Human-readable data-quality problems found in the strip.
    """

    selection: Selection
    boundaries: np.ndarray
    measurements: tuple[StepMeasurement, ...]
    reference_mode: ReferenceMode
    reference_value: float | None
    warnings: tuple[str, ...] = ()

    @property
    def x_values(self) -> np.ndarray:
        """Return relative log exposures as a new ``float64`` plotting array."""
        return np.array([m.relative_log_exposure for m in self.measurements], dtype=np.float64)

    @property
    def y_values(self) -> np.ndarray:
        """Return measured densities as a new ``float64`` plotting array."""
        return np.array([m.density for m in self.measurements], dtype=np.float64)

    @property
    def y_spreads(self) -> np.ndarray:
        """Return per-step density spreads as a new ``float64`` plotting array."""
        return np.array([m.density_spread for m in self.measurements], dtype=np.float64)
