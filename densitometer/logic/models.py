"""Shared data models for loaded images, selections, calibration, and results."""

from __future__ import annotations

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

# ``relative`` reports density above the brightest step (base + fog or paper
# white); ``calibrated`` maps signal through a scanned target of known density.
ReferenceMode = Literal["relative", "calibrated"]

# Orientation identifies the image axis along which wedge steps are arranged.
Orientation = Literal["horizontal", "vertical"]


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
    """

    path: Path
    analysis_image: np.ndarray
    preview_image: PilImage
    mode: str
    source_dtype: str
    width: int
    height: int
    full_scale: float = 65535.0


@dataclass(frozen=True, slots=True)
class Selection:
    """Represent a rectangular image selection using half-open coordinates.

    The first corner is inclusive and the second is exclusive, matching NumPy
    slice semantics. Corners may be supplied in either drag direction and can
    be ordered with :meth:`normalized` before use.

    Attributes:
        x0: Horizontal coordinate of the first corner.
        y0: Vertical coordinate of the first corner.
        x1: Horizontal coordinate of the opposite corner.
        y1: Vertical coordinate of the opposite corner.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def normalized(self) -> "Selection":
        """Order both coordinate pairs from minimum to maximum.

        Returns:
            A new selection whose first corner is top-left and whose second
            corner is bottom-right.
        """
        return Selection(
            x0=min(self.x0, self.x1),
            y0=min(self.y0, self.y1),
            x1=max(self.x0, self.x1),
            y1=max(self.y0, self.y1),
        )

    def clipped(self, width: int, height: int) -> "Selection":
        """Normalize the selection and constrain it to image boundaries.

        Args:
            width: Positive image width in pixels.
            height: Positive image height in pixels.

        Returns:
            A new selection whose start coordinates are valid pixel indices and
            whose exclusive end coordinates do not exceed the image dimensions.
        """
        normalized = self.normalized()
        # Start coordinates identify pixels and stop at dimension - 1. Exclusive
        # end coordinates may equal the full image dimension.
        return Selection(
            x0=max(0, min(normalized.x0, width - 1)),
            y0=max(0, min(normalized.y0, height - 1)),
            x1=max(1, min(normalized.x1, width)),
            y1=max(1, min(normalized.y1, height)),
        )

    @property
    def width(self) -> int:
        """Return the non-negative horizontal extent in pixels."""
        normalized = self.normalized()
        return normalized.x1 - normalized.x0

    @property
    def height(self) -> int:
        """Return the non-negative vertical extent in pixels."""
        normalized = self.normalized()
        return normalized.y1 - normalized.y0

    def is_large_enough(self, minimum_size: int = 21) -> bool:
        """Check whether both dimensions meet a minimum size.

        Args:
            minimum_size: Required pixel extent on each axis.

        Returns:
            ``True`` when both normalized dimensions are at least
            ``minimum_size``; otherwise, ``False``.
        """
        return self.width >= minimum_size and self.height >= minimum_size


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

    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """Return usable ``(log10 signal, density)`` pairs, ascending in signal.

        Steps that are clipped, or whose signal is not below every lighter
        step, are dropped: beyond the scanner's own flare floor the signal
        flattens and cannot be inverted into density.
        """
        signals = np.asarray(self.signals, dtype=np.float64)
        densities = np.asarray(self.densities, dtype=np.float64)
        lighter_minimum = np.minimum.accumulate(np.concatenate(([np.inf], signals[:-1])))
        keep = (signals < lighter_minimum) & (signals < CLIP_LEVEL)
        log_signals = np.log10(np.clip(signals[keep], 1.0, None))
        return log_signals[::-1], densities[keep][::-1]

    @property
    def usable_steps(self) -> int:
        """Number of target steps that contribute to the map."""
        return len(self.table()[0])

    @property
    def min_signal(self) -> float:
        """Darkest signal the map can still resolve."""
        return float(10 ** self.table()[0][0])

    @property
    def max_density(self) -> float:
        """Highest density the map can report; darker samples clamp to it."""
        return float(self.table()[1][0])

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Map scanner signals to density by interpolating in log signal."""
        log_signals, densities = self.table()
        return np.interp(np.log10(np.clip(signals, 1.0, None)), log_signals, densities)


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
    """

    step: int
    spatial_index: int
    signal: float
    density: float
    density_spread: float
    relative_log_exposure: float
    wedge_density: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Collect segmentation data and measurements for one selection.

    Attributes:
        selection: Normalized, image-clipped region that was analyzed.
        orientation: Axis along which the wedge steps are arranged.
        boundaries: Cell endpoints relative to the selected crop's wedge axis,
            including the first and final endpoints.
        measurements: Step records in wedge step order.
        reference_mode: Density reference strategy used for the analysis.
        reference_value: Signal treated as zero density in relative mode.
        warnings: Human-readable data-quality problems found in the strip.
    """

    selection: Selection
    orientation: Orientation
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
