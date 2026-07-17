"""Shared data models for loaded images, selections, and analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL.Image import Image as PilImage


# Density calculations can use either the brightest selected step or the
# scanner's nominal 16-bit maximum as their zero-density reference.
ReferenceMode = Literal["selection_max", "full_scale"]

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
class AnalysisSettings:
    """Configure wedge segmentation and density reference behavior.

    Attributes:
        step_count: Number of expected regions in the selected wedge.
        reference_mode: Signal source treated as the zero-density reference.
    """

    step_count: int = 21
    reference_mode: ReferenceMode = "selection_max"


@dataclass(frozen=True, slots=True)
class StepMeasurement:
    """Describe one measured wedge step and its plotting metadata.

    Attributes:
        graph_index: One-based position after sorting by measured density.
        spatial_index: One-based position along the selected image strip.
        signal: Median luminance measured inside the step.
        density: Base-10 density calculated from ``signal``.
        relative_log_exposure: Exposure-axis value assigned from wedge spacing.
        wedge_density: Nominal optical density of the corresponding T2115 step.
    """

    graph_index: int
    spatial_index: int
    signal: float
    density: float
    relative_log_exposure: float
    wedge_density: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Collect profiles, segmentation data, and measurements for one selection.

    Attributes:
        selection: Normalized, image-clipped region that was analyzed.
        orientation: Axis along which the wedge steps are arranged.
        boundaries: Segment endpoints relative to the selected crop's wedge
            axis, including the first and final endpoints.
        profile: Unsmoothed one-dimensional median signal through the wedge.
        smoothed_profile: Profile used for transition detection.
        measurements: Step records ordered for plotting by increasing density.
        reference_mode: Density reference strategy used for the analysis.
        reference_value: Signal value treated as zero optical density.
    """

    selection: Selection
    orientation: Orientation
    boundaries: np.ndarray
    profile: np.ndarray
    smoothed_profile: np.ndarray
    measurements: tuple[StepMeasurement, ...]
    reference_mode: ReferenceMode
    reference_value: float

    @property
    def x_values(self) -> np.ndarray:
        """Return relative log exposures as a new ``float64`` plotting array."""
        return np.array(
            [measurement.relative_log_exposure for measurement in self.measurements],
            dtype=np.float64,
        )

    @property
    def y_values(self) -> np.ndarray:
        """Return measured densities as a new ``float64`` plotting array."""
        return np.array(
            [measurement.density for measurement in self.measurements],
            dtype=np.float64,
        )
