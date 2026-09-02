"""Measure step-wedge samples on a fixed cell grid and convert signals to density."""

from __future__ import annotations

import numpy as np

from .models import (
    CLIP_LEVEL,
    T2115_DENSITIES,
    AnalysisResult,
    AnalysisSettings,
    Calibration,
    Orientation,
    Selection,
    StepMeasurement,
)

# Fraction of each cell (along the wedge) that is measured. The rest is margin
# for hand-drawn selections that do not line up with the step edges exactly.
CELL_FILL = 0.5


def analyze_selection(
    image: np.ndarray,
    selection: Selection,
    settings: AnalysisSettings | None = None,
) -> AnalysisResult:
    """Measure a sample exposed through the wedge (negative or print).

    The selection is divided into equal cells, one per wedge step, because the
    wedge geometry is fixed. Steps are identified by position: the dark end of
    the strip received the most exposure and is therefore wedge step 1.

    Args:
        image: Two-dimensional luminance image indexed as rows then columns.
        selection: Rectangle enclosing the complete wedge image, edge to edge.
        settings: Wedge definition and density reference configuration.

    Returns:
        The clipped selection, cell boundaries, one measurement per wedge step
        in step order, and data-quality warnings.

    Raises:
        ValueError: If the selection is too small or calibrated mode is
            requested without a calibration.
    """
    settings = settings or AnalysisSettings()
    if settings.reference_mode == "calibrated" and settings.calibration is None:
        raise ValueError("Calibrated mode needs a calibration. Select the scanned T2115 and calibrate first.")

    wedge = np.asarray(settings.wedge_densities, dtype=np.float64)
    crop, clipped_selection, orientation = crop_strip(image, selection, len(wedge))
    boundaries = equal_boundaries(crop.shape[1], len(wedge))
    low, mid, high = measure_cells(crop, boundaries)

    # Whatever was exposed through the wedge is darkest behind its thinnest
    # step, so the low-signal end of the strip is wedge step 1.
    reversed_strip = mid[:3].mean() > mid[-3:].mean()
    cells = np.arange(len(wedge))[::-1] if reversed_strip else np.arange(len(wedge))

    reference_value = None if settings.reference_mode == "calibrated" else float(max(mid.max(), 1.0))
    densities = signal_to_density(mid[cells], settings, reference_value)
    # The 16th/84th percentile signals pushed through the same mapping give one
    # sigma of density variation inside each cell.
    spreads = (
        signal_to_density(low[cells], settings, reference_value)
        - signal_to_density(high[cells], settings, reference_value)
    ) / 2.0
    # The densest wedge step passes the least light and defines zero exposure.
    log_exposures = wedge[-1] - wedge

    measurements = tuple(
        StepMeasurement(
            step=step,
            spatial_index=int(cells[index]) + 1,
            signal=float(mid[cells[index]]),
            density=float(densities[index]),
            density_spread=float(spreads[index]),
            relative_log_exposure=float(log_exposures[index]),
            wedge_density=float(wedge[index]),
        )
        for index, step in enumerate(range(1, len(wedge) + 1))
    )

    warnings: list[str] = []
    clipped = int(np.sum(mid >= CLIP_LEVEL))
    if clipped:
        warnings.append(f"{clipped} step(s) clipped at scanner white; reduce scanner exposure.")
    if settings.calibration is not None:
        beyond = int(np.sum(mid < settings.calibration.min_signal))
        if beyond:
            warnings.append(
                f"{beyond} step(s) darker than the calibration can resolve "
                f"(reported as D {settings.calibration.max_density:.2f})."
            )

    return AnalysisResult(
        selection=clipped_selection,
        orientation=orientation,
        boundaries=boundaries,
        measurements=measurements,
        reference_mode=settings.reference_mode,
        reference_value=reference_value,
        warnings=tuple(warnings),
    )


def calibrate_from_wedge(
    image: np.ndarray,
    selection: Selection,
    wedge_densities: tuple[float, ...] = T2115_DENSITIES,
) -> Calibration:
    """Build a signal-to-density map from a scan of the wedge itself.

    Args:
        image: Two-dimensional luminance image containing the scanned wedge.
        selection: Rectangle enclosing the complete wedge, edge to edge.
        wedge_densities: Known density of each step, step 1 first. Use the
            certificate values for a calibrated wedge.

    Returns:
        A :class:`Calibration` pairing each step's median signal with its
        known density.
    """
    wedge = tuple(float(value) for value in wedge_densities)
    crop, _, _ = crop_strip(image, selection, len(wedge))
    mid = measure_cells(crop, equal_boundaries(crop.shape[1], len(wedge)))[1]
    # The wedge itself transmits least through step 21, so its dark end is
    # the high-density end; the opposite of a sample exposed through it.
    if mid[:3].mean() < mid[-3:].mean():
        mid = mid[::-1]
    return Calibration(signals=tuple(float(value) for value in mid), densities=wedge)


def crop_strip(
    image: np.ndarray,
    selection: Selection,
    step_count: int,
) -> tuple[np.ndarray, Selection, Orientation]:
    """Crop the selection and orient it so the wedge runs along axis 1.

    Args:
        image: Two-dimensional luminance image.
        selection: Rectangle to crop, in image coordinates.
        step_count: Number of cells the strip must be able to hold.

    Returns:
        The crop with the wedge along columns, the clipped selection, and the
        original orientation.

    Raises:
        ValueError: If the strip is too thin or too short to measure.
    """
    clipped = selection.clipped(image.shape[1], image.shape[0])
    crop = image[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1]
    orientation: Orientation = "horizontal" if crop.shape[1] >= crop.shape[0] else "vertical"
    if orientation == "vertical":
        crop = crop.T
    if crop.shape[0] < 16:
        raise ValueError("The selected strip is too thin to analyze reliably.")
    if crop.shape[1] < step_count * 6:
        raise ValueError(f"The selected strip is too short. Include all {step_count} steps.")
    return crop, clipped, orientation


def equal_boundaries(length: int, step_count: int) -> np.ndarray:
    """Divide a strip of ``length`` samples into ``step_count`` equal cells.

    Args:
        length: Number of samples along the wedge.
        step_count: Number of cells to create.

    Returns:
        An integer array of ``step_count + 1`` boundary positions spanning zero
        through ``length``.
    """
    boundaries = np.rint(np.linspace(0, length, step_count + 1)).astype(int)
    boundaries[0] = 0
    boundaries[-1] = length
    # Rounding can duplicate positions; advance them to retain non-empty regions.
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
    # Preserve the exact endpoint after correcting intermediate positions.
    boundaries[-1] = length
    return boundaries


def measure_cells(crop: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Measure the central part of each cell along the wedge axis.

    Args:
        crop: Two-dimensional image region with the wedge along axis 1.
        boundaries: Consecutive cell boundaries along axis 1.

    Returns:
        A ``(3, cells)`` array holding the 16th, 50th, and 84th percentile
        signal of each cell. The median is robust to dust and step numerals;
        the outer percentiles measure the noise around it.
    """
    percentiles = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        margin = int((end - start) * (1.0 - CELL_FILL) / 2.0)
        percentiles.append(np.percentile(crop[:, start + margin : end - margin], [16, 50, 84]))
    return np.array(percentiles, dtype=np.float64).T


def signal_to_density(
    signals: np.ndarray,
    settings: AnalysisSettings,
    reference_value: float | None,
) -> np.ndarray:
    """Convert scanner signals to density.

    Args:
        signals: Measured luminance values.
        settings: Supplies the reference mode and calibration.
        reference_value: Zero-density signal for ``"relative"`` mode.

    Returns:
        Densities in signal order.

    Raises:
        ValueError: If the reference mode is unsupported or lacks its inputs.
    """
    if settings.reference_mode == "calibrated" and settings.calibration is not None:
        return settings.calibration.apply(np.asarray(signals, dtype=np.float64))
    if settings.reference_mode == "relative" and reference_value is not None:
        # Density above the brightest step. This assumes a linear scanner with
        # no black offset, so it compresses at high density; calibrate to fix.
        return np.log10(reference_value / np.clip(signals, 1.0, None))
    raise ValueError(f"Unsupported reference mode: {settings.reference_mode}")
