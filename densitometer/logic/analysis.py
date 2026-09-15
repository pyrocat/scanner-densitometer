"""Measure step-wedge samples on a fixed cell grid and convert signals to density."""

from __future__ import annotations

import numpy as np

from .models import (
    BLACK_LEVEL,
    CLIP_LEVEL,
    T2115_DENSITIES,
    AnalysisResult,
    AnalysisSettings,
    Calibration,
    Selection,
    StepMeasurement,
    validate_wedge_densities,
)

# Fraction of each cell (along the wedge) that is measured. The rest is margin
# for hand-drawn selections that do not line up with the step edges exactly.
CELL_FILL = 0.5
# Cells compared at each end of the strip to decide which end is darker.
DIRECTION_WINDOW = 3


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
        selection: Strip whose centreline spans the complete wedge, end to end.
        settings: Wedge definition and density reference configuration.

    Returns:
        The selection, cell boundaries, one measurement per wedge step in step
        order, and data-quality warnings.

    Raises:
        ValueError: If the wedge definition is invalid, the selection is too
            small or reaches outside the image, its direction cannot be told,
            or calibrated mode is requested without a calibration.
    """
    settings = settings or AnalysisSettings()
    if settings.reference_mode == "calibrated" and settings.calibration is None:
        raise ValueError("Calibrated mode needs a calibration. Select the scanned T2115 and calibrate first.")

    wedge = np.asarray(validate_wedge_densities(settings.wedge_densities), dtype=np.float64)
    crop, covered = extract_strip(image, selection, len(wedge))
    boundaries = equal_boundaries(crop.shape[1], len(wedge))
    require_coverage(covered, boundaries)
    low, mid, high = measure_cells(crop, boundaries)

    warnings: list[str] = []
    # Whatever was exposed through the wedge is darkest behind its thinnest
    # step, so the low-signal end of the strip is wedge step 1.
    dark_first = dark_end_is_first(mid)
    if dark_first is None:
        if np.ptp(mid) > 0:
            raise ValueError(
                "Cannot tell which end of the strip is darker: both ends measure the same signal. "
                "Check that the selection runs along the wedge from end to end."
            )
        dark_first = True
        warnings.append("Every cell measures the same signal; the strip shows no wedge.")
    cells = np.arange(len(wedge)) if dark_first else np.arange(len(wedge))[::-1]

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
    clipped_cells, clamped_cells = flag_signals(mid, settings)

    measurements = tuple(
        StepMeasurement(
            step=step,
            spatial_index=int(cells[index]) + 1,
            signal=float(mid[cells[index]]),
            density=float(densities[index]),
            density_spread=float(spreads[index]),
            relative_log_exposure=float(log_exposures[index]),
            wedge_density=float(wedge[index]),
            clipped=bool(clipped_cells[cells[index]]),
            clamped=bool(clamped_cells[cells[index]]),
        )
        for index, step in enumerate(range(1, len(wedge) + 1))
    )

    clipped = int(clipped_cells.sum())
    if clipped:
        warnings.append(f"{clipped} step(s) clipped at scanner white or black; adjust scanner exposure.")
    clamped = int(clamped_cells.sum())
    if clamped:
        warnings.append(
            f"{clamped} step(s) outside the calibration range "
            f"(clamped to D {settings.calibration.table()[1][-1]:.2f} to {settings.calibration.max_density:.2f})."
        )

    return AnalysisResult(
        selection=selection,
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
        selection: Strip whose centreline spans the complete wedge, end to end.
        wedge_densities: Known density of each step, step 1 first. Use the
            certificate values for a calibrated wedge.

    Returns:
        A :class:`Calibration` pairing each step's median signal with its
        known density.

    Raises:
        ValueError: If the target definition is invalid, the selection reaches
            outside the image, the target's direction cannot be told, or fewer
            than two steps are usable.
    """
    wedge = validate_wedge_densities(wedge_densities)
    crop, covered = extract_strip(image, selection, len(wedge))
    boundaries = equal_boundaries(crop.shape[1], len(wedge))
    require_coverage(covered, boundaries)
    mid = measure_cells(crop, boundaries)[1]
    # The wedge itself transmits least through its last step, so its dark end
    # is the high-density end; the opposite of a sample exposed through it.
    dark_first = dark_end_is_first(mid)
    if dark_first is None and np.ptp(mid) > 0:
        raise ValueError(
            "Cannot tell which end of the target is darker: both ends measure the same signal. "
            "Check that the selection runs along the target from end to end."
        )
    if dark_first:
        mid = mid[::-1]
    calibration = Calibration(signals=tuple(float(value) for value in mid), densities=wedge)
    if calibration.usable_steps < 2:
        raise ValueError("Calibration failed: fewer than two target steps are unclipped and distinct.")
    return calibration


def dark_end_is_first(signals: np.ndarray) -> bool | None:
    """Tell which end of a strip is darker from non-overlapping end windows.

    Args:
        signals: Median signal of each cell along the strip.

    Returns:
        ``True`` if the first cells are darker than the last, ``False`` if the
        last are darker, ``None`` if both windows measure the same signal.
        The windows hold up to ``DIRECTION_WINDOW`` cells but never overlap,
        so short targets compare their two ends rather than the same cells.
    """
    signals = np.asarray(signals, dtype=np.float64)
    window = max(1, min(DIRECTION_WINDOW, len(signals) // 2))
    first, last = float(signals[:window].mean()), float(signals[-window:].mean())
    if first == last:
        return None
    return first < last


def flag_signals(signals: np.ndarray, settings: AnalysisSettings) -> tuple[np.ndarray, np.ndarray]:
    """Flag signals the density mapping cannot represent.

    Args:
        signals: Median scanner signals.
        settings: Supplies the reference mode and calibration.

    Returns:
        Two boolean arrays: clipped at scanner white or black, and outside the
        calibration range (always false in relative mode). Signals equal to a
        calibration anchor are inside the range.
    """
    signals = np.asarray(signals, dtype=np.float64)
    clipped = (signals >= CLIP_LEVEL) | (signals < BLACK_LEVEL)
    clamped = np.zeros_like(clipped)
    if settings.reference_mode == "calibrated" and settings.calibration is not None:
        calibration = settings.calibration
        clamped = (signals < calibration.min_signal) | (signals > calibration.max_signal)
    return clipped, clamped


def measure_patch(image: np.ndarray, selection: Selection) -> float:
    """Median signal of a uniform patch, such as unexposed processed paper.

    Args:
        image: Two-dimensional luminance image.
        selection: Strip covering only the patch; it may be short.

    Returns:
        The median signal, to be converted with the same density mapping as
        the sample it accompanies.

    Raises:
        ValueError: If the selection is too small or reaches outside the image.
    """
    crop, covered = extract_strip(image, selection, 1)
    if not covered.all():
        raise ValueError("The patch selection extends outside the image; move it fully inside.")
    return float(np.median(crop))


def extract_strip(image: np.ndarray, selection: Selection, step_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample the (possibly rotated) strip so the wedge runs along axis 1.

    Nearest-neighbour sampling keeps every value an actual scanner reading;
    the percentile statistics taken later do not benefit from interpolation.

    Args:
        image: Two-dimensional luminance image.
        selection: Strip to extract, in image coordinates.
        step_count: Number of cells the strip must be able to hold.

    Returns:
        Two arrays of shape ``(width, length)`` with the first end of the
        centreline at column zero: the sampled values, and a boolean mask that
        is ``True`` where the sample lies inside the image. Samples outside
        the image repeat its edge and are ``False`` in the mask; callers must
        check the mask before measuring, because edge padding is not data.

    Raises:
        ValueError: If the strip is too thin or too short to measure.
    """
    length = int(round(selection.length))
    width = int(round(selection.width))
    if width < 16:
        raise ValueError("The selected strip is too thin to analyze reliably.")
    if length < step_count * 6:
        raise ValueError(f"The selected strip is too short. Include all {step_count} steps.")

    ux, uy = selection.axis
    along = np.arange(length) + 0.5
    across = np.arange(width) - (width - 1) / 2.0
    xs = selection.x0 + ux * along[np.newaxis, :] - uy * across[:, np.newaxis]
    ys = selection.y0 + uy * along[np.newaxis, :] + ux * across[:, np.newaxis]
    rows = np.floor(ys).astype(int)
    columns = np.floor(xs).astype(int)
    covered = (rows >= 0) & (rows < image.shape[0]) & (columns >= 0) & (columns < image.shape[1])
    crop = image[np.clip(rows, 0, image.shape[0] - 1), np.clip(columns, 0, image.shape[1] - 1)]
    return crop, covered


def require_coverage(covered: np.ndarray, boundaries: np.ndarray) -> None:
    """Refuse a selection whose measured cell regions reach outside the image.

    Args:
        covered: Mask from :func:`extract_strip`.
        boundaries: Cell boundaries from :func:`equal_boundaries`.

    Raises:
        ValueError: Naming the one-based cells (in strip order) that are not
            fully inside the image.
    """
    missing = [index + 1 for index, cell in enumerate(cell_slices(boundaries)) if not covered[:, cell].all()]
    if missing:
        raise ValueError(
            f"Cell(s) {missing} of the selection extend outside the image; move the selection so every step is inside."
        )


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


def cell_slices(boundaries: np.ndarray) -> list[slice]:
    """Return the measured part of each cell along the wedge axis.

    Only the central ``CELL_FILL`` fraction of a cell is measured; the margin
    absorbs selections that do not line up with the step edges exactly.
    """
    slices = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        margin = int((end - start) * (1.0 - CELL_FILL) / 2.0)
        slices.append(slice(int(start) + margin, int(end) - margin))
    return slices


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
    percentiles = [np.percentile(crop[:, cell], [16, 50, 84]) for cell in cell_slices(boundaries)]
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
