"""Analyze selected Stouffer T2115 strips and convert signals to density."""

from __future__ import annotations

import numpy as np

from .models import AnalysisResult, AnalysisSettings, Selection, StepMeasurement

# Nominal optical densities of the 21-step wedge, spaced by 0.15 density units.
T2115_DENSITIES = np.array([0.05 + 0.15 * index for index in range(21)], dtype=np.float64)


def analyze_selection(
    image: np.ndarray,
    selection: Selection,
    settings: AnalysisSettings | None = None,
) -> AnalysisResult:
    """Analyze a selected step-wedge strip and build density measurements.

    Args:
        image: Two-dimensional luminance image indexed as rows then columns.
        selection: Rectangle enclosing one complete step wedge in image
            coordinates.
        settings: Step count and signal-reference configuration. Defaults to
            :class:`AnalysisSettings` when omitted.

    Returns:
        The clipped selection, profiles, detected boundaries, reference value,
        and one measurement per wedge step.

    Raises:
        ValueError: If the selection cannot resolve the requested steps or the
            configured reference mode is unsupported.
    """
    settings = settings or AnalysisSettings()
    clipped_selection = selection.clipped(image.shape[1], image.shape[0])

    if not clipped_selection.is_large_enough(settings.step_count):
        raise ValueError("The selected region is too small to resolve 21 steps.")

    crop = image[
        clipped_selection.y0 : clipped_selection.y1,
        clipped_selection.x0 : clipped_selection.x1,
    ]
    if min(crop.shape) < 16:
        raise ValueError("The selected strip is too thin to analyze reliably.")

    # The long dimension runs along the wedge. Taking a median across its short
    # dimension suppresses dust and local scanner noise in the 1-D profile.
    orientation = "horizontal" if crop.shape[1] >= crop.shape[0] else "vertical"
    profile = np.median(crop, axis=0 if orientation == "horizontal" else 1).astype(np.float64)

    if profile.size < settings.step_count * 6:
        raise ValueError("The selected strip is too short. Include the full 21-step wedge.")

    # Scale smoothing with the selected strip while retaining a centered,
    # odd-width window for short profiles.
    smoothing_window = max(5, _nearest_odd(profile.size // 60))
    smoothed_profile = moving_average(profile, smoothing_window)
    boundaries = find_step_boundaries(smoothed_profile, settings.step_count)
    signals = measure_segment_signals(crop, boundaries, orientation)
    densities, reference_value = calculate_densities(signals, settings.reference_mode)
    measurements = build_measurements(signals, densities)

    return AnalysisResult(
        selection=clipped_selection,
        orientation=orientation,
        boundaries=boundaries,
        profile=profile,
        smoothed_profile=smoothed_profile,
        measurements=measurements,
        reference_mode=settings.reference_mode,
        reference_value=reference_value,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a one-dimensional signal with a centered moving average.

    Args:
        values: Samples to smooth.
        window: Requested kernel width; values below one are promoted to one,
            and even widths are promoted to the next odd number.

    Returns:
        The floating-point convolution result using zero padding at the edges.
    """
    # Odd kernel widths place the current sample at the center of the window.
    window = max(1, _nearest_odd(window))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def find_step_boundaries(profile: np.ndarray, step_count: int) -> np.ndarray:
    """Locate step transitions in a smoothed wedge profile.

    Args:
        profile: One-dimensional signal along the length of the wedge.
        step_count: Number of expected constant-signal regions.

    Returns:
        Integer boundaries including zero and ``len(profile)``. When reliable
        transitions cannot be found, the profile is divided equally.
    """
    # Strong changes in adjacent samples indicate transitions between steps;
    # smoothing reduces the influence of isolated noise spikes.
    edges = np.abs(np.diff(profile))
    edge_strength = moving_average(edges, max(3, _nearest_odd(len(edges) // 80)))
    candidate_indices = local_maxima(edge_strength)

    # Ranking by strength finds prominent transitions first, while the minimum
    # gap prevents multiple peaks around one physical edge from being selected.
    minimum_gap = max(3, len(profile) // (step_count * 2))
    ranked_candidates = sorted(
        candidate_indices,
        key=lambda index: edge_strength[index],
        reverse=True,
    )

    chosen: list[int] = []
    for index in ranked_candidates:
        # Edge convolution is least trustworthy next to the profile endpoints.
        if index < 1 or index > len(edges) - 2:
            continue
        if all(abs(index - existing) >= minimum_gap for existing in chosen):
            chosen.append(index)
        if len(chosen) == step_count - 1:
            break

    if len(chosen) != step_count - 1:
        # A complete segmentation requires one interior boundary per transition.
        return equal_boundaries(len(profile), step_count)

    # ``np.diff`` positions transitions between samples, hence the +1 offset.
    boundaries = np.array([0, *sorted(index + 1 for index in chosen), len(profile)], dtype=int)
    widths = np.diff(boundaries)
    minimum_width = max(2, len(profile) // (step_count * 6))
    if np.any(widths < minimum_width):
        # Reject implausibly narrow regions even when enough peaks were found.
        return equal_boundaries(len(profile), step_count)

    return boundaries


def local_maxima(values: np.ndarray) -> np.ndarray:
    """Return indices whose values are not lower than either neighbor.

    Args:
        values: One-dimensional samples in which to locate peaks.

    Returns:
        Integer indices of interior local maxima, or an empty array when fewer
        than three samples are available.
    """
    if len(values) < 3:
        return np.array([], dtype=int)
    maxima_mask = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    # The mask starts at the second input sample, so restore its index offset.
    return np.flatnonzero(maxima_mask) + 1


def equal_boundaries(length: int, step_count: int) -> np.ndarray:
    """Divide a profile into equally sized fallback regions.

    Args:
        length: Number of samples in the profile.
        step_count: Number of regions to create.

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


def measure_segment_signals(
    crop: np.ndarray,
    boundaries: np.ndarray,
    orientation: str,
) -> np.ndarray:
    """Measure a representative luminance signal within each wedge step.

    Args:
        crop: Two-dimensional image region containing the wedge.
        boundaries: Consecutive segment boundaries along the wedge axis.
        orientation: ``"horizontal"`` when boundaries address columns;
            otherwise, boundaries address rows.

    Returns:
        One median luminance value per boundary interval, in spatial order.
    """
    signals: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_length = end - start
        trim = max(0, int(round(segment_length * 0.1)))
        # Avoid transition pixels at both edges, but retain at least three
        # samples along the wedge axis for narrow segments.
        if segment_length - (trim * 2) >= 3:
            start += trim
            end -= trim

        if orientation == "horizontal":
            region = crop[:, start:end]
        else:
            region = crop[start:end, :]

        # A median is robust to dust, scratches, and isolated hot pixels.
        signals.append(float(np.median(region)))

    return np.array(signals, dtype=np.float64)


def calculate_densities(signals: np.ndarray, reference_mode: str) -> tuple[np.ndarray, float]:
    """Convert scanner signals to optical-density-like logarithmic values.

    Args:
        signals: Measured luminance values for the wedge steps.
        reference_mode: ``"selection_max"`` for relative density or
            ``"full_scale"`` for a 16-bit reference value of 65535.

    Returns:
        A tuple containing densities in signal order and the reference signal
        used in their calculation.

    Raises:
        ValueError: If ``reference_mode`` is unsupported.
    """
    # The lower bound prevents division by zero and infinite logarithms for
    # completely black or invalid samples.
    safe_signals = np.clip(signals, 1.0, None)
    if reference_mode == "selection_max":
        reference_value = float(np.max(safe_signals, initial=1.0))
    elif reference_mode == "full_scale":
        reference_value = 65535.0
    else:
        raise ValueError(f"Unsupported reference mode: {reference_mode}")

    # Optical density is the base-10 logarithm of reference over transmission.
    densities = np.log10(reference_value / safe_signals)
    return densities, reference_value


def build_measurements(
    signals: np.ndarray,
    densities: np.ndarray,
) -> tuple[StepMeasurement, ...]:
    """Pair measured signals with T2115 exposure metadata for plotting.

    Args:
        signals: Step signals in their spatial order within the selected strip.
        densities: Calculated densities in the same spatial order.

    Returns:
        Measurements ordered by increasing calculated density, with one-based
        graph and spatial indices.
    """
    # Sorting makes the plotted curve independent of which physical end of the
    # wedge appears first in the selected image.
    order = np.argsort(densities)
    # The densest wedge step transmits the least exposure and therefore maps to
    # zero relative log exposure at the start of the print-density curve.
    ordered_wedge_densities = T2115_DENSITIES[::-1]
    ordered_log_exposure = ordered_wedge_densities.max() - ordered_wedge_densities

    measurements: list[StepMeasurement] = []
    for graph_index, spatial_index in enumerate(order, start=1):
        measurements.append(
            StepMeasurement(
                graph_index=graph_index,
                spatial_index=int(spatial_index) + 1,
                signal=float(signals[spatial_index]),
                density=float(densities[spatial_index]),
                relative_log_exposure=float(ordered_log_exposure[graph_index - 1]),
                wedge_density=float(ordered_wedge_densities[graph_index - 1]),
            )
        )

    return tuple(measurements)


def _nearest_odd(value: int) -> int:
    """Return ``value`` when odd, or the next greater integer when even.

    Args:
        value: Integer to normalize for use as a centered kernel width.

    Returns:
        An odd integer equal to ``value`` or ``value + 1``.
    """
    return value if value % 2 else value + 1
