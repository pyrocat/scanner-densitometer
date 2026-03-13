from __future__ import annotations

import numpy as np

from .models import AnalysisResult, AnalysisSettings, Selection, StepMeasurement

T2115_DENSITIES = np.array([0.05 + 0.15 * index for index in range(21)], dtype=np.float64)


def analyze_selection(
    image: np.ndarray,
    selection: Selection,
    settings: AnalysisSettings | None = None,
) -> AnalysisResult:
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

    orientation = "horizontal" if crop.shape[1] >= crop.shape[0] else "vertical"
    profile = np.median(crop, axis=0 if orientation == "horizontal" else 1).astype(np.float64)

    if profile.size < settings.step_count * 6:
        raise ValueError("The selected strip is too short. Include the full 21-step wedge.")

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
    window = max(1, _nearest_odd(window))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def find_step_boundaries(profile: np.ndarray, step_count: int) -> np.ndarray:
    edges = np.abs(np.diff(profile))
    edge_strength = moving_average(edges, max(3, _nearest_odd(len(edges) // 80)))
    candidate_indices = local_maxima(edge_strength)

    minimum_gap = max(3, len(profile) // (step_count * 2))
    ranked_candidates = sorted(
        candidate_indices,
        key=lambda index: edge_strength[index],
        reverse=True,
    )

    chosen: list[int] = []
    for index in ranked_candidates:
        if index < 1 or index > len(edges) - 2:
            continue
        if all(abs(index - existing) >= minimum_gap for existing in chosen):
            chosen.append(index)
        if len(chosen) == step_count - 1:
            break

    if len(chosen) != step_count - 1:
        return equal_boundaries(len(profile), step_count)

    boundaries = np.array([0, *sorted(index + 1 for index in chosen), len(profile)], dtype=int)
    widths = np.diff(boundaries)
    minimum_width = max(2, len(profile) // (step_count * 6))
    if np.any(widths < minimum_width):
        return equal_boundaries(len(profile), step_count)

    return boundaries


def local_maxima(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)
    maxima_mask = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    return np.flatnonzero(maxima_mask) + 1


def equal_boundaries(length: int, step_count: int) -> np.ndarray:
    boundaries = np.rint(np.linspace(0, length, step_count + 1)).astype(int)
    boundaries[0] = 0
    boundaries[-1] = length
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
    boundaries[-1] = length
    return boundaries


def measure_segment_signals(
    crop: np.ndarray,
    boundaries: np.ndarray,
    orientation: str,
) -> np.ndarray:
    signals: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_length = end - start
        trim = max(0, int(round(segment_length * 0.1)))
        if segment_length - (trim * 2) >= 3:
            start += trim
            end -= trim

        if orientation == "horizontal":
            region = crop[:, start:end]
        else:
            region = crop[start:end, :]

        signals.append(float(np.median(region)))

    return np.array(signals, dtype=np.float64)


def calculate_densities(signals: np.ndarray, reference_mode: str) -> tuple[np.ndarray, float]:
    safe_signals = np.clip(signals, 1.0, None)
    if reference_mode == "selection_max":
        reference_value = float(np.max(safe_signals, initial=1.0))
    elif reference_mode == "full_scale":
        reference_value = 65535.0
    else:
        raise ValueError(f"Unsupported reference mode: {reference_mode}")

    densities = np.log10(reference_value / safe_signals)
    return densities, reference_value


def build_measurements(
    signals: np.ndarray,
    densities: np.ndarray,
) -> tuple[StepMeasurement, ...]:
    order = np.argsort(densities)
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
    return value if value % 2 else value + 1
