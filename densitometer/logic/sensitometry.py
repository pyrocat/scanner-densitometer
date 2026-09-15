"""Effective ISO(R) estimate from a paper curve, using ISO 6846:1992 endpoints.

See ``.docs/ADR/0001-iso-r-from-step-wedge-curve.md`` for the endpoints and
``.docs/ADR/0003-iso-r-validity-checks.md`` for the acceptance policies. The
scanner workflow does not meet the standard's densitometry or exposure
conditions, so this is an estimate that borrows the standard's definitions,
never an ISO range.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Sequence

import numpy as np

from .models import AnalysisResult

# Readings at each end of the curve whose densities stay within this range are
# treated as the plateau (paper white or Dmax) and averaged.
PLATEAU_TOLERANCE = 0.02
# Larger of the two per-step density errors assumed by the ADR 0001 error
# model (0.005 and 0.01 D). The thresholds below are multiples of it; their
# false-refusal and false-acceptance rates are scored in the ADR 0003
# benchmark, which is where any change must be justified.
ASSUMED_STEP_ERROR = 0.01
# The shoulder endpoint HS sits 0.10 DN below Dmax. That margin must be at
# least the standard's own toe offset (0.04 D, twice the plateau tolerance)
# for HS to be resolved against the dark plateau rather than its scatter.
MIN_NET_DENSITY = 0.04 / 0.10
# Exposure never lowers density, so an unexposed patch cannot read darker
# than the strip's light end beyond measurement error: four assumed step
# errors between two independent readings.
SUPPLIED_DMIN_TOLERANCE = 4 * ASSUMED_STEP_ERROR
# Largest change the monotone fit may make to any step. Two assumed step
# errors earn a warning; five refuse the curve, because a reversal that size
# is not measurement noise but dust, a misplaced cell or a damaged sample.
FIT_ADJUSTMENT_WARNING = 2 * ASSUMED_STEP_ERROR
MAX_FIT_ADJUSTMENT = 5 * ASSUMED_STEP_ERROR
# Exposure span (log E) the dark plateau must cover before Dmax counts as
# established rather than provisional: a decade of exposure with no
# systematic increase beyond the plateau tolerance.
DMAX_PLATEAU_SPAN = 1.0
# Fewest readings a plateau needs: a single reading is a point, not a run.
MIN_PLATEAU_READINGS = 2


@dataclass(frozen=True, slots=True)
class DarkPatch:
    """An overexposed patch used as evidence for Dmax (ISO 6846 clause 5.6.3).

    Attributes:
        log_exposure_offset: Log exposure of the patch beyond the most exposed
            wedge step (step 1). An unmasked patch given the strip's exposure
            time sits at the density of wedge step 1 beyond it (0.05 for the
            nominal T2115); exposing it ``k`` times longer adds ``log10(k)``.
        density: Density of the patch, measured with the same mapping as the
            strip it accompanies.
    """

    log_exposure_offset: float
    density: float


@dataclass(frozen=True, slots=True)
class IsoRangeEstimate:
    """Effective ISO(R) estimate and the intermediate values behind it.

    Attributes:
        dmin: Density of the unexposed paper used for the endpoints.
        dmin_source: ``"supplied"`` from an unexposed patch (clause 5.6.2) or
            ``"estimated"`` as the mean of the light plateau.
        dmax: Mean density of the dark plateau (clause 5.6.3).
        dmax_source: ``"established"`` when the plateau spans at least
            ``DMAX_PLATEAU_SPAN`` of exposure, else ``"provisional"``: a
            plateau seen only over a short exposure interval is a lower
            bound on Dmax, not evidence that density stopped increasing.
        dmax_span: Log exposure interval covered by the dark plateau readings.
        log_ht: Log exposure at density ``dmin + 0.04``.
        log_hs: Log exposure at density ``dmin + 0.90 (dmax - dmin)``.
        ler: ``log_hs - log_ht`` rounded to two decimals, ties away from zero.
        raw_r: ``100 (log_hs - log_ht)`` before rounding.
        iso_range: ISO 6846 table 2 class (40 to 190), or ``None`` outside it.
        light_plateau_steps: Steps averaged for the estimated Dmin.
        dark_plateau_steps: Readings (steps and patches) averaged for Dmax.
        dark_patches_used: Overexposed patch readings inside the dark plateau.
        fit_adjustment: Largest change the monotone fit made to any step.
        adjusted_steps: Number of steps the monotone fit changed.
        warnings: Human-readable caveats about the estimate.
    """

    dmin: float
    dmin_source: Literal["supplied", "estimated"]
    dmax: float
    dmax_source: Literal["established", "provisional"]
    dmax_span: float
    log_ht: float
    log_hs: float
    ler: float
    raw_r: float
    iso_range: int | None
    light_plateau_steps: int
    dark_plateau_steps: int
    dark_patches_used: int
    fit_adjustment: float
    adjusted_steps: int
    warnings: tuple[str, ...] = ()

    @property
    def density_ht(self) -> float:
        """Density defining the toe endpoint HT."""
        return self.dmin + 0.04

    @property
    def density_hs(self) -> float:
        """Density defining the shoulder endpoint HS."""
        return self.dmin + 0.90 * (self.dmax - self.dmin)

    @property
    def net_density(self) -> float:
        """Maximum net density DN = Dmax - Dmin (clause 5.6.4)."""
        return self.dmax - self.dmin


def iso_range_from_analysis(
    result: AnalysisResult,
    dmin: float | None = None,
    dark_patches: Sequence[DarkPatch] = (),
) -> IsoRangeEstimate:
    """Estimate ISO(R) from a strip analysis, refusing unreliable readings.

    Args:
        result: Analysis of a print exposed through the wedge.
        dmin: Density of an unexposed, processed patch measured with the same
            density mapping; ``None`` estimates it from the light plateau.
        dark_patches: Overexposed patches measured with the same density
            mapping, with their exposure offsets beyond wedge step 1.

    Raises:
        ValueError: If any step is clipped by the scanner or clamped by the
            calibration, or if the curve cannot support the endpoints.
    """
    clipped = [m.step for m in result.measurements if m.clipped]
    if clipped:
        raise ValueError(f"ISO(R) refused: step(s) {clipped} clipped at scanner white.")
    clamped = [m.step for m in result.measurements if m.clamped]
    if clamped:
        raise ValueError(f"ISO(R) refused: step(s) {clamped} outside the calibration range.")
    most_exposed = float(result.x_values.max())
    patches = [(most_exposed + patch.log_exposure_offset, patch.density) for patch in dark_patches]
    return iso_range(result.x_values, result.y_values, dmin, patches)


def iso_range(
    log_exposures: np.ndarray,
    densities: np.ndarray,
    dmin: float | None = None,
    dark_patches: Sequence[tuple[float, float]] = (),
) -> IsoRangeEstimate:
    """Estimate ISO(R) from sampled densities versus log exposure.

    Args:
        log_exposures: Relative log exposure of each sample, any order.
        densities: Density of each sample.
        dmin: Density of unexposed paper, or ``None`` to use the light plateau.
        dark_patches: ``(log exposure, density)`` of overexposed patches on
            the same exposure axis, all beyond the most exposed sample. They
            extend the dark end of the curve when judging the Dmax plateau.

    Returns:
        The estimate with its endpoints and plateau statistics.

    Raises:
        ValueError: If the inputs break the contract (matching finite
            one-dimensional arrays, distinct exposures), a plateau is too
            short, the fit alters a step beyond ``MAX_FIT_ADJUSTMENT``, the
            supplied Dmin contradicts the strip, the net density is below
            ``MIN_NET_DENSITY``, or an endpoint is not bracketed.
    """
    x, raw = _check_curve_inputs(log_exposures, densities)
    patch_x, patch_y = _check_patch_inputs(dark_patches, float(x[-1]))
    if dmin is not None and not np.isfinite(dmin):
        raise ValueError("ISO(R) refused: the supplied Dmin is not a finite number.")

    warnings: list[str] = []
    # The monotone fit removes noise-induced reversals so each endpoint
    # density is crossed exactly once. Its adjustments measure how far the
    # data is from a curve at all.
    d = pava(raw)
    adjustment = np.abs(d - raw)
    fit_adjustment = float(adjustment.max())
    adjusted_steps = int(np.count_nonzero(adjustment > 1e-9))
    if fit_adjustment > MAX_FIT_ADJUSTMENT:
        raise ValueError(
            f"ISO(R) refused: the monotone fit changes a step by {fit_adjustment:.3f} D "
            f"(limit {MAX_FIT_ADJUSTMENT:.2f} D); a reversal that large is not measurement noise. "
            "Check the strip for dust, misplaced cells or a damaged sample."
        )
    if fit_adjustment > FIT_ADJUSTMENT_WARNING:
        warnings.append(
            f"Monotone fit adjusted {adjusted_steps} step(s) by up to {fit_adjustment:.3f} D; "
            "check those steps for dust or cell misalignment."
        )

    # Plateaus are judged on the measured densities: a monotone fit would pool
    # a still-rising end into a flat run and manufacture a plateau. Patches
    # extend the dark end so the plateau can be judged over more exposure.
    light_mean, light_steps = plateau_mean(raw)
    dark_x = np.concatenate([x, patch_x])
    dark_y = np.concatenate([raw, patch_y])
    dmax, dark_steps = plateau_mean(dark_y[::-1])
    dark_span = float(dark_x[-1] - dark_x[-dark_steps])
    patches_used = min(dark_steps, len(patch_x))
    if dark_steps < MIN_PLATEAU_READINGS:
        if len(patch_x):
            rise = float(dark_y[-1] - dark_y[-2])
            raise ValueError(
                f"ISO(R) refused: density still rises by {rise:.3f} D at the last overexposed patch "
                f"(+{patch_x[-1] - x[-1]:.2f} log E beyond step 1), so Dmax is not reached. "
                "Expose a patch at a higher exposure."
            )
        raise ValueError("ISO(R) refused: the curve does not reach Dmax (dark plateau shorter than two steps).")
    if dark_span + 1e-9 >= DMAX_PLATEAU_SPAN:
        dmax_source: Literal["established", "provisional"] = "established"
    else:
        dmax_source = "provisional"
        evidence = f"{dark_steps} readings over {dark_span:.2f} log E"
        if patches_used:
            evidence += f", including {patches_used} overexposed patch(es)"
        warnings.append(
            f"Dmax is provisional ({evidence}) and is a lower bound; a plateau over at least "
            f"{DMAX_PLATEAU_SPAN:.1f} log E is needed to establish it. Measure an overexposed patch."
        )

    if dmin is None:
        if light_steps < MIN_PLATEAU_READINGS:
            raise ValueError(
                "ISO(R) refused: the curve does not reach paper white (light plateau shorter than two "
                "steps). Measure an unexposed patch instead."
            )
        dmin = light_mean
        source: Literal["supplied", "estimated"] = "estimated"
        warnings.append("Dmin estimated from the light plateau; measure an unexposed patch for soft papers.")
    else:
        source = "supplied"
        if dmin > light_mean + SUPPLIED_DMIN_TOLERANCE:
            raise ValueError(
                f"ISO(R) refused: the supplied Dmin {dmin:.3f} exceeds the strip's light end "
                f"({light_mean:.3f}, mean of {light_steps} step(s)) by more than {SUPPLIED_DMIN_TOLERANCE:.2f} D. "
                "Exposure cannot lower density, so the patch or the strip was measured wrongly."
            )

    net_density = float(dmax - dmin)
    if net_density < MIN_NET_DENSITY:
        raise ValueError(
            f"ISO(R) refused: net density {net_density:.3f} D (Dmax {dmax:.3f}, Dmin {dmin:.3f}) is below the "
            f"{MIN_NET_DENSITY:.2f} D needed to resolve the endpoints; the sample shows no usable curve."
        )

    log_ht = cubic_crossing(x, d, dmin + 0.04, from_dark_end=False)
    log_hs = cubic_crossing(x, d, dmin + 0.90 * (dmax - dmin), from_dark_end=True)
    if log_ht is None or log_hs is None:
        raise ValueError("ISO(R) refused: an endpoint density is not bracketed by the measured steps.")
    # Defensive: with a non-decreasing fit and HS above HT this cannot fail,
    # so a failure means the inputs escaped the checks above.
    if not (np.isfinite(log_ht) and np.isfinite(log_hs)) or log_hs <= log_ht:
        raise ValueError("ISO(R) refused: the endpoint exposures are not in order (HS must lie beyond HT).")

    ler = round_ler(log_hs - log_ht)
    return IsoRangeEstimate(
        dmin=float(dmin),
        dmin_source=source,
        dmax=float(dmax),
        dmax_source=dmax_source,
        dmax_span=dark_span,
        log_ht=float(log_ht),
        log_hs=float(log_hs),
        ler=float(ler),
        raw_r=float(100.0 * (log_hs - log_ht)),
        iso_range=iso_range_class(ler),
        light_plateau_steps=light_steps,
        dark_plateau_steps=dark_steps,
        dark_patches_used=patches_used,
        fit_adjustment=fit_adjustment,
        adjusted_steps=adjusted_steps,
        warnings=tuple(warnings),
    )


def _check_curve_inputs(log_exposures: np.ndarray, densities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate the direct-caller contract and return the curve sorted by exposure."""
    x = np.asarray(log_exposures, dtype=np.float64)
    raw = np.asarray(densities, dtype=np.float64)
    if x.ndim != 1 or raw.shape != x.shape:
        raise ValueError("ISO(R) refused: log exposures and densities must be matching one-dimensional arrays.")
    if len(x) < 3:
        raise ValueError("ISO(R) refused: at least three samples are needed.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(raw))):
        raise ValueError("ISO(R) refused: log exposures and densities must be finite.")
    order = np.argsort(x, kind="stable")
    x, raw = x[order], raw[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("ISO(R) refused: log exposures must be distinct.")
    return x, raw


def _check_patch_inputs(dark_patches: Sequence[tuple[float, float]], most_exposed: float) -> tuple[np.ndarray, np.ndarray]:
    """Validate overexposed patches and return them sorted by exposure."""
    if len(dark_patches) == 0:
        return np.empty(0), np.empty(0)
    patches = np.asarray(dark_patches, dtype=np.float64)
    if patches.ndim != 2 or patches.shape[1] != 2 or not np.all(np.isfinite(patches)):
        raise ValueError("ISO(R) refused: overexposed patches must be finite (log exposure, density) pairs.")
    patches = patches[np.argsort(patches[:, 0], kind="stable")]
    if patches[0, 0] <= most_exposed:
        raise ValueError("ISO(R) refused: overexposed patches must lie beyond the most exposed wedge step.")
    if np.any(np.diff(patches[:, 0]) <= 0):
        raise ValueError("ISO(R) refused: overexposed patches must have distinct exposures.")
    return patches[:, 0], patches[:, 1]


def round_ler(value: float) -> Decimal:
    """Round LER to two decimals (clause 6.2.1), ties away from zero.

    The standard gives no tie rule; rounding the decimal representation avoids
    binary artefacts such as 0.345 stored as 0.34499...
    """
    return Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def iso_range_class(ler: Decimal) -> int | None:
    """Classify a two-decimal LER by ISO 6846 table 2 (R40 to R190 in 0.10 bands)."""
    cents = int(ler * 100)
    if 35 <= cents <= 194:
        return 10 * ((cents + 5) // 10)
    return None


def pava(values: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit by pooling adjacent violators."""
    blocks: list[list[float]] = []
    for value in map(float, values):
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            upper, lower = blocks.pop(), blocks.pop()
            weight = lower[1] + upper[1]
            blocks.append([(lower[0] * lower[1] + upper[0] * upper[1]) / weight, weight])
    return np.concatenate([[mean] * int(count) for mean, count in blocks])


def plateau_mean(values: np.ndarray, tolerance: float = PLATEAU_TOLERANCE) -> tuple[float, int]:
    """Mean and length of the longest run from index 0 whose range stays within ``tolerance``."""
    count = 1
    while count < len(values) and values[: count + 1].max() - values[: count + 1].min() <= tolerance:
        count += 1
    return float(values[:count].mean()), count


def cubic_crossing(x: np.ndarray, y: np.ndarray, target: float, from_dark_end: bool) -> float | None:
    """Log exposure where a monotone cubic through the samples reaches ``target``.

    Fritsch-Carlson slopes keep the interpolant monotone between samples, so
    the crossing inside the bracketing interval is unique. Searching from the
    dark end for the shoulder endpoint keeps a flat-topped curve from matching
    a lower interval first.

    Returns:
        The crossing, or ``None`` if no interval brackets ``target``.
    """
    indices = range(len(y) - 2, -1, -1) if from_dark_end else range(len(y) - 1)
    interval = next(
        (i for i in indices if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]),
        None,
    )
    if interval is None:
        return None

    h = np.diff(x)
    slopes = np.diff(y) / h
    m = np.zeros(len(y))
    m[0], m[-1] = slopes[0], slopes[-1]
    for k in range(1, len(y) - 1):
        if slopes[k - 1] * slopes[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / slopes[k - 1] + w2 / slopes[k])

    i, hh = interval, h[interval]
    # Cubic Hermite segment in t on [0, 1], written as polynomial coefficients.
    coefficients = [
        2 * y[i] + hh * m[i] - 2 * y[i + 1] + hh * m[i + 1],
        -3 * y[i] - 2 * hh * m[i] + 3 * y[i + 1] - hh * m[i + 1],
        hh * m[i],
        y[i] - target,
    ]
    roots = np.roots(coefficients)
    real = roots[np.abs(roots.imag) < 1e-9].real
    inside = real[(real >= -1e-9) & (real <= 1.0 + 1e-9)]
    if inside.size == 0:
        return None
    return float(x[i] + float(np.clip(inside[0], 0.0, 1.0)) * hh)
