"""Effective ISO(R) estimate from a paper curve, using ISO 6846:1992 endpoints.

See ``.docs/ADR/0001-iso-r-from-step-wedge-curve.md``. The scanner workflow
does not meet the standard's densitometry or exposure conditions, so this is
an estimate that borrows the standard's definitions, never an ISO range.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

import numpy as np

from .models import AnalysisResult

# Steps at each end of the curve whose densities stay within this range are
# treated as the plateau (paper white or Dmax) and averaged.
PLATEAU_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class IsoRangeEstimate:
    """Effective ISO(R) estimate and the intermediate values behind it.

    Attributes:
        dmin: Density of the unexposed paper used for the endpoints.
        dmin_source: ``"supplied"`` from an unexposed patch (clause 5.6.2) or
            ``"estimated"`` as the mean of the light plateau.
        dmax: Mean density of the dark plateau (clause 5.6.3).
        log_ht: Log exposure at density ``dmin + 0.04``.
        log_hs: Log exposure at density ``dmin + 0.90 (dmax - dmin)``.
        ler: ``log_hs - log_ht`` rounded to two decimals, ties away from zero.
        raw_r: ``100 (log_hs - log_ht)`` before rounding.
        iso_range: ISO 6846 table 2 class (40 to 190), or ``None`` outside it.
        light_plateau_steps: Steps averaged for the estimated Dmin.
        dark_plateau_steps: Steps averaged for Dmax.
        warnings: Human-readable caveats about the estimate.
    """

    dmin: float
    dmin_source: Literal["supplied", "estimated"]
    dmax: float
    log_ht: float
    log_hs: float
    ler: float
    raw_r: float
    iso_range: int | None
    light_plateau_steps: int
    dark_plateau_steps: int
    warnings: tuple[str, ...] = ()

    @property
    def density_ht(self) -> float:
        """Density defining the toe endpoint HT."""
        return self.dmin + 0.04

    @property
    def density_hs(self) -> float:
        """Density defining the shoulder endpoint HS."""
        return self.dmin + 0.90 * (self.dmax - self.dmin)


def iso_range_from_analysis(result: AnalysisResult, dmin: float | None = None) -> IsoRangeEstimate:
    """Estimate ISO(R) from a strip analysis, refusing unreliable readings.

    Args:
        result: Analysis of a print exposed through the wedge.
        dmin: Density of an unexposed, processed patch measured with the same
            density mapping; ``None`` estimates it from the light plateau.

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
    return iso_range(result.x_values, result.y_values, dmin)


def iso_range(
    log_exposures: np.ndarray,
    densities: np.ndarray,
    dmin: float | None = None,
) -> IsoRangeEstimate:
    """Estimate ISO(R) from sampled densities versus log exposure.

    Args:
        log_exposures: Relative log exposure of each sample, any order.
        densities: Density of each sample.
        dmin: Density of unexposed paper, or ``None`` to use the light plateau.

    Returns:
        The estimate with its endpoints and plateau statistics.

    Raises:
        ValueError: If a plateau is too short or an endpoint is not bracketed.
    """
    order = np.argsort(np.asarray(log_exposures, dtype=np.float64))
    x = np.asarray(log_exposures, dtype=np.float64)[order]
    raw = np.asarray(densities, dtype=np.float64)[order]
    # Plateaus are judged on the measured densities: a monotone fit would pool
    # a still-rising end into a flat run and manufacture a plateau.
    light_mean, light_steps = plateau_mean(raw)
    dmax, dark_steps = plateau_mean(raw[::-1])
    # The monotone fit removes noise-induced reversals so each endpoint
    # density is crossed exactly once.
    d = pava(raw)
    if dark_steps < 2:
        raise ValueError("ISO(R) refused: the curve does not reach Dmax (dark plateau shorter than two steps).")
    warnings: list[str] = []
    if dmin is None:
        if light_steps < 2:
            raise ValueError(
                "ISO(R) refused: the curve does not reach paper white (light plateau shorter than two "
                "steps). Measure an unexposed patch instead."
            )
        dmin = light_mean
        source: Literal["supplied", "estimated"] = "estimated"
        warnings.append("Dmin estimated from the light plateau; measure an unexposed patch for soft papers.")
    else:
        source = "supplied"

    log_ht = cubic_crossing(x, d, dmin + 0.04, from_dark_end=False)
    log_hs = cubic_crossing(x, d, dmin + 0.90 * (dmax - dmin), from_dark_end=True)
    if log_ht is None or log_hs is None:
        raise ValueError("ISO(R) refused: an endpoint density is not bracketed by the measured steps.")

    ler = round_ler(log_hs - log_ht)
    return IsoRangeEstimate(
        dmin=float(dmin),
        dmin_source=source,
        dmax=float(dmax),
        log_ht=float(log_ht),
        log_hs=float(log_hs),
        ler=float(ler),
        raw_r=float(100.0 * (log_hs - log_ht)),
        iso_range=iso_range_class(ler),
        light_plateau_steps=light_steps,
        dark_plateau_steps=dark_steps,
        warnings=tuple(warnings),
    )


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
