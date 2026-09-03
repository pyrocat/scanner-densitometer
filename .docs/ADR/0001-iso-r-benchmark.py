"""Benchmark for ADR 0001: recovering ISO 6846 range from a 21-step wedge print.

Run with ``python .docs/ADR/0001-iso-r-benchmark.py``. NumPy only, fixed seed.

Synthetic paper curve (density versus log exposure x, 0 to 3):

    D(x) = Dmin + (Dmax - Dmin) * Phi((x - 1.5) / s) ** a

where Phi is the standard normal CDF, s sets the width (and therefore the
LER) and a skews toe against shoulder (a < 1 long shoulder, a > 1 long toe).
The true LER is read off a 30001-point evaluation of the same curve.

Endpoints follow ISO 6846:1992 clause 6.2 and figure 1: T at Dmin + 0.04, S at
Dmin + 0.90 (Dmax - Dmin). Accuracy target from clause 6.2.3: LER error below
0.01 or 3 %, whichever is greater. Classification follows table 2 after
rounding LER to two decimals, ties away from zero (an explicit choice; the
standard only says "to two decimal places").

Estimators compared on the 21 samples at 0.15 log E spacing, all fed the same
noise draw per trial:

    naive        Dmin/Dmax from the single lightest/darkest step, linear crossing
    pava+lin     pool-adjacent-violators monotone fit, plateau means, linear crossing
    pava+cubic   as above with monotone (Fritsch-Carlson) cubic crossing
    cubic+Dmin   as pava+cubic but Dmin supplied exactly (unexposed patch, clause 5.6.2)

Error models, all assumed rather than measured: Gaussian density error per
step (sigma 0, 0.005, 0.01); scanner flare as a constant offset of 0.3 % of
the paper-white signal, read in relative mode (density above paper white);
uncalibrated wedge whose step densities are each off nominal by N(0, 0.02),
independently (how step-tablet tolerance is specified), while the estimator
assumes nominal 0.15 steps. Modelling the error on the increments instead,
so it accumulates along the wedge, roughly doubles the spread.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

SEED = 20260903
TRIALS = 500
STEPS = np.arange(21) * 0.15
FINE = np.linspace(0.0, 3.0, 30001)


def paper_curve(x, dmin, dmax, s, a):
    phi = 0.5 * (1.0 + np.vectorize(math.erf)((np.asarray(x, float) - 1.5) / (s * math.sqrt(2.0))))
    return dmin + (dmax - dmin) * phi ** a


def pava(y):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for value in map(float, y):
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b, a_ = blocks.pop(), blocks.pop()
            blocks.append([(a_[0] * a_[1] + b[0] * b[1]) / (a_[1] + b[1]), a_[1] + b[1]])
    return np.concatenate([[v] * int(n) for v, n in blocks])


def plateau_mean(d, tol=0.02):
    """Mean of the longest run from index 0 whose max-min stays within tol."""
    run = [d[0]]
    for value in d[1:]:
        if max(run + [value]) - min(run + [value]) <= tol:
            run.append(value)
        else:
            break
    return float(np.mean(run)), len(run)


def bracket(x, y, target, from_dark_end):
    order = range(len(y) - 2, -1, -1) if from_dark_end else range(len(y) - 1)
    for i in order:
        if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
            return i
    return None


def linear_crossing(x, y, target, from_dark_end):
    i = bracket(x, y, target, from_dark_end)
    if i is None:
        return None
    return x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def cubic_crossing(x, y, target, from_dark_end):
    i = bracket(x, y, target, from_dark_end)
    if i is None:
        return None
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(len(y))
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    t = np.linspace(0.0, 1.0, 401)
    hh = h[i]
    hermite = ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * hh * m[i]
               + (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * hh * m[i + 1])
    return x[i] + t[np.argmin(np.abs(hermite - target))] * hh


def ler_from(x, d, dmin, dmax, crossing):
    lo = crossing(x, d, dmin + 0.04, False)
    hi = crossing(x, d, dmin + 0.90 * (dmax - dmin), True)
    return None if lo is None or hi is None else hi - lo


def naive(x, d, dmin=None, dmax=None):
    return ler_from(x, d, d.min(), d.max(), linear_crossing)


def estimate(x, d, crossing, dmin=None, dmax=None):
    """PAVA fit; Dmin and Dmax from the plateaus unless supplied."""
    d = pava(d)
    dmin = plateau_mean(d)[0] if dmin is None else dmin
    dmax = plateau_mean(d[::-1])[0] if dmax is None else dmax
    return ler_from(x, d, dmin, dmax, crossing)


ESTIMATORS = {
    "naive": lambda x, d, dmin: naive(x, d),
    "pava+lin": lambda x, d, dmin: estimate(x, d, linear_crossing),
    "pava+cubic": lambda x, d, dmin: estimate(x, d, cubic_crossing),
    "cubic+Dmin": lambda x, d, dmin: estimate(x, d, cubic_crossing, dmin=dmin),
}


def ler_two_decimals(ler):
    """Clause 6.2.1 rounding with an explicit tie rule: half away from zero."""
    return Decimal(repr(float(ler))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def iso_range_class(ler):
    """ISO 6846 table 2: R40 .. R190 in 0.10 bands, none outside 0.35 .. 1.94."""
    cents = int(ler_two_decimals(ler) * 100)
    if 35 <= cents <= 194:
        return 10 * ((cents + 5) // 10)
    return None


def flare(d, dmin, fraction=0.003):
    """Relative-mode reading (density above paper white) when the scanner signal
    carries a constant offset of ``fraction`` of the paper-white signal."""
    net = d - dmin
    return np.log10((1.0 + fraction) / (10.0 ** -net + fraction))


CURVES = [  # (dmin, dmax, s, a)
    (0.07, 2.10, 0.15, 1.0),
    (0.07, 2.10, 0.26, 1.0),
    (0.07, 2.10, 0.32, 1.0),
    (0.07, 2.10, 0.41, 1.0),
    (0.07, 2.10, 0.50, 1.0),
    (0.07, 2.10, 0.32, 0.5),
    (0.07, 2.10, 0.32, 2.0),
    (0.10, 1.85, 0.26, 1.5),
]


def true_ler(dmin, dmax, s, a):
    return ler_from(FINE, paper_curve(FINE, dmin, dmax, s, a), dmin, dmax, linear_crossing)


def score(errors_ler, true, misses):
    tol = max(0.01, 0.03 * true)
    errors = np.asarray(errors_ler)
    n = len(errors) + misses
    outside = misses + int(np.sum(np.abs(errors) > tol))
    wrong = misses + sum(iso_range_class(true + e) != iso_range_class(true) for e in errors)
    return f"{100*np.mean(errors):+5.1f}/{100*np.std(errors):4.1f} {100*outside/n:4.0f}% {100*wrong/n:4.0f}%"


def main():
    for value, expected in ((0.345, 40), (0.344, None), (0.445, 50), (1.944, 190), (1.945, None), (1.07, 110)):
        assert iso_range_class(value) == expected, (value, iso_range_class(value), expected)

    rng = np.random.default_rng(SEED)
    print(f"seed {SEED}, {TRIALS} trials per cell, identical noise draws across estimators.\n"
          "Cells: mean error / sd in R units, % outside clause 6.2.3 tolerance, % misclassified by table 2.\n")
    for sigma in (0.0, 0.005, 0.01):
        print(f"--- density error sigma = {sigma} ---")
        print(f"{'true':>6} {'class':>5} | " + " | ".join(f"{name:^22}" for name in ESTIMATORS))
        for dmin, dmax, s, a in CURVES:
            true = true_ler(dmin, dmax, s, a)
            base = paper_curve(STEPS, dmin, dmax, s, a)
            results = {name: ([], 0) for name in ESTIMATORS}
            for _ in range(TRIALS if sigma else 1):
                noisy = base + rng.normal(0, sigma, base.size)
                for name, est in ESTIMATORS.items():
                    got = est(STEPS, noisy, dmin)
                    errors, misses = results[name]
                    if got is None:
                        results[name] = (errors, misses + 1)
                    else:
                        errors.append(got - true)
            print(f"{100*true:6.1f} R{iso_range_class(true):<4} | "
                  + " | ".join(score(errs, true, miss) for errs, miss in results.values()))
        print()

    print("--- attribution for the soft curve (R167), noiseless, cubic crossing ---")
    dmin, dmax, s, a = CURVES[4]
    true = true_ler(dmin, dmax, s, a)
    base = paper_curve(STEPS, dmin, dmax, s, a)
    for label, kw in (("plateau Dmin, plateau Dmax", {}), ("exact Dmin, plateau Dmax", {"dmin": dmin}),
                      ("exact Dmin, exact Dmax", {"dmin": dmin, "dmax": dmax})):
        got = estimate(STEPS, base, cubic_crossing, **kw)
        _, n_lo = plateau_mean(pava(base)); _, n_hi = plateau_mean(pava(base)[::-1])
        print(f"{label:28s}: error {100*(got-true):+.1f} R  (light plateau {n_lo} steps, dark plateau {n_hi} steps)")

    print("\n--- flare 0.3 % of paper white, relative-mode reading, noiseless ---")
    print(f"{'true':>6} | {'pava+cubic':>10} | {'cubic+Dmin':>10}   (tolerance)")
    for dmin, dmax, s, a in CURVES[:5]:
        true = true_ler(dmin, dmax, s, a)
        read = flare(paper_curve(STEPS, dmin, dmax, s, a), dmin)
        plateau = estimate(STEPS, read, cubic_crossing)
        supplied = estimate(STEPS, read, cubic_crossing, dmin=0.0)
        print(f"{100*true:6.1f} | {100*(plateau-true):+10.1f} | {100*(supplied-true):+10.1f}   (±{100*max(0.01, 0.03*true):.1f})")

    print("\n--- uncalibrated wedge: each step density off nominal by N(0, 0.02), estimator assumes nominal steps, noiseless ---")
    print(f"{'true':>6} | {'pava+cubic':>14} | {'cubic+Dmin':>14}")
    for dmin, dmax, s, a in (CURVES[1], CURVES[2], CURVES[3]):
        true = true_ler(dmin, dmax, s, a)
        errs = {"pava+cubic": [], "cubic+Dmin": []}
        for _ in range(TRIALS):
            actual_x = STEPS + rng.normal(0, 0.02, STEPS.size)
            d = paper_curve(actual_x, dmin, dmax, s, a)
            errs["pava+cubic"].append(estimate(STEPS, d, cubic_crossing) - true)
            errs["cubic+Dmin"].append(estimate(STEPS, d, cubic_crossing, dmin=dmin) - true)
        print(f"{100*true:6.1f} | " + " | ".join(f"{100*np.mean(e):+6.1f} / sd {100*np.std(e):4.1f}" for e in errs.values()))


if __name__ == "__main__":
    main()
