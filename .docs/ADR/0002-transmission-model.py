"""Model for ADR 0002: ISO 6846 endpoints when a print is read in transmission.

Run with ``python .docs/ADR/0002-transmission-model.py``. NumPy only. Reuses
the curve and crossing helpers from ``0001-iso-r-benchmark.py``.

Print optics (Williams and Clapper 1953, simplified two-flux form, no
oblique-path factor): a silver image of single-pass transmittance T in gelatin
over a diffuse white base of reflectance rho, with surface reflectance r_s and
internal gelatin/air reflectance r_i for diffuse light:

    R = r_s + (1 - r_s) (1 - r_i) rho T^2 / (1 - r_i rho T^2)

A transmission reading of the print sees D_base + D_T. The base is assumed an
additive, uniform diffuser, so it cancels in Dmin-relative quantities and only
D_T matters for the endpoints.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("0001-iso-r-benchmark.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def reflection_density(d_t, r_s, r_i=0.6, rho=0.9):
    t2 = 10.0 ** (-2 * np.asarray(d_t, float))
    return -np.log10(r_s + (1 - r_s) * (1 - r_i) * rho * t2 / (1 - r_i * rho * t2))


def ler(x, d):
    return bench.ler_from(x, d, d[0], d[-1], bench.linear_crossing)


def main():
    x = bench.FINE
    print("Silver image: D_T(x) = Dmax_T * Phi((x - 1.5) / s), single-pass transmission density.\n")
    print(f"{'surface':>10} {'Dmax_T':>6} {'s':>5} | {'D_R min':>7} {'D_R max':>7} {'R refl':>6} | {'R trans':>7} {'delta':>6} | reflection density at the transmission endpoints")
    for label, r_s in (("glossy", 0.005), ("semi-matte", 0.02), ("matte", 0.05)):
        for dmax_t in (1.2, 1.6, 2.0):
            for s in (0.26, 0.41):
                d_t = bench.paper_curve(x, 0.0, dmax_t, s, 1.0)
                d_r = reflection_density(d_t, r_s)
                r_refl, r_trans = 100 * ler(x, d_r), 100 * ler(x, d_t)
                lo = bench.linear_crossing(x, d_t, d_t[0] + 0.04, False)
                hi = bench.linear_crossing(x, d_t, d_t[0] + 0.9 * (d_t[-1] - d_t[0]), True)
                dr_lo, dr_hi = np.interp(lo, x, d_r), np.interp(hi, x, d_r)
                print(f"{label:>10} {dmax_t:6.1f} {s:5.2f} | {d_r[0]:7.2f} {d_r[-1]:7.2f} {r_refl:6.1f} | {r_trans:7.1f} {r_trans - r_refl:+6.1f} | "
                      f"{dr_lo - d_r[0]:.3f} above Dmin, {(dr_hi - d_r[0]) / (d_r[-1] - d_r[0]) * 100:.0f}% of DN")
    gain = (reflection_density(0.02, 0.005) - reflection_density(0.0, 0.005)) / 0.02
    print(f"\nSlope dD_R/dD_T near Dmin, glossy: {gain:.2f}")


if __name__ == "__main__":
    main()
