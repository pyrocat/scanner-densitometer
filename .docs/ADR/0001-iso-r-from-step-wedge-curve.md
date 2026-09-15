# ADR 0001: Estimate effective ISO(R) from the step-wedge curve

- Status: accepted, implemented 2026-09-03 (`densitometer/logic/sensitometry.py`, `tests/test_sensitometry.py`);
  validity checks, overexposed-patch Dmax evidence and a production benchmark added by ADR 0003 (2026-09-16)
- Date: 2026-09-03

## Context

The app plots a paper characteristic curve (density versus relative log
exposure) from a scan of a print made through a Stouffer T2115 step wedge.
We want it to report the effective ISO(R) of the paper under the user's
actual enlarger, filtration, and developer.

### What ISO 6846:1992 specifies

Verified against the text of the second edition (clause numbers below).
The ISO record shows the edition reviewed and confirmed in 2025.

- Clause 6.2: `R = 100 (log10 HS − log10 HT)`. HT is the exposure producing
  density 0.04 above Dmin. HS is the exposure producing density 0.90 DN,
  where DN = Dmax − Dmin is the maximum net density (clause 5.6.4, figure 1).
  So the upper point is `Dmin + 0.90 (Dmax − Dmin)`, not 0.90 Dmax.
- Clause 5.6.2: Dmin is measured on an unexposed sample processed together
  with the exposed one. Clause 5.3.5 requires an unexposed area on the
  test paper.
- Clause 5.6.3: Dmax is the density of the sample whose density no longer
  systematically increases with exposure, that is, the plateau.
- Clause 6.2.1 and table 2: determine LER to two decimal places, then
  classify in 0.10 bands: 0.35 to 0.44 is R40, 0.45 to 0.54 is R50, and so
  on to 1.85 to 1.94 is R190. Values outside 0.35 to 1.94 have no class.
  The standard does not state a tie rule for the two-decimal rounding; we
  round ties away from zero on the decimal representation (0.345 becomes
  0.35, hence R40) and test that.
- Clause 6.2.3: the equipment and process shall keep the absolute error in
  LER below 0.01 or 3 %, whichever is greater.
- Clause 5.5: densities are ISO standard visual reflection densities
  (ISO 5-4 geometry, ISO 5-3 spectral response). Clause 5.3.3: tungsten at
  2 856 K. Clause 5.3.4: stepped modulation increments shall not exceed
  0.15 log H, so the T2115 sits exactly at the permitted limit.
- Introduction: every manufacturer has its own grade system; ISO range was
  introduced to replace, not map onto, grade numbers.

A flatbed scanner reading a print through an enlarger exposure meets none
of the densitometry or exposure conditions. The app therefore reports an
"effective R estimate using ISO 6846 endpoints", never an ISO range. Paper
speed is out of scope: it needs absolute exposure in lux seconds.

### What was verified

`0001-iso-r-benchmark.py` beside this file is a self-contained, seeded
NumPy benchmark (seed 20260903, 500 trials per cell, identical noise draws
across estimators). It samples synthetic paper curves at the wedge's 0.15
log E spacing, applies the clause 6.2 endpoints, and scores each estimator
by mean error, spread, the share of trials outside the clause 6.2.3
tolerance, and the share misclassified by table 2. The curve equation and
the error models are documented in the script. All error models are
assumptions, not measurements of the app.

Estimators: naive (Dmin and Dmax from the single lightest and darkest
step, linear crossing); PAVA + linear (pool-adjacent-violators monotone
fit, plateau means, linear crossing); PAVA + cubic (same with monotone
Fritsch-Carlson cubic crossing); cubic + Dmin (PAVA + cubic with Dmin
supplied from an unexposed patch, clause 5.6.2).

Mean error in R units, and share of trials outside tolerance, at an
assumed 0.005 D per-step error:

| True | Naive | PAVA + linear | PAVA + cubic | Cubic + Dmin |
|---|---|---|---|---|
| R50 | +6.1, 100 % | +3.4, 88 % | +0.8, 26 % | +0.9, 27 % |
| R87 | +5.0, 97 % | +2.8, 59 % | +0.1, 8 % | +0.4, 5 % |
| R107 | +4.7, 79 % | +1.7, 22 % | −0.5, 8 % | +0.1, 2 % |
| R137 | +3.4, 33 % | +0.3, 5 % | −1.7, 14 % | −0.3, 4 % |
| R167 | +0.5, 8 % | −2.8, 18 % | −4.1, 43 % | −0.8, 6 % |

Findings:

- Linear crossing carries a systematic +3 R bias for contrasty papers
  (R50 to R90), which alone exceeds the clause 6.2.3 budget of ±1.5 to
  ±2.6 R there. The monotone cubic removes it and is kept (about fifteen
  lines, covered by tests).
- Averaging the flat run of steps at each end is what makes the estimate
  usable at all. Dmin from the single lightest step is the dominant
  failure of the naive method.
- For soft papers the error comes from the estimated light plateau, not
  from Dmax. Noiseless R167: −5.1 with both ends from plateaus, −1.1 with
  Dmin supplied, 0.0 with both supplied. The toe of a soft paper never
  flattens inside the wedge's 3.0 log E span, so the plateau mean sits above
  the true Dmin. Supplying Dmin from an unexposed patch fixes this and is
  the clause 5.6.2 procedure anyway.
- With Dmin supplied, the estimate is within 1 R noiseless for every curve
  and within 5 % of trials outside tolerance from R87 up at 0.005 D error.
  The remaining failures at R50 (27 %) are noise against a ±1.5 R budget,
  which no estimator changes. Doubling the error to 0.01 D roughly doubles
  the spread.
- Scanner flare of 0.3 % of paper white, read in relative mode (density
  above paper white), biases R low: −0.6 (R50) to −9.6 (R167) with plateau
  Dmin, −0.5 to −5.7 with supplied Dmin. That is outside tolerance from
  about R107 up. An offset-correcting calibration removes this modelled
  bias; it does not by itself bring every case within tolerance.
- An uncalibrated wedge whose step densities are each off nominal by
  ±0.02 D adds 2.3 to 2.6 R of spread, comparable to the whole tolerance
  budget. Calibrated wedge values (T2115C certificate) matter for R, not
  only for the plotted axis.
- Table 2 misclassification is inherent near band edges (a true LER of
  1.37 sits 0.02 from the R130/R140 boundary), so the raw value must be
  shown alongside the class.

Scope of these tables (added 2026-09-16): the benchmark above judges
plateaus on the fitted curve and never refuses, whereas the application
judges them on the raw readings and refuses short plateaus. The noiseless,
flare and wedge-tolerance rows reproduce through the application within
0.1 R. The noisy rows do not include the application's refusals: with Dmin
supplied it refuses 0.2 to 0.8 % of trials at 0.005 D and 13 to 19 % at
0.01 D, with estimated Dmin 0.4 to 3.4 % and 25 to 33 %. ADR 0003 tabulates
the production estimator, including refusals, and supersedes the noisy rows
here.

## Decision

1. Add `densitometer/logic/sensitometry.py` with
   `iso_range(log_exposures, densities, dmin=None)` returning a small
   dataclass: Dmin and its source (supplied or estimated), Dmax, HT and HS
   as log exposures, LER to two decimals with the tie rule above, raw
   100 × LER, the table 2 class or none, plateau step counts, and warnings.
   Inside: sort by exposure; plateaus (flatness 0.02 D) judged on the
   measured densities, never on the monotone fit, which would pool a still
   rising end into a false plateau; Dmin from the supplied reading when
   given, else the light plateau mean labelled as estimated; Dmax from the
   dark plateau mean; PAVA monotone fit for the crossings only; crossings at
   Dmin + 0.04 and Dmin + 0.90 (Dmax − Dmin) by monotone cubic
   interpolation. NumPy only.
2. Refuse, rather than warn, when any step or the unexposed patch is
   clipped at scanner white or black or clamped by the calibration range, when the dark plateau has fewer than
   two steps, when Dmin is not supplied and the light plateau has fewer
   than two steps, or when a crossing is not bracketed. The analysis result
   gains a per-step flag for clipped and clamped readings so this check is
   exact. No light plateau is required when Dmin is supplied.
3. Report no uncertainty figure. The per-step density spread is a spatial
   pixel percentile, not a measurement uncertainty, and cannot feed a
   Monte Carlo. Revisit when repeated scans or a validated bootstrap exist.
4. UI: one summary line reading "effective R estimate (ISO 6846
   endpoints): R110, LER 1.07, Dmin 0.05 (estimated from plateau),
   Dmax 2.02". Two dashed density lines and two exposure markers on the
   plot. Refusal reasons, the flare caveat for relative mode, and a prompt
   to measure an unexposed patch when Dmin was estimated, in the status
   bar. No grade mapping: manufacturers publish product-specific R values
   per filter.
5. Add an unexposed-patch reading: a second, small selection measured with
   the current density mapping and passed as `dmin`. This follows clauses
   5.6.2 and 5.3.5, removes the light-plateau assumption, and is what makes
   soft papers measurable.
6. Add a reflection calibration path. `Calibration` already accepts any
   density list; add a way to enter the known densities of a scanned
   reflection gray scale (for example a Kodak Q-13) so prints get absolute
   densities. This removes the modelled flare bias.
7. Tests reuse the benchmark's curves: R within 2 units noiseless with Dmin
   supplied for R50 to R170, a reversed strip, a cut-off curve that must be
   refused, a clamped calibration that must be refused, and table 2
   classification at band edges including the 0.345 tie.

## Consequences

- An effective R estimate becomes available for any print whose exposure
  brackets Dmax and, unless an unexposed patch is measured, paper white.
  The app refuses when it does not.
- Without a reflection calibration the estimate reads low because of
  scanner flare; the UI says so. With it, the modelled flare bias is gone,
  but residual noise and wedge tolerance still put a share of readings
  outside the standard's tolerance, most at R50 where the budget is ±1.5.
- Soft papers depend on the unexposed-patch reading; the estimated-Dmin
  label and prompt make that visible.
- Calibrated wedge values are worth entering: uncalibrated step tolerance
  alone costs about 2.5 R of spread.
- The custom monotone cubic is kept on benchmark evidence and must stay
  covered by the tests that reproduce those numbers.
- Film curves (contrast index, gamma) are a natural follow-on but are not
  covered here.

## Sources

- ISO 6846:1992, Photography, Black-and-white continuous-tone papers,
  Determination of ISO speed and ISO range for printing, second edition.
  Sample pages read from
  https://cdn.standards.iteh.ai/samples/13355/5b11c8fb17f843aab03de56691b4b353/ISO-6846-1992.pdf
  and record at https://www.iso.org/standard/13355.html
- `0001-iso-r-benchmark.py` in this directory (run with the project's
  Python; prints the tables above).
- Ilford Multigrade FB Warmtone datasheet, example of product-specific
  ISO(R) per filter.
  https://www.ilfordphoto.com/amfile/file/download/file/1881/product/738/
