# ADR 0002: Transmission scanning for calibration and film, not for prints

- Status: proposed
- Date: 2026-09-03

## Context

A flatbed scanner with a transparency unit can scan a print as if it were a
transparency. The proposal: use clear glass as absolute white, an opaque
patch as absolute black, and the Stouffer T2115 as known intermediate
densities, then read wedge-exposed prints in the same transmission pass to
get better step separation than reflection scanning offers.

Reflection scanning of prints is flare limited near paper Dmax, which is
where the 90 % net-density endpoint of ISO 6846 sits (see ADR 0001). So the
motivation is real. The question is whether a transmission reading of a
print measures the quantity ISO(R) is defined on.

### Investigation

`0002-transmission-model.py` beside this file models a print with the
Williams and Clapper reflection model: a silver layer of single-pass
transmission density D_T over a diffuse white base, with surface
reflectance for the print finish and internal reflectance at the gelatin
surface. A transmission reading of the print sees the base plus D_T; the
base is a uniform additive diffuser to first order and cancels in
Dmin-relative quantities. The ISO 6846 endpoints (0.04 above Dmin and 90 %
of net density) were evaluated on the reflection curve and on the
transmission curve of the same silver image.

| Finish, emulsion Dmax_T | R from reflection | R from transmission |
|---|---|---|
| glossy, 1.2 | 81 | 81 |
| glossy, 1.6 | 116 | 133 |
| semi-matte, 1.6 | 102 | 133 |
| matte, 1.6 | 93 | 133 |
| glossy, 2.0 | 107 | 137 |
| matte, 2.0 | 88 | 137 |

Findings:

- Transmission ignores the surface. Glossy and matte prints of the same
  silver image read identically in transmission while their reflection R
  differs by up to 40 units. The reflection shoulder is caused by surface
  reflection, which does not exist in the transmission path.
- The transmission endpoints land in the wrong places on the reflection
  curve: the 0.04 point corresponds to about 0.15 above Dmin, and the 90 %
  point to 97 to 100 % of the net density range.
- Near Dmin the reflection density rises about four times faster than the
  transmission density (light crosses the silver at least twice and bounces
  between base and surface). Resolving the 0.04 reflection point therefore
  needs 0.01 D precision in transmission.
- Paper base transmission is not uniform. Fibre and baryta texture shows
  through as cloudiness at the level of the 0.04 threshold.
- Converting transmission readings to reflection density needs the paper's
  surface and internal reflectance, which can only be measured in
  reflection. At that point a reflection scan is the direct route.
- Dynamic range would have been adequate: conventional paper has a
  transmission density near 0.9 to 1.0, so a black patch totals about 2.2
  to 3.0 D, inside a consumer transparency unit's range.

What the transmission path does provide is a complete absolute calibration
of the scanner: glass is a true zero, the opaque patch gives the flare and
black floor directly, and the wedge gives 21 known points between them. It
is also the correct geometry for wedge-exposed film, which is a
transmission object.

## Decision

1. Do not read prints in transmission for density or ISO(R). The app
   documents transmission mode as unsuitable for print densitometry and
   keeps prints on the reflection path.
2. Extend `calibrate_from_wedge` with optional clear-glass and opaque
   readings: the glass reading adds the (signal, 0.0) point above the
   wedge's 0.05 D step, and the opaque reading defines the floor below the
   wedge's 3.05 D step so out-of-range samples are refused rather than
   clamped. This is the black reference ADR 0001 needs.
3. Keep the reflection fix in the reflection path, as planned in ADR 0001:
   a black reference patch for the flare floor and a reflection gray scale
   with known densities reaching about 2.0 as the calibration target.
4. Treat wedge-exposed film as the first-class use of transmission
   calibration. Film curve measures (contrast index, gamma) remain a
   follow-on.

## Consequences

- Users with a transparency unit get an absolute, offset-corrected
  calibration for film and for the wedge itself, and a clear statement of
  why prints must still be scanned in reflection.
- Print densitometry stays flare limited until the reflection calibration
  of ADR 0001 is implemented; transmission scanning does not shortcut it.
- The model is first order: it omits oblique-path factors and treats the
  base as a uniform diffuser. Its purpose is to show that the discrepancy
  is large and finish dependent, not to predict R for a given paper.

## Sources

- Williams, F. C. and Clapper, F. R., Multiple internal reflections in
  photographic color prints, Journal of the Optical Society of America 43,
  1953.
- `0002-transmission-model.py` in this directory (prints the table above).
- Photographic paper support patent giving the transmission density of
  conventional paper.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5082724
- ADR 0001 in this directory for the ISO 6846 endpoints and tolerance.
