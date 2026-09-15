# Scanner Densitometer

Tkinter app for using a flatbed scanner as a practical densitometer surrogate when analyzing prints made with a Stouffer T2115 21-step wedge.

The app does not talk to the scanner directly. It assumes the scan has already been made and saved as a 16-bit TIFF. TIFFs are decoded from their own tags (bit depth, photometric interpretation, channel layout) so 16-bit RGB scans keep their precision. The workflow is:

1. Open a TIFF scan.
2. Optionally drag a strip over a scan of the T2115 itself and click
   **Calibrate from Selection**. This maps scanner signal to real density
   using the wedge's known 0.05–3.05 D steps.
3. Drag from one end of the wedge image on the sample (negative or print) to
   the other. The strip can sit at any angle: drag an end handle to rotate or
   stretch it, a side handle to set the width, or its inside to move it. The
   app divides the strip into 21 equal cells, because the T2115 geometry is
   fixed, measures the centre of each cell, and plots the curve.
4. Optionally select an unexposed, processed patch and click **Unexposed
   Patch from Selection** (Dmin), and one or more overexposed patches with
   **Overexposed Patch from Selection**, entering each patch's log exposure
   beyond wedge step 1. The patches establish Dmax; without them the
   wedge-only Dmax is reported as a provisional lower bound.

## Why this approach

The reference material points to the same core pattern:

- Use a scanner only as a capture device, not as the densitometer UI.
- Keep the scan as linear and unprocessed as possible.
- Measure a rectangular region of interest instead of single pixels.
- Convert measured scanner signal into density with a log relationship.

For this project, that becomes a simpler desktop workflow:

- acquisition stays in scanner software such as VueScan or SilverFast
- analysis happens inside this app from a saved TIFF
- UI and math are split so the analysis can be tested independently

## Research summary

### References the implementation is based on

- Photrio thread on using the Stouffer 21-step wedge and scanner-based calculations:
  https://www.photrio.com/forum/threads/nerd-alert-got-a-stouffer-21-step-wedge-and-want-to-share-some-interesting-things-ive-been-learning.193908/
- Archived NegFix `scan_dens` article, referenced repeatedly in scanner-densitometry discussions:
  https://web.archive.org/web/20120428233443/https://sites.google.com/site/negfix/scan_dens
- Stouffer T2115 specifications:
  https://www.stouffer.net/T2115spec.htm
- Stouffer guidance on the 21-step wedge and its 1/2-stop spacing:
  https://www.stouffer.net/using21step.htm

### Existing mature solutions and what they contribute

1. NegFix `scan_dens`
   Summary: older but directly relevant scanner-as-densitometer method. It is the clearest conceptual ancestor for this app.
   Adaptation: keep the linear scan assumption and density-from-signal workflow, but replace the manual procedure with ROI selection and automatic 21-step segmentation.

2. VueScan raw TIFF workflow
   Summary: mature scanner front-end with support for 16-bit linear raw TIFF output and reprocessing from file.
   Adaptation: use it only for acquisition. This app starts after the TIFF already exists.

3. SilverFast densitometer / HDR workflow
   Summary: mature commercial scan software with point densitometer tools and 16-bit workflows.
   Adaptation: useful for making clean scans, but it is not specialized for plotting a full 21-step print curve from one selected strip.

4. ImageJ optical-density calibration workflow
   Summary: mature image-analysis tool with ROI measurement, calibration curves, and profile plotting.
   Adaptation: the ROI-first workflow and curve presentation are directly applicable, but this app removes the manual multi-step setup and is purpose-built for the Stouffer 21-step strip.

5. Dedicated hardware densitometers
   Summary: still the reference standard for absolute density accuracy.
   Adaptation: use hardware readings to validate scanner behavior later, but not as a dependency for this first version.

### Improvements over the manual workflows

- Fixed 21-cell grid from one selection, overlaid on the scan so misalignment is visible.
- Steps identified by position, with the strip direction detected automatically.
- Built-in curve plotting for relative log exposure versus measured density, with per-step noise bars.
- A relative mode (density above the brightest step) and a calibrated mode built from a scan of any target with known step densities (T2115 nominal or certificate values, or a reflection gray scale such as a Kodak Q-13).
- Warnings for clipped steps and for samples outside the calibration range; selections that leave the image are refused rather than padded.
- An effective ISO(R) estimate using the ISO 6846 endpoints (HT at Dmin + 0.04, HS at Dmin + 0.90 of the net Dmax), with the endpoints drawn on the plot. Dmin comes from an unexposed patch when one is measured ("Unexposed Patch from Selection"), otherwise from the light plateau. Dmax comes from the dark plateau, extended by overexposed patches, and is labelled established or provisional. See `.docs/ADR/0001-iso-r-from-step-wedge-curve.md` and `.docs/ADR/0003-iso-r-validity-checks.md`.
- Testable logic separated from the Tkinter UI.

## Assumptions and limitations

- Best results require a linear 16-bit TIFF with scanner corrections disabled. Calibrated mode also works with gamma-encoded scans, because the calibration absorbs any monotonic encoding.
- The app supports unsigned 8- and 16-bit grayscale and RGB TIFFs (RGB is converted to luminance of the linear samples; WhiteIsZero files are inverted). Palette, CMYK, YCbCr, floating-point and signed TIFFs are refused. Other formats are decoded by Pillow at 8 bits per channel, or 16 bits for grayscale.
- The strip's centreline must span the 21 steps end to end; the grid is not detected, only divided. Check the overlaid cell lines and the ±D column.
- A calibration is only valid for scans made with the same scanner, light path (transmission or reflection), and locked exposure. Scanning the wedge alongside the sample is the safest workflow.
- Calibrating with the T2115 covers transmission scans. Reflection prints get relative densities unless a reflection step tablet is used as the calibration target.
- The ISO(R) figure is an estimate, not an ISO range: the scanner meets none of the standard's densitometry conditions. It is refused when any step is clipped or clamped, when the curve does not reach Dmax or an overexposed patch shows it still rising, when Dmin is neither measured nor visible as a plateau, when a supplied Dmin contradicts the strip's light end, when the net density is below 0.40 D, or when the monotone fit has to change a step by more than 0.05 D. A wedge-only Dmax is a provisional lower bound until a plateau spans a decade of exposure. In relative mode scanner flare biases it low; soft papers need the unexposed-patch reading.
- Relative mode assumes a linear scanner with no black offset, so it compresses high densities; treat it as approximate.
- The plotted x-axis is relative log exposure derived from the T2115 spacing, not enlarger time in seconds.

## Run

Requires Python 3.12 or later; `poetry install` creates the environment.

```bash
poetry run densitometer
```

## Test

```bash
poetry run python -m unittest discover -s tests -v
```
