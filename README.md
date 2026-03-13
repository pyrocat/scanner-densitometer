# Scanner Densitometer

Tkinter app for using a flatbed scanner as a practical densitometer surrogate when analyzing prints made with a Stouffer T2115 21-step wedge.

The app does not talk to the scanner directly. It assumes the scan has already been made and saved as a 16-bit TIFF. The workflow is:

1. Open a TIFF scan.
2. Manually draw a rectangle over the step wedge on the scan.
3. Let the app detect the 21 steps and build a density curve.

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

- Automatic 21-step boundary detection from one selection.
- Immediate overlay of detected step boundaries on the scan.
- Built-in curve plotting for relative log exposure versus measured density.
- A relative mode and a full-scale mode for density reference handling.
- Testable logic separated from the Tkinter UI.

## Assumptions and limitations

- Best results require a linear 16-bit TIFF with scanner corrections disabled.
- The app supports grayscale TIFFs directly and also accepts RGB TIFFs by converting them to luminance.
- The manually selected ROI should contain one complete 21-step strip with little extra background.
- The plotted x-axis is relative log exposure derived from the T2115 spacing, not enlarger time in seconds.
- Without scanner calibration against a reflective standard, absolute density should be treated as approximate.

## Run

```bash
python3 -m densitometer
```

## Test

```bash
python3 -m unittest discover -s tests -v
```
