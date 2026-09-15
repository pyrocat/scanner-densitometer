from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..logic.analysis import (
    analyze_selection,
    calibrate_from_wedge,
    flag_signals,
    measure_patch,
    signal_to_density,
)
from ..logic.image_loader import load_image
from ..logic.models import (
    AnalysisResult,
    AnalysisSettings,
    Calibration,
    LoadedImage,
    ReferenceMode,
    Selection,
    T2115_DENSITIES,
    validate_wedge_densities,
)
from ..logic.sensitometry import DarkPatch, IsoRangeEstimate, iso_range_from_analysis
from .image_canvas import ImageCanvas

REFERENCE_LABELS: dict[str, ReferenceMode] = {
    "Relative to brightest step": "relative",
    "Calibrated from T2115 scan": "calibrated",
}


class MainWindow(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=12)
        self._loaded_image: LoadedImage | None = None
        self._selection: Selection | None = None
        self._analysis_result: AnalysisResult | None = None
        self._calibration: Calibration | None = None
        # Median signal of an unexposed, processed patch (ISO 6846 clause 5.6.2).
        self._patch_signal: float | None = None
        # Overexposed patches as (log exposure offset beyond wedge step 1,
        # median signal): the clause 5.6.3 evidence that density stopped rising.
        self._dark_patches: list[tuple[float, float]] = []
        # Density of each step of the wedge the sample was exposed through;
        # replace the nominal values with certificate values when known.
        self._wedge_densities: tuple[float, ...] = T2115_DENSITIES

        self.reference_label = tk.StringVar(value=next(iter(REFERENCE_LABELS)))
        self.status_text = tk.StringVar(
            value="Open a linear 16-bit TIFF, then manually drag a rectangle over the step wedge."
        )
        self.file_text = tk.StringVar(value="No image loaded.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._draw_placeholder_plot()

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.columnconfigure(10, weight=1)

        ttk.Button(toolbar, text="Open TIFF...", command=self._open_file).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Label(toolbar, text="Density reference:").grid(row=0, column=1, padx=(0, 8))
        self.reference_box = ttk.Combobox(
            toolbar,
            state="readonly",
            textvariable=self.reference_label,
            values=list(REFERENCE_LABELS.keys()),
            width=30,
        )
        self.reference_box.grid(row=0, column=2, padx=(0, 8))
        self.reference_box.bind("<<ComboboxSelected>>", self._reanalyze_current_selection)

        self.analyze_button = ttk.Button(
            toolbar,
            text="Analyze Selection",
            command=self._run_analysis,
            state="disabled",
        )
        self.analyze_button.grid(row=0, column=3, padx=(0, 8))
        self.calibrate_button = ttk.Button(
            toolbar,
            text="Calibrate from Selection",
            command=self._calibrate,
            state="disabled",
        )
        self.calibrate_button.grid(row=0, column=4, padx=(0, 8))
        self.patch_button = ttk.Button(
            toolbar,
            text="Unexposed Patch from Selection",
            command=self._measure_patch,
            state="disabled",
        )
        self.patch_button.grid(row=0, column=5, padx=(0, 8))
        self.dark_patch_button = ttk.Button(
            toolbar,
            text="Overexposed Patch from Selection",
            command=self._measure_dark_patch,
            state="disabled",
        )
        self.dark_patch_button.grid(row=0, column=6, padx=(0, 8))
        self.clear_patches_button = ttk.Button(
            toolbar,
            text="Clear Patches",
            command=self._clear_patches,
            state="disabled",
        )
        self.clear_patches_button.grid(row=0, column=7, padx=(0, 8))
        ttk.Button(toolbar, text="Wedge Densities...", command=self._edit_wedge_densities).grid(
            row=0, column=8, padx=(0, 8)
        )
        ttk.Label(
            toolbar,
            text="Drag along the wedge end to end; handles rotate, stretch, widen, move.",
        ).grid(row=0, column=9, padx=(12, 8))

        ttk.Label(toolbar, textvariable=self.file_text).grid(row=0, column=10, sticky="e")

    def _build_body(self) -> None:
        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew")

        image_panel = ttk.Frame(body, padding=0)
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(0, weight=1)
        self.image_canvas = ImageCanvas(image_panel, self._handle_selection)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")

        graph_panel = ttk.Frame(body, padding=(10, 0, 0, 0))
        graph_panel.columnconfigure(0, weight=1)
        graph_panel.rowconfigure(0, weight=3)
        graph_panel.rowconfigure(2, weight=2)

        self.figure = Figure(figsize=(6.2, 5.6), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=graph_panel)
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.summary_label = ttk.Label(
            graph_panel,
            text="No analysis yet.",
            anchor="w",
            justify="left",
        )
        self.summary_label.grid(row=1, column=0, sticky="ew", pady=(8, 8))

        columns = ("step", "strip", "loge", "wedge", "density", "spread", "signal")
        self.results_table = ttk.Treeview(
            graph_panel,
            columns=columns,
            show="headings",
            height=14,
        )
        headings = {
            "step": ("Step", 55),
            "strip": ("Strip Pos", 75),
            "loge": ("Rel Log E", 90),
            "wedge": ("Wedge OD", 85),
            "density": ("Density", 85),
            "spread": ("±D", 65),
            "signal": ("Signal", 95),
        }
        for column, (label, width) in headings.items():
            self.results_table.heading(column, text=label)
            self.results_table.column(column, width=width, anchor="center", stretch=True)

        table_scrollbar = ttk.Scrollbar(
            graph_panel,
            orient="vertical",
            command=self.results_table.yview,
        )
        self.results_table.configure(yscrollcommand=table_scrollbar.set)

        table_frame = ttk.Frame(graph_panel)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.results_table.grid(in_=table_frame, row=0, column=0, sticky="nsew")
        table_scrollbar.grid(in_=table_frame, row=0, column=1, sticky="ns")

        body.add(image_panel, weight=3)
        body.add(graph_panel, weight=2)

    def _build_statusbar(self) -> None:
        status_frame = ttk.Frame(self)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_text, anchor="w").grid(
            row=0,
            column=0,
            sticky="ew",
        )

    def _open_file(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Open 16-bit TIFF",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not selected_path:
            return

        try:
            loaded_image = load_image(selected_path)
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Unable to open image", str(error))
            self.status_text.set(f"Failed to open {Path(selected_path).name}.")
            return

        self._loaded_image = loaded_image
        self._selection = None
        self._analysis_result = None
        self._patch_signal = None
        self._dark_patches = []
        self.file_text.set(
            f"{loaded_image.path.name}  |  {loaded_image.width}x{loaded_image.height}  |  {loaded_image.mode}"
        )
        if loaded_image.bits_per_sample <= 8:
            self.status_text.set(
                f"{loaded_image.bits_per_sample}-bit scan loaded: it is probably gamma-encoded, so relative "
                "densities will be compressed. Calibrate from a T2115 scan or rescan as linear 16-bit."
            )
        else:
            self.status_text.set("Drag from one end of the step wedge to the other, then adjust the handles.")
        self.image_canvas.set_image(loaded_image)
        self._clear_results()
        self._update_action_state()

    def _handle_selection(self, selection: Selection) -> None:
        self._selection = selection
        self.status_text.set("Selection captured. Analyzing the strip.")
        self._update_action_state()
        self._run_analysis()

    def _reanalyze_current_selection(self, *_: object) -> None:
        if self._loaded_image is not None and self._selection is not None:
            self._run_analysis()

    def _ask_densities(self, title: str, prompt: str, initial: tuple[float, ...]) -> tuple[float, ...] | None:
        answer = simpledialog.askstring(
            title, prompt, initialvalue=", ".join(f"{d:.2f}" for d in initial), parent=self
        )
        if not answer:
            return None
        return validate_wedge_densities(answer.replace(";", ",").split(","))

    def _edit_wedge_densities(self) -> None:
        try:
            densities = self._ask_densities(
                "Wedge step densities",
                "Density of each step of the wedge the sample was exposed through, step 1 first:",
                self._wedge_densities,
            )
        except ValueError as error:
            messagebox.showerror("Invalid wedge densities", str(error))
            return
        if densities is None:
            return
        self._wedge_densities = densities
        self.status_text.set(f"Using {len(densities)} wedge step densities for the exposure axis.")
        self._reanalyze_current_selection()

    def _measure_patch(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return
        try:
            self._patch_signal = measure_patch(self._loaded_image.analysis_image, self._selection)
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Patch measurement failed", str(error))
            return
        self.status_text.set(
            f"Unexposed patch signal {self._patch_signal:.0f} stored as Dmin. Now select the wedge strip."
        )

    def _measure_dark_patch(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return
        try:
            signal = measure_patch(self._loaded_image.analysis_image, self._selection)
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Patch measurement failed", str(error))
            return
        # The offset is relative to wedge step 1, the most exposed strip step,
        # so an unmasked patch at the strip's exposure time sits step 1's
        # density beyond it; the user records longer exposures as log10(k).
        step_one = self._wedge_densities[0]
        offset = simpledialog.askfloat(
            "Overexposed patch",
            "Log exposure of this patch beyond wedge step 1 (the most exposed strip step).\n"
            f"An unmasked patch given the strip's exposure time is +{step_one:.2f}; "
            "k times the exposure time adds log10(k).",
            initialvalue=round(step_one + 1.0, 2),
            parent=self,
        )
        if offset is None:
            return
        if offset <= 0.0:
            messagebox.showerror("Invalid exposure offset", "The offset must be greater than zero.")
            return
        self._dark_patches.append((float(offset), signal))
        self._dark_patches.sort()
        self.status_text.set(
            f"Overexposed patch at +{offset:.2f} log E stored (signal {signal:.0f}); "
            f"{len(self._dark_patches)} patch reading(s) will support Dmax. Now select the wedge strip."
        )

    def _clear_patches(self) -> None:
        self._patch_signal = None
        self._dark_patches = []
        self.status_text.set("Unexposed and overexposed patch readings cleared.")
        self._reanalyze_current_selection()

    def _calibrate(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return
        # Any scanned target of known densities works: the T2115 (nominal or
        # certificate values) for transmission, a reflection gray scale such
        # as a Kodak Q-13 for prints. The cell count follows the list length.
        try:
            densities = self._ask_densities(
                "Target step densities",
                "Known density of each step of the scanned target, lightest first:",
                self._wedge_densities,
            )
            if densities is None:
                return
            self._calibration = calibrate_from_wedge(
                self._loaded_image.analysis_image, self._selection, densities
            )
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Calibration failed", str(error))
            return
        calibration = self._calibration
        self.status_text.set(
            f"Calibrated from {calibration.usable_steps} of {len(calibration.densities)} wedge steps, "
            f"up to D {calibration.max_density:.2f}. Keep the scanner exposure locked. "
            "Now select the sample strip."
        )
        self.reference_label.set(next(label for label, mode in REFERENCE_LABELS.items() if mode == "calibrated"))

    def _patch_density(self, signal: float, settings: AnalysisSettings, result: AnalysisResult, label: str) -> float:
        """Convert a patch signal with the strip's mapping, refusing unusable readings."""
        clipped, clamped = flag_signals([signal], settings)
        if clipped[0] or clamped[0]:
            raise ValueError(f"ISO(R) refused: {label} is clipped or outside the calibration range.")
        return float(signal_to_density([signal], settings, result.reference_value)[0])

    def _run_analysis(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return

        settings = AnalysisSettings(
            wedge_densities=self._wedge_densities,
            reference_mode=REFERENCE_LABELS[self.reference_label.get()],
            calibration=self._calibration,
        )
        try:
            result = analyze_selection(
                self._loaded_image.analysis_image,
                self._selection,
                settings,
            )
        except ValueError as error:
            # Expected for a selection that is not a wedge strip, such as a
            # patch about to be stored; keep it in the status bar, not modal.
            self.status_text.set(f"{error} Short selections can still be stored as a patch reading.")
            self._analysis_result = None
            return
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Analysis failed", str(error))
            self.status_text.set(str(error))
            self._analysis_result = None
            return

        self._analysis_result = result
        notes = list(result.warnings)
        estimate: IsoRangeEstimate | None = None
        patches: list[DarkPatch] = []
        try:
            dmin = None
            if self._patch_signal is not None:
                dmin = self._patch_density(self._patch_signal, settings, result, "the unexposed patch")
            for offset, signal in self._dark_patches:
                density = self._patch_density(signal, settings, result, f"the overexposed patch at +{offset:.2f} log E")
                patches.append(DarkPatch(offset, density))
            estimate = iso_range_from_analysis(result, dmin, patches)
            notes.extend(estimate.warnings)
            if result.reference_mode == "relative":
                notes.append("Relative mode: scanner flare biases R low; calibrate from a reflection gray scale.")
        except ValueError as error:
            notes.append(str(error))
        self._draw_result_plot(result, estimate, patches)
        self._populate_table(result)
        self._update_summary(result, estimate)
        self.status_text.set(" ".join(notes) or "Analysis complete. Check that the cell lines sit on the step edges.")
        self._update_action_state()

    def _draw_placeholder_plot(self) -> None:
        self.axis.clear()
        self.axis.set_title("Density Curve")
        self.axis.set_xlabel("Relative log exposure")
        self.axis.set_ylabel("Density")
        self.axis.text(
            0.5,
            0.5,
            "No analysis yet",
            ha="center",
            va="center",
            transform=self.axis.transAxes,
        )
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _draw_result_plot(
        self,
        result: AnalysisResult,
        estimate: IsoRangeEstimate | None = None,
        patches: list[DarkPatch] | None = None,
    ) -> None:
        self.axis.clear()
        if estimate is not None:
            for density in (estimate.density_ht, estimate.density_hs):
                self.axis.axhline(density, linestyle="--", linewidth=0.9, color="#888888")
            for log_exposure, label in ((estimate.log_ht, "HT"), (estimate.log_hs, "HS")):
                self.axis.axvline(log_exposure, linestyle=":", linewidth=0.9, color="#c0392b")
                self.axis.annotate(label, (log_exposure, 0.0), xytext=(2, 2), textcoords="offset points", color="#c0392b")
        self.axis.errorbar(
            result.x_values,
            result.y_values,
            yerr=result.y_spreads,
            marker="o",
            linewidth=1.8,
            capsize=2,
            color="#1d6fa5",
        )
        x_max = float(result.x_values.max())
        if patches:
            self.axis.plot(
                [x_max + patch.log_exposure_offset for patch in patches],
                [patch.density for patch in patches],
                linestyle="none",
                marker="s",
                markerfacecolor="none",
                color="#1d6fa5",
                label="overexposed patch",
            )
            self.axis.legend(loc="lower right")
        self.axis.set_title("Stouffer T2115 Print Density Curve")
        self.axis.set_xlabel("Relative log exposure")
        ylabel = "Density above brightest step" if result.reference_mode == "relative" else "Density (D)"
        self.axis.set_ylabel(ylabel)
        self.axis.grid(True, alpha=0.25)
        right = max([x_max] + [x_max + patch.log_exposure_offset for patch in patches or []])
        self.axis.set_xlim(float(result.x_values.min()), right)
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _populate_table(self, result: AnalysisResult) -> None:
        self.results_table.delete(*self.results_table.get_children())
        for measurement in result.measurements:
            self.results_table.insert(
                "",
                "end",
                values=(
                    measurement.step,
                    measurement.spatial_index,
                    f"{measurement.relative_log_exposure:.2f}",
                    f"{measurement.wedge_density:.2f}",
                    f"{measurement.density:.3f}",
                    f"{measurement.density_spread:.3f}",
                    f"{measurement.signal:.0f}",
                ),
            )

    def _update_summary(self, result: AnalysisResult, estimate: IsoRangeEstimate | None = None) -> None:
        if result.reference_mode == "relative":
            reference_text = f"density above brightest step (signal {result.reference_value:.0f})"
        else:
            reference_text = f"calibrated, valid up to D {self._calibration.max_density:.2f}"
        lines = [f"{len(result.measurements)} equal cells along the strip. Reference: {reference_text}."]
        if estimate is None:
            lines.append("Effective R estimate (ISO 6846 endpoints): refused, see status bar.")
        else:
            iso_text = f"R{estimate.iso_range}" if estimate.iso_range is not None else "outside table 2"
            dmin_text = "from unexposed patch" if estimate.dmin_source == "supplied" else "estimated from plateau"
            if estimate.dmax_source == "established":
                dmax_text = f"established over {estimate.dmax_span:.2f} log E"
            else:
                dmax_text = f"provisional lower bound, {estimate.dmax_span:.2f} log E"
            if estimate.dark_patches_used:
                dmax_text += f" with {estimate.dark_patches_used} patch(es)"
            fit_text = f", fit adjusted up to {estimate.fit_adjustment:.3f} D" if estimate.adjusted_steps else ""
            lines.append(
                f"Effective R estimate (ISO 6846 endpoints): {iso_text}, LER {estimate.ler:.2f} "
                f"(raw {estimate.raw_r:.1f}), Dmin {round(estimate.dmin, 2) + 0.0:.2f} ({dmin_text}), "
                f"Dmax {estimate.dmax:.2f} ({dmax_text}){fit_text}."
            )
        self.summary_label.config(text="\n".join(lines))

    def _clear_results(self) -> None:
        self.results_table.delete(*self.results_table.get_children())
        self.summary_label.config(text="No analysis yet.")
        self._draw_placeholder_plot()

    def _update_action_state(self) -> None:
        state = "normal" if self._loaded_image is not None and self._selection is not None else "disabled"
        self.analyze_button.config(state=state)
        self.calibrate_button.config(state=state)
        self.patch_button.config(state=state)
        self.dark_patch_button.config(state=state)
        self.clear_patches_button.config(state="normal" if self._loaded_image is not None else "disabled")
