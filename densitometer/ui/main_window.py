from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..logic.analysis import analyze_selection, calibrate_from_wedge
from ..logic.image_loader import load_image
from ..logic.models import (
    AnalysisResult,
    AnalysisSettings,
    Calibration,
    LoadedImage,
    ReferenceMode,
    Selection,
)
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
        toolbar.columnconfigure(6, weight=1)

        ttk.Button(toolbar, text="Open TIFF...", command=self._open_file).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Label(toolbar, text="Density reference:").grid(row=0, column=1, padx=(0, 8))
        self.calibrate_button = ttk.Button(
            toolbar,
            text="Calibrate from Selection",
            command=self._calibrate,
            state="disabled",
        )
        self.calibrate_button.grid(row=0, column=4, padx=(0, 8))
        ttk.Label(
            toolbar,
            text="Draw a rectangle edge to edge over the 21 steps.",
        ).grid(row=0, column=5, padx=(12, 8))

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

        ttk.Label(toolbar, textvariable=self.file_text).grid(row=0, column=6, sticky="e")

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
        self.file_text.set(
            f"{loaded_image.path.name}  |  {loaded_image.width}x{loaded_image.height}  |  {loaded_image.mode}"
        )
        if loaded_image.source_dtype == "uint8":
            self.status_text.set(
                "8-bit scan loaded: it is probably gamma-encoded, so relative densities will be "
                "compressed. Calibrate from a T2115 scan or rescan as linear 16-bit."
            )
        else:
            self.status_text.set("Draw a rectangle edge to edge over the step wedge to analyze it.")
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

    def _calibrate(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return
        try:
            self._calibration = calibrate_from_wedge(self._loaded_image.analysis_image, self._selection)
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

    def _run_analysis(self) -> None:
        if self._loaded_image is None or self._selection is None:
            return

        settings = AnalysisSettings(
            reference_mode=REFERENCE_LABELS[self.reference_label.get()],
            calibration=self._calibration,
        )
        try:
            result = analyze_selection(
                self._loaded_image.analysis_image,
                self._selection,
                settings,
            )
        except Exception as error:  # pragma: no cover - Tk error dialog
            messagebox.showerror("Analysis failed", str(error))
            self.status_text.set(str(error))
            self._analysis_result = None
            self.image_canvas.set_analysis_result(None)
            return

        self._analysis_result = result
        self._selection = result.selection
        self.image_canvas.set_analysis_result(result)
        self._draw_result_plot(result)
        self._populate_table(result)
        self._update_summary(result)
        self.status_text.set(
            " ".join(result.warnings) or "Analysis complete. Check that the cell lines sit on the step edges."
        )
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

    def _draw_result_plot(self, result: AnalysisResult) -> None:
        self.axis.clear()
        self.axis.errorbar(
            result.x_values,
            result.y_values,
            yerr=result.y_spreads,
            marker="o",
            linewidth=1.8,
            capsize=2,
            color="#1d6fa5",
        )
        self.axis.set_title("Stouffer T2115 Print Density Curve")
        self.axis.set_xlabel("Relative log exposure")
        ylabel = "Density above brightest step" if result.reference_mode == "relative" else "Density (D)"
        self.axis.set_ylabel(ylabel)
        self.axis.grid(True, alpha=0.25)
        self.axis.set_xlim(float(result.x_values.min()), float(result.x_values.max()))
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

    def _update_summary(self, result: AnalysisResult) -> None:
        if result.reference_mode == "relative":
            reference_text = f"density above brightest step (signal {result.reference_value:.0f})"
        else:
            reference_text = f"calibrated, valid up to D {self._calibration.max_density:.2f}"
        self.summary_label.config(
            text=(
                f"{len(result.measurements)} equal cells in a {result.orientation} strip. "
                f"Reference: {reference_text}."
            )
        )

    def _clear_results(self) -> None:
        self.results_table.delete(*self.results_table.get_children())
        self.summary_label.config(text="No analysis yet.")
        self._draw_placeholder_plot()

    def _update_action_state(self) -> None:
        state = "normal" if self._loaded_image is not None and self._selection is not None else "disabled"
        self.analyze_button.config(state=state)
        self.calibrate_button.config(state=state)
