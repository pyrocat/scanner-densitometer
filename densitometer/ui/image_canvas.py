from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

from ..logic.models import AnalysisResult, LoadedImage, Selection


class ImageCanvas(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_selection_committed: Callable[[Selection], None],
    ) -> None:
        super().__init__(master)
        self._on_selection_committed = on_selection_committed

        self.canvas = tk.Canvas(self, background="#161616", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._loaded_image: LoadedImage | None = None
        self._selection: Selection | None = None
        self._analysis_result: AnalysisResult | None = None
        self._anchor_selection: Selection | None = None
        self._photo_image: ImageTk.PhotoImage | None = None
        self._cached_preview_size: tuple[int, int] | None = None
        self._cached_resized_image: Image.Image | None = None
        self._display_rect = (0.0, 0.0, 1.0, 1.0)

        self.canvas.bind("<Configure>", self._render)
        self.canvas.bind("<ButtonPress-1>", self._start_selection)
        self.canvas.bind("<B1-Motion>", self._update_selection)
        self.canvas.bind("<ButtonRelease-1>", self._finish_selection)

    def set_image(self, loaded_image: LoadedImage | None) -> None:
        self._loaded_image = loaded_image
        self._selection = None
        self._analysis_result = None
        self._anchor_selection = None
        self._cached_preview_size = None
        self._cached_resized_image = None
        self._render()

    def set_selection(self, selection: Selection | None) -> None:
        self._selection = selection
        self._render()

    def set_analysis_result(self, result: AnalysisResult | None) -> None:
        self._analysis_result = result
        self._selection = result.selection if result else self._selection
        self._render()

    def _render(self, *_: object) -> None:
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 2)
        canvas_height = max(self.canvas.winfo_height(), 2)

        if self._loaded_image is None:
            self._display_rect = (0.0, 0.0, 1.0, 1.0)
            self.canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text="Open a 16-bit TIFF, then manually drag a rectangle over the step wedge.",
                fill="#cfd3d6",
                font=("TkDefaultFont", 12),
                width=max(canvas_width - 80, 320),
            )
            return

        preview = self._loaded_image.preview_image
        scale = min(canvas_width / preview.width, canvas_height / preview.height)
        draw_width = max(1, int(round(preview.width * scale)))
        draw_height = max(1, int(round(preview.height * scale)))
        offset_x = (canvas_width - draw_width) / 2
        offset_y = (canvas_height - draw_height) / 2
        self._display_rect = (offset_x, offset_y, draw_width, draw_height)

        rendered_preview = self._get_resized_preview(draw_width, draw_height)
        self._photo_image = ImageTk.PhotoImage(rendered_preview)
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo_image)

        if self._selection is not None:
            self._draw_selection(self._selection)

        if self._analysis_result is not None:
            self._draw_boundaries(self._analysis_result)

    def _get_resized_preview(self, width: int, height: int) -> Image.Image:
        if self._loaded_image is None:
            raise RuntimeError("No image loaded.")

        target_size = (width, height)
        if self._cached_preview_size != target_size or self._cached_resized_image is None:
            self._cached_preview_size = target_size
            self._cached_resized_image = self._loaded_image.preview_image.resize(
                target_size,
                Image.Resampling.BILINEAR,
            )
        return self._cached_resized_image

    def _start_selection(self, event: tk.Event[tk.Misc]) -> None:
        if self._loaded_image is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self._anchor_selection = Selection(point[0], point[1], point[0], point[1])
        self._selection = self._anchor_selection
        self._analysis_result = None
        self._render()

    def _update_selection(self, event: tk.Event[tk.Misc]) -> None:
        if self._loaded_image is None or self._anchor_selection is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self._selection = Selection(
            self._anchor_selection.x0,
            self._anchor_selection.y0,
            point[0],
            point[1],
        ).clipped(self._loaded_image.width, self._loaded_image.height)
        self._render()

    def _finish_selection(self, event: tk.Event[tk.Misc]) -> None:
        if self._loaded_image is None or self._anchor_selection is None:
            return

        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            self._anchor_selection = None
            return

        selection = Selection(
            self._anchor_selection.x0,
            self._anchor_selection.y0,
            point[0],
            point[1],
        ).clipped(self._loaded_image.width, self._loaded_image.height)
        self._anchor_selection = None

        if not selection.is_large_enough():
            self._selection = None
            self._render()
            return

        self._selection = selection.normalized()
        self._render()
        self._on_selection_committed(self._selection)

    def _draw_selection(self, selection: Selection) -> None:
        x0, y0 = self._image_to_canvas(selection.x0, selection.y0)
        x1, y1 = self._image_to_canvas(selection.x1, selection.y1)
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ffb347",
            width=2,
        )

    def _draw_boundaries(self, result: AnalysisResult) -> None:
        selection = result.selection
        if result.orientation == "horizontal":
            for boundary in result.boundaries[1:-1]:
                x = selection.x0 + int(boundary)
                x0, y0 = self._image_to_canvas(x, selection.y0)
                x1, y1 = self._image_to_canvas(x, selection.y1)
                self.canvas.create_line(x0, y0, x1, y1, fill="#52d1dc", width=1)
        else:
            for boundary in result.boundaries[1:-1]:
                y = selection.y0 + int(boundary)
                x0, y0 = self._image_to_canvas(selection.x0, y)
                x1, y1 = self._image_to_canvas(selection.x1, y)
                self.canvas.create_line(x0, y0, x1, y1, fill="#52d1dc", width=1)

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[int, int] | None:
        if self._loaded_image is None:
            return None

        offset_x, offset_y, draw_width, draw_height = self._display_rect
        if not (
            offset_x <= canvas_x <= offset_x + draw_width
            and offset_y <= canvas_y <= offset_y + draw_height
        ):
            return None

        preview_x = (canvas_x - offset_x) * self._loaded_image.preview_image.width / draw_width
        preview_y = (canvas_y - offset_y) * self._loaded_image.preview_image.height / draw_height
        image_x = int(round(preview_x * self._loaded_image.width / self._loaded_image.preview_image.width))
        image_y = int(round(preview_y * self._loaded_image.height / self._loaded_image.preview_image.height))

        return (
            max(0, min(image_x, self._loaded_image.width - 1)),
            max(0, min(image_y, self._loaded_image.height - 1)),
        )

    def _image_to_canvas(self, image_x: int, image_y: int) -> tuple[float, float]:
        if self._loaded_image is None:
            return 0.0, 0.0

        offset_x, offset_y, draw_width, draw_height = self._display_rect
        preview_x = image_x * self._loaded_image.preview_image.width / self._loaded_image.width
        preview_y = image_y * self._loaded_image.preview_image.height / self._loaded_image.height
        canvas_x = offset_x + preview_x * draw_width / self._loaded_image.preview_image.width
        canvas_y = offset_y + preview_y * draw_height / self._loaded_image.preview_image.height
        return canvas_x, canvas_y
