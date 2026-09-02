from __future__ import annotations

import math
import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

from ..logic.models import T2115_DENSITIES, LoadedImage, Selection

# Pointer distance, in canvas pixels, within which a handle is grabbed.
HANDLE_RADIUS = 7
# Width given to a freshly dragged strip, as a fraction of its length. The
# T2115 is 0.5 in by 5.25 in (0.095); a little narrower stays inside the steps.
DEFAULT_ASPECT = 0.07
# Strips shorter than this many image pixels are treated as accidental clicks.
MINIMUM_LENGTH = 21


class ImageCanvas(ttk.Frame):
    """Show the scan and let the user place a rotatable, resizable strip.

    Drag on empty image to draw the centreline from one end of the wedge to
    the other. Then drag an end handle to rotate or stretch, a side handle to
    set the width, or the inside of the strip to move it.
    """

    def __init__(
        self,
        master: tk.Misc,
        on_selection_committed: Callable[[Selection], None],
        step_count: int = len(T2115_DENSITIES),
    ) -> None:
        super().__init__(master)
        self._on_selection_committed = on_selection_committed
        self._step_count = step_count

        self.canvas = tk.Canvas(self, background="#161616", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._loaded_image: LoadedImage | None = None
        self._selection: Selection | None = None
        # Active gesture: ("new",), ("end0",), ("end1",), ("side0",), ("side1",) or
        # ("move", last_x, last_y) with the previous pointer position.
        self._drag: tuple | None = None
        self._photo_image: ImageTk.PhotoImage | None = None
        self._cached_preview_size: tuple[int, int] | None = None
        self._cached_resized_image: Image.Image | None = None
        self._display_rect = (0.0, 0.0, 1.0, 1.0)

        self.canvas.bind("<Configure>", self._render)
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._update_drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish_drag)

    def set_image(self, loaded_image: LoadedImage | None) -> None:
        self._loaded_image = loaded_image
        self._selection = None
        self._drag = None
        self._cached_preview_size = None
        self._cached_resized_image = None
        self._render()

    def set_selection(self, selection: Selection | None) -> None:
        self._selection = selection
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
                text="Open a 16-bit TIFF, then drag from one end of the step wedge to the other.",
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

    def _start_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._loaded_image is None:
            return
        selection = self._selection
        if selection is not None:
            for name, (handle_x, handle_y) in self._handles(selection).items():
                if math.hypot(event.x - handle_x, event.y - handle_y) <= HANDLE_RADIUS:
                    self._drag = (name,)
                    return
            image_x, image_y = self._canvas_to_image(event.x, event.y)
            if selection.contains(image_x, image_y):
                self._drag = ("move", image_x, image_y)
                return
        if not self._inside_display(event.x, event.y):
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        self._selection = Selection(image_x, image_y, image_x, image_y, 0.0)
        self._drag = ("new",)
        self._render()

    def _update_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._drag is None or self._selection is None:
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        selection = self._selection
        kind = self._drag[0]
        if kind == "new":
            selection = replace(selection, x1=image_x, y1=image_y)
            selection = replace(selection, width=max(8.0, selection.length * DEFAULT_ASPECT))
        elif kind == "end0":
            selection = replace(selection, x0=image_x, y0=image_y)
        elif kind == "end1":
            selection = replace(selection, x1=image_x, y1=image_y)
        elif kind.startswith("side"):
            # The width stays centred on the line: twice the pointer's offset.
            selection = replace(selection, width=max(2.0, 2.0 * abs(selection.local(image_x, image_y)[1])))
        elif kind == "move":
            _, last_x, last_y = self._drag
            dx, dy = image_x - last_x, image_y - last_y
            selection = replace(selection, x0=selection.x0 + dx, y0=selection.y0 + dy, x1=selection.x1 + dx, y1=selection.y1 + dy)
            self._drag = ("move", image_x, image_y)
        self._selection = selection
        self._render()

    def _finish_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._drag is None:
            return
        self._update_drag(event)
        self._drag = None
        selection = self._selection
        if selection is None or selection.length < MINIMUM_LENGTH:
            self._selection = None
            self._render()
            return
        self._on_selection_committed(selection)

    def _handles(self, selection: Selection) -> dict[str, tuple[float, float]]:
        """Return canvas positions of the drag handles."""
        half = selection.width / 2.0
        middle = selection.length / 2.0
        return {
            "end0": self._image_to_canvas(selection.x0, selection.y0),
            "end1": self._image_to_canvas(selection.x1, selection.y1),
            "side0": self._image_to_canvas(*selection.point_at(middle, -half)),
            "side1": self._image_to_canvas(*selection.point_at(middle, half)),
        }

    def _draw_selection(self, selection: Selection) -> None:
        corners = [self._image_to_canvas(x, y) for x, y in selection.corners()]
        self.canvas.create_polygon(*corners, outline="#ffb347", fill="", width=2)

        # The wedge geometry is fixed, so the cell grid follows the strip live.
        half = selection.width / 2.0
        for index in range(1, self._step_count):
            along = selection.length * index / self._step_count
            x0, y0 = self._image_to_canvas(*selection.point_at(along, -half))
            x1, y1 = self._image_to_canvas(*selection.point_at(along, half))
            self.canvas.create_line(x0, y0, x1, y1, fill="#52d1dc", width=1)

        for name, (x, y) in self._handles(selection).items():
            fill = "#ffb347" if name.startswith("end") else "#52d1dc"
            self.canvas.create_oval(
                x - HANDLE_RADIUS, y - HANDLE_RADIUS, x + HANDLE_RADIUS, y + HANDLE_RADIUS, fill=fill, outline="#161616"
            )

    def _inside_display(self, canvas_x: float, canvas_y: float) -> bool:
        offset_x, offset_y, draw_width, draw_height = self._display_rect
        return offset_x <= canvas_x <= offset_x + draw_width and offset_y <= canvas_y <= offset_y + draw_height

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        """Map a canvas point to continuous image coordinates, clamped to the image."""
        if self._loaded_image is None:
            return 0.0, 0.0
        offset_x, offset_y, draw_width, draw_height = self._display_rect
        image_x = (canvas_x - offset_x) * self._loaded_image.width / draw_width
        image_y = (canvas_y - offset_y) * self._loaded_image.height / draw_height
        return (
            max(0.0, min(image_x, float(self._loaded_image.width))),
            max(0.0, min(image_y, float(self._loaded_image.height))),
        )

    def _image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float]:
        if self._loaded_image is None:
            return 0.0, 0.0
        offset_x, offset_y, draw_width, draw_height = self._display_rect
        return (
            offset_x + image_x * draw_width / self._loaded_image.width,
            offset_y + image_y * draw_height / self._loaded_image.height,
        )
