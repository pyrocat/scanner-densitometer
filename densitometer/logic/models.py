from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL.Image import Image as PilImage


ReferenceMode = Literal["selection_max", "full_scale"]
Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class LoadedImage:
    path: Path
    analysis_image: np.ndarray
    preview_image: PilImage
    mode: str
    source_dtype: str
    width: int
    height: int
    full_scale: float = 65535.0


@dataclass(frozen=True, slots=True)
class Selection:
    x0: int
    y0: int
    x1: int
    y1: int

    def normalized(self) -> "Selection":
        return Selection(
            x0=min(self.x0, self.x1),
            y0=min(self.y0, self.y1),
            x1=max(self.x0, self.x1),
            y1=max(self.y0, self.y1),
        )

    def clipped(self, width: int, height: int) -> "Selection":
        normalized = self.normalized()
        return Selection(
            x0=max(0, min(normalized.x0, width - 1)),
            y0=max(0, min(normalized.y0, height - 1)),
            x1=max(1, min(normalized.x1, width)),
            y1=max(1, min(normalized.y1, height)),
        )

    @property
    def width(self) -> int:
        normalized = self.normalized()
        return normalized.x1 - normalized.x0

    @property
    def height(self) -> int:
        normalized = self.normalized()
        return normalized.y1 - normalized.y0

    def is_large_enough(self, minimum_size: int = 21) -> bool:
        return self.width >= minimum_size and self.height >= minimum_size


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    step_count: int = 21
    reference_mode: ReferenceMode = "selection_max"


@dataclass(frozen=True, slots=True)
class StepMeasurement:
    graph_index: int
    spatial_index: int
    signal: float
    density: float
    relative_log_exposure: float
    wedge_density: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    selection: Selection
    orientation: Orientation
    boundaries: np.ndarray
    profile: np.ndarray
    smoothed_profile: np.ndarray
    measurements: tuple[StepMeasurement, ...]
    reference_mode: ReferenceMode
    reference_value: float

    @property
    def x_values(self) -> np.ndarray:
        return np.array(
            [measurement.relative_log_exposure for measurement in self.measurements],
            dtype=np.float64,
        )

    @property
    def y_values(self) -> np.ndarray:
        return np.array(
            [measurement.density for measurement in self.measurements],
            dtype=np.float64,
        )
