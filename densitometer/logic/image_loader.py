from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import LoadedImage

PREVIEW_MAX_DIMENSION = 1800


def load_image(path: str | Path, preview_max_dimension: int = PREVIEW_MAX_DIMENSION) -> LoadedImage:
    image_path = Path(path)
    with Image.open(image_path) as source_image:
        source_image.load()
        image_mode = source_image.mode
        pixel_array = np.array(source_image)

    analysis_image = _normalize_to_luminance_16bit(pixel_array)
    preview_image = _build_preview_image(analysis_image, preview_max_dimension)
    height, width = analysis_image.shape

    return LoadedImage(
        path=image_path,
        analysis_image=analysis_image,
        preview_image=preview_image,
        mode=image_mode,
        source_dtype=str(pixel_array.dtype),
        width=width,
        height=height,
    )


def _normalize_to_luminance_16bit(pixel_array: np.ndarray) -> np.ndarray:
    if pixel_array.ndim == 2:
        mono = pixel_array.astype(np.float64, copy=False)
        mono = _scale_to_16bit(mono, pixel_array.dtype)
    elif pixel_array.ndim == 3 and pixel_array.shape[2] >= 3:
        rgb = pixel_array[..., :3].astype(np.float64, copy=False)
        rgb = _scale_to_16bit(rgb, pixel_array.dtype)
        mono = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))
    else:
        raise ValueError("Unsupported TIFF format. Use grayscale or RGB TIFF files.")

    mono = np.nan_to_num(mono, nan=0.0, posinf=65535.0, neginf=0.0)
    mono = np.clip(mono, 0.0, 65535.0)
    return mono.astype(np.float32)


def _scale_to_16bit(values: np.ndarray, source_dtype: np.dtype) -> np.ndarray:
    if source_dtype == np.uint8:
        return values * 257.0
    if np.issubdtype(source_dtype, np.integer) and np.max(values, initial=0.0) <= 255.0:
        return values * 257.0
    return values


def _build_preview_image(analysis_image: np.ndarray, max_dimension: int) -> Image.Image:
    low_percentile, high_percentile = np.percentile(analysis_image, [1.0, 99.5])
    if high_percentile <= low_percentile:
        high_percentile = max(float(np.max(analysis_image, initial=1.0)), 1.0)
        low_percentile = float(np.min(analysis_image, initial=0.0))

    scaled = np.clip(
        (analysis_image - low_percentile) / max(high_percentile - low_percentile, 1e-6),
        0.0,
        1.0,
    )
    preview_array = np.round(scaled * 255.0).astype(np.uint8)
    preview_image = Image.fromarray(preview_array, mode="L").convert("RGB")

    if max(preview_image.size) > max_dimension:
        scale = max_dimension / max(preview_image.size)
        preview_image = preview_image.resize(
            (
                max(1, int(round(preview_image.width * scale))),
                max(1, int(round(preview_image.height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )

    return preview_image
