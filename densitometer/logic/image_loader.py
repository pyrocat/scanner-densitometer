"""Load scanner images into separate analysis and display representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import LoadedImage

PREVIEW_MAX_DIMENSION = 1800


def load_image(
    path: str | Path,
    preview_max_dimension: int = PREVIEW_MAX_DIMENSION,
) -> LoadedImage:
    """Load an image and prepare full-resolution analysis and preview data.

    Args:
        path: Path to an image that Pillow can decode.
        preview_max_dimension: Maximum width or height of the display preview,
            in pixels.

    Returns:
        Image data, source metadata, and an RGB preview packaged as a
        :class:`LoadedImage`.

    Raises:
        OSError: If Pillow cannot open or decode the image.
        ValueError: If the decoded pixels are neither grayscale nor RGB-like.
    """
    image_path = Path(path)
    with Image.open(image_path) as source_image:
        # Decode while Pillow still owns an open file handle; the NumPy array
        # remains valid after the context manager closes the source image.
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
    """Convert grayscale or RGB-like pixels to 16-bit-scale luminance.

    Args:
        pixel_array: A two-dimensional grayscale array or an array whose last
            dimension contains at least red, green, and blue channels.

    Returns:
        A two-dimensional ``float32`` array clipped to the inclusive range
        0–65535. Common 8-bit inputs are expanded to that range.

    Raises:
        ValueError: If ``pixel_array`` has an unsupported shape.
    """
    if pixel_array.ndim == 2:
        mono = pixel_array.astype(np.float64, copy=False)
        mono = _scale_to_16bit(mono, pixel_array.dtype)
    elif pixel_array.ndim == 3 and pixel_array.shape[2] >= 3:
        # Alpha and any extra channels do not contribute to measured luminance.
        rgb = pixel_array[..., :3].astype(np.float64, copy=False)
        rgb = _scale_to_16bit(rgb, pixel_array.dtype)
        # Rec. 709 coefficients retain perceptual brightness when RGB scans are
        # reduced to the single channel required by the density analysis.
        mono = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))
    else:
        raise ValueError("Unsupported TIFF format. Use grayscale or RGB TIFF files.")

    # Sanitize floating-point sources before conversion so invalid samples do
    # not propagate into percentile calculations or density measurements.
    mono = np.nan_to_num(mono, nan=0.0, posinf=65535.0, neginf=0.0)
    mono = np.clip(mono, 0.0, 65535.0)
    return mono.astype(np.float32)


def _scale_to_16bit(values: np.ndarray, source_dtype: np.dtype) -> np.ndarray:
    """Expand integer samples with an 8-bit range into a 16-bit-scale range.

    Args:
        values: Pixel values represented as floating-point numbers.
        source_dtype: Data type of the image before conversion to ``values``.

    Returns:
        ``values`` multiplied by 257 when they represent 8-bit integer data;
        otherwise, the original values unchanged.
    """
    if source_dtype == np.uint8:
        # 257 maps both endpoints exactly: 0 to 0 and 255 to 65535.
        return values * 257.0
    # Some decoders expose low-bit-depth samples in a wider integer dtype.
    if np.issubdtype(source_dtype, np.integer) and np.max(values, initial=0.0) <= 255.0:
        return values * 257.0
    return values


def _build_preview_image(analysis_image: np.ndarray, max_dimension: int) -> Image.Image:
    """Create a contrast-stretched RGB preview without altering analysis data.

    Args:
        analysis_image: Full-resolution, two-dimensional luminance samples.
        max_dimension: Maximum allowed width or height of the returned preview.

    Returns:
        An 8-bit RGB Pillow image, resized proportionally when necessary.
    """
    # Percentiles keep a few unusually dark or bright pixels from flattening
    # the display contrast. This stretch applies only to the preview.
    low_percentile, high_percentile = np.percentile(analysis_image, [1.0, 99.5])
    if high_percentile <= low_percentile:
        # Constant or near-constant images need a non-zero display range.
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
        # Apply one scale factor to preserve the source aspect ratio.
        scale = max_dimension / max(preview_image.size)
        preview_image = preview_image.resize(
            (
                max(1, int(round(preview_image.width * scale))),
                max(1, int(round(preview_image.height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )

    return preview_image
