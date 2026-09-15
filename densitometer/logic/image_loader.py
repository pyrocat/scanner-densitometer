"""Load scanner images into separate analysis and display representations.

TIFF files are decoded with ``tifffile`` and interpreted from their own tags
(bit depth, sample format, samples per pixel, photometric interpretation) so
that every stored sample reaches the analysis unchanged; Pillow would reduce
16-bit RGB to 8 bits while decoding. Other formats fall back to Pillow under a
strict mode contract. Signal scaling always follows the declared bit depth,
never the image content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .models import LoadedImage

PREVIEW_MAX_DIMENSION = 1800
TIFF_SUFFIXES = (".tif", ".tiff")
# Nominal full scale of the analysis image. Every bit depth is rescaled to it
# from its declared range: 8-bit data by 257, 16-bit data by 1.
ANALYSIS_FULL_SCALE = 65535.0
# Rec. 709 luminance weights, applied to linear samples so RGB scans reduce to
# the single channel the density analysis needs.
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)

# TIFF PhotometricInterpretation values the loader understands, and names for
# the ones it refuses.
MIN_IS_WHITE = 0
MIN_IS_BLACK = 1
RGB = 2
PHOTOMETRIC_NAMES = {
    0: "MinIsWhite",
    1: "MinIsBlack",
    2: "RGB",
    3: "Palette",
    4: "TransparencyMask",
    5: "Separated (CMYK)",
    6: "YCbCr",
    8: "CIELab",
    9: "ICCLab",
    10: "ITULab",
    32803: "CFA",
    32844: "LogL",
    32845: "LogLuv",
    34892: "LinearRaw",
}
SAMPLE_FORMAT_UNSIGNED = 1
SAMPLE_FORMAT_NAMES = {
    1: "unsigned integer",
    2: "signed integer",
    3: "floating point",
    4: "undefined",
    5: "complex integer",
    6: "complex floating point",
}
# Pillow modes with an unambiguous integer range, for non-TIFF files.
PILLOW_MODE_BITS = {
    "L": 8,
    "LA": 8,
    "RGB": 8,
    "RGBA": 8,
    "RGBX": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
    "I;16N": 16,
}


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """Analysis samples plus the metadata that determined their scaling.

    Attributes:
        analysis_image: Two-dimensional ``float32`` luminance in the inclusive
            range 0 to ``ANALYSIS_FULL_SCALE``.
        mode: Short description of the source encoding.
        source_dtype: NumPy data type of the decoded source samples.
        bits_per_sample: Declared bit depth of the source samples.
        photometric: Name of the photometric interpretation that was applied.
    """

    analysis_image: np.ndarray
    mode: str
    source_dtype: str
    bits_per_sample: int
    photometric: str


def load_image(
    path: str | Path,
    preview_max_dimension: int = PREVIEW_MAX_DIMENSION,
) -> LoadedImage:
    """Load an image and prepare full-resolution analysis and preview data.

    Args:
        path: Path to a TIFF file, or to another image Pillow can decode.
        preview_max_dimension: Maximum width or height of the display preview,
            in pixels.

    Returns:
        Image data, source metadata, and an RGB preview packaged as a
        :class:`LoadedImage`.

    Raises:
        OSError: If the file cannot be opened or decoded.
        ValueError: If the encoding is outside the supported contract.
    """
    image_path = Path(path)
    if image_path.suffix.lower() in TIFF_SUFFIXES:
        decoded = decode_tiff(image_path)
    else:
        decoded = decode_with_pillow(image_path)
    preview_image = _build_preview_image(decoded.analysis_image, preview_max_dimension)
    height, width = decoded.analysis_image.shape
    return LoadedImage(
        path=image_path,
        analysis_image=decoded.analysis_image,
        preview_image=preview_image,
        mode=decoded.mode,
        source_dtype=decoded.source_dtype,
        width=width,
        height=height,
        bits_per_sample=decoded.bits_per_sample,
        photometric=decoded.photometric,
    )


def decode_tiff(path: str | Path) -> DecodedImage:
    """Decode the first image of a TIFF file from its directory tags.

    Supported: unsigned integer samples of 1 to 16 bits, MinIsWhite,
    MinIsBlack or RGB photometric interpretation, any layout and compression
    ``tifffile`` can decode. Extra samples such as alpha are ignored.

    Raises:
        ValueError: If the file is not a decodable TIFF or uses an unsupported
            sample format, bit depth, or photometric interpretation.
    """
    try:
        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            photometric = int(page.photometric)
            bits = int(page.bitspersample)
            sample_format = int(page.sampleformat)
            samples_per_pixel = int(page.samplesperpixel)
            compression = page.compression
            height, width = int(page.imagelength), int(page.imagewidth)
            _check_tiff_contract(photometric, bits, sample_format, samples_per_pixel)
            samples = np.asarray(page.asarray())
    except tifffile.TiffFileError as error:
        raise ValueError(f"Cannot decode TIFF: {error}") from None

    samples = _samples_last(samples, height, width)
    if not np.issubdtype(samples.dtype, np.unsignedinteger):
        raise ValueError(f"Unsupported TIFF: decoded samples are {samples.dtype}, not unsigned integers.")
    compression_name = getattr(compression, "name", str(compression))
    mode = f"{PHOTOMETRIC_NAMES[photometric]} {bits}-bit"
    if int(compression) != 1:
        mode += f", {compression_name}"
    return DecodedImage(
        analysis_image=_to_analysis_image(samples, bits, photometric),
        mode=mode,
        source_dtype=str(samples.dtype),
        bits_per_sample=bits,
        photometric=PHOTOMETRIC_NAMES[photometric],
    )


def decode_with_pillow(path: str | Path) -> DecodedImage:
    """Decode a non-TIFF image with Pillow under a strict mode contract.

    Only modes with an unambiguous integer range are accepted: 8-bit
    grayscale and RGB (with or without alpha) and 16-bit grayscale. Pillow
    decodes 16-bit colour images to 8 bits, so such files should be supplied
    as TIFF instead.

    Raises:
        OSError: If Pillow cannot open or decode the image.
        ValueError: If the decoded mode is outside the contract.
    """
    with Image.open(path) as source_image:
        # Decode while Pillow still owns an open file handle; the NumPy array
        # remains valid after the context manager closes the source image.
        source_image.load()
        mode = source_image.mode
        samples = np.array(source_image)
    bits = PILLOW_MODE_BITS.get(mode)
    if bits is None:
        raise ValueError(
            f"Unsupported image mode {mode!r}. Use an 8- or 16-bit grayscale or RGB image, preferably a TIFF."
        )
    photometric = RGB if mode.startswith("RGB") else MIN_IS_BLACK
    if samples.ndim == 2:
        samples = samples[..., np.newaxis]
    return DecodedImage(
        analysis_image=_to_analysis_image(samples, bits, photometric),
        mode=f"{mode} {bits}-bit",
        source_dtype=str(samples.dtype),
        bits_per_sample=bits,
        photometric=PHOTOMETRIC_NAMES[photometric],
    )


def _check_tiff_contract(photometric: int, bits: int, sample_format: int, samples_per_pixel: int) -> None:
    """Refuse encodings whose brightness meaning or range the loader cannot state."""
    if sample_format != SAMPLE_FORMAT_UNSIGNED:
        name = SAMPLE_FORMAT_NAMES.get(sample_format, str(sample_format))
        raise ValueError(f"Unsupported TIFF sample format: {name}. Use an unsigned-integer scan.")
    if not 1 <= bits <= 16:
        raise ValueError(f"Unsupported TIFF bit depth: {bits} bits per sample. Use an 8- or 16-bit scan.")
    if photometric not in (MIN_IS_WHITE, MIN_IS_BLACK, RGB):
        name = PHOTOMETRIC_NAMES.get(photometric, str(photometric))
        raise ValueError(f"Unsupported TIFF photometric interpretation: {name}. Use a grayscale or RGB scan.")
    if photometric == RGB and samples_per_pixel < 3:
        raise ValueError(f"Unsupported TIFF: RGB data with {samples_per_pixel} sample(s) per pixel.")


def _samples_last(samples: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return samples as ``(height, width, samples_per_pixel)`` whatever the stored order."""
    if samples.ndim == 2:
        return samples[..., np.newaxis]
    if samples.ndim == 3 and samples.shape[:2] == (height, width):
        return samples
    if samples.ndim == 3 and samples.shape[1:] == (height, width):
        # Planar files may come back with the sample axis first.
        return np.moveaxis(samples, 0, -1)
    raise ValueError(f"Unsupported TIFF layout: decoded shape {samples.shape} for a {width}x{height} image.")


def _to_analysis_image(samples: np.ndarray, bits: int, photometric: int) -> np.ndarray:
    """Convert stored samples to 16-bit-scale luminance.

    Args:
        samples: Unsigned integer array of shape ``(height, width, channels)``.
        bits: Declared bits per sample; sets the full-scale value.
        photometric: ``MIN_IS_WHITE``, ``MIN_IS_BLACK`` or ``RGB``.

    Returns:
        A ``float32`` array of shape ``(height, width)`` scaled so the declared
        full scale maps to ``ANALYSIS_FULL_SCALE``.
    """
    full_scale = float(2**bits - 1)
    if photometric == RGB:
        mono = np.zeros(samples.shape[:2], dtype=np.float32)
        for channel, weight in enumerate(LUMINANCE_WEIGHTS):
            mono += np.float32(weight) * samples[..., channel].astype(np.float32)
    else:
        mono = samples[..., 0].astype(np.float32)
        if photometric == MIN_IS_WHITE:
            # Zero means white in the file; the analysis expects zero as black.
            mono = np.float32(full_scale) - mono
    mono *= np.float32(ANALYSIS_FULL_SCALE / full_scale)
    return np.clip(mono, 0.0, ANALYSIS_FULL_SCALE)


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
