"""Input preparation: accept either base64 data or an R2-uploaded file name,
return a local path PLUS the original bytes + real file extension (so the
original can be persisted with the correct type).

The file TYPE is decided from the bytes (magic sniff), never from a caller-
supplied extension/mime — a wrong or missing extension must not mislabel the
persisted original or mis-dispatch the pipeline (PaddleX picks PDF-vs-image by
extension). Both paths hand predict() a temp file named with the sniffed ext.
"""
import base64
import os
import tempfile
from typing import NamedTuple, Optional

from ocr.config import MOUNT_PATH


class PreparedInput(NamedTuple):
    path: str       # local temp path ready for pipeline.predict (correctly named)
    data: bytes     # original bytes (for persisting to R2)
    ext: str        # real extension incl. dot, sniffed from content: ".png"/".pdf"/...
    is_temp: bool    # always True now — path is a temp file to unlink after use


# Magic-byte signatures -> extension.
_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"%PDF", ".pdf"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"II*\x00", ".tiff"),
    (b"MM\x00*", ".tiff"),
]


def _sniff_ext(data: bytes, default: str = "") -> str:
    """Extension from the leading magic bytes, or `default` if unrecognised."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _MAGIC:
        if data.startswith(sig):
            return ext
    return default


def _resolve_ext(data: bytes, file_name: Optional[str] = None) -> str:
    """Content is authoritative: sniff the bytes; only if unrecognised fall back
    to the filename's extension, then `.png`."""
    return _sniff_ext(data) or (os.path.splitext(file_name or "")[1].lower() or ".png")


def prepare_input_file(
    image_data: Optional[str] = None, file_name: Optional[str] = None
) -> PreparedInput:
    """Prepare input from base64 data OR an R2 file upload.

    Raises ValueError if neither/both are given, FileNotFoundError if the named
    upload is missing.
    """
    if not image_data and not file_name:
        raise ValueError("Either image_data or file_name must be provided")
    if image_data and file_name:
        raise ValueError("Provide either image_data OR file_name, not both")

    if image_data:
        # Base64 input: strip data-URL prefix if present, decode.
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        data = base64.b64decode(image_data)
    else:
        # R2 file upload: read the bytes from the mounted bucket.
        assert file_name is not None  # guaranteed by validation above
        file_path = os.path.join(MOUNT_PATH, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found in uploads: {file_name}")
        with open(file_path, "rb") as f:
            data = f.read()

    # Type from content. Write a temp file named with the sniffed ext so predict()
    # always dispatches correctly regardless of the upload's filename.
    ext = _resolve_ext(data, file_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(data)
        return PreparedInput(tmp_file.name, data, ext, True)
