"""Input preparation: accept either base64 data or an R2-uploaded file name,
return a local path PLUS the original bytes + real file extension (so the
original can be persisted with the correct type)."""
import base64
import os
import tempfile
from typing import NamedTuple, Optional

from ocr.config import MOUNT_PATH


class PreparedInput(NamedTuple):
    path: str       # local path ready for pipeline.predict
    data: bytes     # original bytes (for persisting to R2)
    ext: str        # real extension incl. dot: ".png" / ".jpg" / ".pdf"
    is_temp: bool    # True if path is a temp file to unlink after use


# Magic-byte signatures -> extension. base64 inputs carry no filename, so the
# type must be sniffed (a base64 PDF was previously mislabeled .png).
_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"%PDF", ".pdf"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"II*\x00", ".tiff"),
    (b"MM\x00*", ".tiff"),
]


def _sniff_ext(data: bytes, default: str = ".png") -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _MAGIC:
        if data.startswith(sig):
            return ext
    return default


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
        # Base64 input: strip data-URL prefix if present, sniff type, decode.
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        data = base64.b64decode(image_data)
        ext = _sniff_ext(data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(data)
            return PreparedInput(tmp_file.name, data, ext, True)

    # R2 file upload: read from the mounted bucket.
    assert file_name is not None  # guaranteed by validation above
    file_path = os.path.join(MOUNT_PATH, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found in uploads: {file_name}")
    with open(file_path, "rb") as f:
        data = f.read()
    ext = (os.path.splitext(file_name)[1] or _sniff_ext(data)).lower()
    return PreparedInput(file_path, data, ext, False)
