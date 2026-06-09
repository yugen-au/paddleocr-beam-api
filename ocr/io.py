"""Input preparation: accept either base64 data or an R2-uploaded file name,
return a local path ready for the pipeline."""
import base64
import os
import tempfile
from typing import Optional

from ocr.config import MOUNT_PATH


def _debug_listdir(path: str) -> None:
    """Best-effort directory listing for cold-start debugging (never fatal)."""
    try:
        print(f"Files in {path}:", os.listdir(path))
    except OSError as e:
        print(f"Could not list {path}: {e}")


def prepare_input_file(
    image_data: Optional[str] = None, file_name: Optional[str] = None
) -> str:
    """Prepare input from base64 data OR an R2 file upload.

    Returns a path ready for processing. Raises ValueError if neither/both are
    given, FileNotFoundError if the named upload is missing.
    """
    print("Current directory:", os.getcwd())
    _debug_listdir(MOUNT_PATH)

    if not image_data and not file_name:
        raise ValueError("Either image_data or file_name must be provided")

    if image_data and file_name:
        raise ValueError("Provide either image_data OR file_name, not both")

    if image_data:
        # Base64 input: strip data-URL prefix if present, decode to a temp file.
        if image_data.startswith("data:image") or image_data.startswith("data:application"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_bytes)
            return tmp_file.name

    # R2 file upload: read from the mounted bucket.
    assert file_name is not None  # guaranteed by the validation above
    file_path = os.path.join(MOUNT_PATH, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found in uploads: {file_name}")
    return file_path
