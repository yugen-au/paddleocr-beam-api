"""R2 image persistence: walk the result tree, save PIL Images to the mounted
R2 bucket, and replace them with URL-reference dicts."""
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from ocr.config import MOUNT_PATH


def should_save_to_file(obj) -> bool:
    """Heuristic: does this object look like an image/file/binary to persist?"""
    obj_type_str = str(type(obj)).lower()

    if any(keyword in obj_type_str for keyword in ["image", "file", "stream", "buffer", "binary"]):
        return True
    if hasattr(obj, "save") and hasattr(obj, "size"):
        return True
    if isinstance(obj, (bytes, bytearray)):
        return True
    if hasattr(obj, "width") and hasattr(obj, "height"):
        return True
    if hasattr(obj, "read") and hasattr(obj, "seek"):
        return True
    return False


def save_pil_image_to_r2(image_obj, folder_name: str, path_context: str) -> Dict[str, str]:
    """Save a PIL Image to the mounted R2 volume; return URL metadata dict."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{image_id}.png"

        output_dir = f"{MOUNT_PATH}/images/{folder_name}"
        os.makedirs(output_dir, exist_ok=True)
        full_path = f"{output_dir}/{filename}"

        image_obj.save(full_path)

        width = getattr(image_obj, "width", "?")
        height = getattr(image_obj, "height", "?")
        r2_path = f"images/{folder_name}/{filename}"

        print(f"Saved PIL Image to mounted volume: {full_path} -> {r2_path}")

        return {
            "type": "extracted_image",
            "url": r2_path,
            "size": f"{width}x{height}",
            "format": "PNG",
            "content_type": "image/png",
            "context": path_context.replace(".", "/"),
            "folder": folder_name,
            "local_path": full_path,
        }
    except Exception as e:
        print(f"Failed to save PIL Image to R2: {str(e)}")
        return {
            "type": "image_error",
            "error": f"Failed to save image: {str(e)}",
            "context": path_context,
            "object_type": str(type(image_obj)),
        }


def save_images_to_r2(data, session_id: Optional[str] = None, original_filename: Optional[str] = None) -> Any:
    """Recursively replace PIL Images in `data` with R2 URL references."""
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    if original_filename:
        clean_name = os.path.splitext(os.path.basename(original_filename))[0]
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in "_-")[:50]
    else:
        clean_name = "base64_document"

    folder_name = f"{clean_name}_{session_id}"

    def clean_recursive(obj, path=""):
        if isinstance(obj, dict):
            return {key: clean_recursive(value, f"{path}.{key}") for key, value in obj.items()}
        elif isinstance(obj, list):
            return [clean_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
        elif "PIL" in str(type(obj)) and "Image" in str(type(obj)):
            return save_pil_image_to_r2(obj, folder_name, path)
        return obj

    return clean_recursive(data)
