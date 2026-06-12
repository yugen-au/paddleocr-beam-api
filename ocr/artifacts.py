"""R2 artifact persistence via the S3 API (boto3).

Why boto3 and not the CloudBucketMount: the mount can't set Content-Type, so
objects would store as application/octet-stream and a browser wouldn't render
them. put_object sets the right type for admin/public serving. The bucket stays
private — the API returns {bucket, key}; a consumer presigns or fetches with the
shared creds.

Flat, session-keyed layout (deletion is app-managed; no lifecycle rule):

    ocr/<session_id>/
        original/input.<ext>
        result.json                     # manifest, written LAST = commit marker
        p0001/
            raw_result.json             # full res.json (raw, not joined)
            page.md                     # res.markdown['markdown_texts']
            viz/<layer>.png             # res.img layers (preprocessed_img, layout_det_res)
            extracted/<name>            # res.markdown['markdown_images'] crops
"""
import io
import json
import os
from typing import Any, Dict, List, Optional

from ocr.config import ARTIFACT_ROOT, R2_BUCKET, R2_ENDPOINT

# NB: boto3 is imported lazily inside _s3() — it lives only in the container
# image, not the local deploy env (same pattern as paddleocr in pipeline.py), so
# `modal deploy` can import this module locally without boto3 installed.

_CACHE_CONTROL = "public, max-age=31536000, immutable"
_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".tiff": "image/tiff",
    ".pdf": "application/pdf", ".json": "application/json",
    ".md": "text/markdown; charset=utf-8",
}

_client = None


def _s3():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        # Creds (AWS_ACCESS_KEY_ID/SECRET) come from the r2-creds secret in env.
        # Path-style addressing avoids bucket-in-host DNS/cert issues on R2.
        _client = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT, region_name="auto",
            config=Config(s3={"addressing_style": "path"}),
        )
    return _client


def _content_type(key: str) -> str:
    return _CONTENT_TYPES.get(os.path.splitext(key)[1].lower(), "application/octet-stream")


def put_bytes(key: str, data: bytes, content_type: Optional[str] = None) -> str:
    _s3().put_object(
        Bucket=R2_BUCKET, Key=key, Body=data,
        ContentType=content_type or _content_type(key),
        CacheControl=_CACHE_CONTROL,
    )
    return key


def put_pil(key: str, image, fmt: str = "PNG") -> str:
    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return put_bytes(key, buf.getvalue())


def put_json(key: str, obj: Any) -> str:
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    return put_bytes(key, body, "application/json")


def session_prefix(session_id: str) -> str:
    return f"{ARTIFACT_ROOT}/{session_id}"


def manifest_key(session_id: str) -> str:
    return f"{session_prefix(session_id)}/result.json"


def save_original(session_id: str, data: bytes, ext: str) -> str:
    return put_bytes(f"{session_prefix(session_id)}/original/input{ext}", data)


def _block_order(b: Dict[str, Any]) -> int:
    """Sort key tolerant of missing/None ordering fields (image blocks can have
    block_order=None, which breaks a naive None-vs-int comparison)."""
    for key in ("block_order", "block_id"):
        v = b.get(key)
        if isinstance(v, int):
            return v
    return 0


def _join_block_content(raw: Dict[str, Any]) -> str:
    """Reading-order plain text from parsing_res_list block_content."""
    res = raw.get("res", raw) if isinstance(raw, dict) else {}
    blocks = res.get("parsing_res_list") or []
    parts = []
    for b in sorted(blocks, key=_block_order):
        content = b.get("block_content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def persist_page(session_id: str, page_no: int, res) -> Dict[str, Any]:
    """Write raw_result.json, page.md, viz/*, extracted/* for one page.

    Returns a summary with the written keys, the joined text, and the raw json +
    markdown text (so the endpoint can build its response without recomputing the
    heavy res.img/res.markdown properties)."""
    pdir = f"{session_prefix(session_id)}/p{page_no:04d}"

    raw = res.json  # {'res': {...}}
    raw_key = put_json(f"{pdir}/raw_result.json", raw)

    md = res.markdown if isinstance(res.markdown, dict) else {}
    md_text = md.get("markdown_texts") or ""
    md_key = put_bytes(f"{pdir}/page.md", md_text.encode("utf-8"), "text/markdown; charset=utf-8") if md_text else None

    viz_keys: List[str] = []
    for name, img in (res.img or {}).items():
        viz_keys.append(put_pil(f"{pdir}/viz/{name}.png", img))

    extracted_keys: List[str] = []
    for relpath, img in (md.get("markdown_images") or {}).items():
        name = os.path.basename(relpath)
        fmt = "JPEG" if name.lower().endswith((".jpg", ".jpeg")) else "PNG"
        extracted_keys.append(put_pil(f"{pdir}/extracted/{name}", img, fmt))

    return {
        "page": page_no,
        "prefix": pdir,
        "raw_result_key": raw_key,
        "page_md_key": md_key,
        "viz_keys": viz_keys,
        "extracted_keys": extracted_keys,
        "text_content": _join_block_content(raw),
        "raw": raw,             # not stored in manifest; for inline response
        "markdown_text": md_text,
    }


# Keys persisted into the manifest (raw/markdown_text are response-only).
_MANIFEST_PAGE_KEYS = (
    "page", "prefix", "raw_result_key", "page_md_key",
    "viz_keys", "extracted_keys", "text_content",
)


def save_manifest(
    session_id: str, original_key: str, original_filename: Optional[str],
    input_method: str, pages: List[Dict[str, Any]],
) -> str:
    """Write result.json LAST — its presence marks the request complete."""
    manifest = {
        "session_id": session_id,
        "model": "PaddleOCR-VL",
        "original_filename": original_filename,
        "input_method": input_method,
        "original_key": original_key,
        "total_pages": len(pages),
        "pages": [{k: p[k] for k in _MANIFEST_PAGE_KEYS} for p in pages],
    }
    return put_json(manifest_key(session_id), manifest)
