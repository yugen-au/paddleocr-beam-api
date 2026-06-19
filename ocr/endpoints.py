"""Modal service. Boots the pipeline once per container (@modal.enter), then
serves the two OCR endpoints as POST routes on a FastAPI app (@modal.asgi_app).

Per request we persist all artifacts to R2 under ocr/<session_id>/ (original,
per-page raw json + markdown + visualizations + extracted images, then a
result.json manifest written last as the commit marker) and return the keys
alongside the inline extraction.
"""
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import modal

from ocr import artifacts
from ocr.config import GPU, R2_BUCKET, bucket_for
from ocr.io import PreparedInput, prepare_input_file
from ocr.metrics import calculate_character_metrics
from ocr.pipeline import boot
from ocr.resources import SECRETS, VOLUMES, app, cpu_image, image


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]


def _persist_all(
    pipeline, prepared: PreparedInput, session_id: str,
    file_name: Optional[str], image_data: Optional[str], unwarp: bool = False,
    bucket: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run prediction, persist every artifact under the session prefix, write the
    manifest last. Returns (per-page summaries, r2 reference block)."""
    input_method = "base64" if image_data else "s3_upload"
    original_key = artifacts.save_original(session_id, prepared.data, prepared.ext, bucket=bucket)

    # use_doc_unwarping off unless the caller opts in -- the dewarp isn't
    # idempotent and bends edges even on flat docs (drive it from a skew check).
    output = pipeline.predict(prepared.path, use_doc_unwarping=unwarp)
    pages = [artifacts.persist_page(session_id, i + 1, res, bucket=bucket) for i, res in enumerate(output)]

    artifacts.save_manifest(session_id, original_key, file_name, input_method, pages, bucket=bucket)
    r2 = {
        "bucket": bucket or R2_BUCKET,
        "prefix": artifacts.session_prefix(session_id) + "/",
        "manifest_key": artifacts.manifest_key(session_id),
        "original_key": original_key,
    }
    return pages, r2


def _extract_and_analyze(
    pipeline,
    image_data: Optional[str],
    file_name: Optional[str],
    output_format: str,
    include_character_metrics: bool,
    include_layout_analysis: bool,
    unwarp: bool = False,
    private: bool = False,
) -> Dict[str, Any]:
    try:
        session_id = _session_id()
        bucket = bucket_for(private)
        prepared = prepare_input_file(image_data, file_name, bucket=bucket)
        try:
            print(f"Processing document with PaddleOCR-VL: {prepared.path}")
            pages, r2 = _persist_all(
                pipeline, prepared, session_id, file_name, image_data, unwarp, bucket=bucket
            )

            results = []
            for p in pages:
                rd = {
                    "page": p["page"],
                    "text_content": p["text_content"],
                    "structure_info": {"json": p["raw"]},
                    "artifacts": {
                        "raw_result": p["raw_result_key"],
                        "page_md": p["page_md_key"],
                        "viz": p["viz_keys"],
                        "extracted": p["extracted_keys"],
                    },
                }
                if output_format == "markdown":
                    rd["markdown"] = p["markdown_text"]
                if include_character_metrics:
                    rd["character_metrics"] = calculate_character_metrics(p["text_content"])
                results.append(rd)

            return {
                "success": True,
                "session_id": session_id,
                "r2": r2,
                "results": results,
                "total_pages": len(results),
                "original_filename": file_name,
                "input_method": "base64" if image_data else "s3_upload",
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "features_used": {
                        "doc_orientation_classify": True,
                        "doc_unwarping": unwarp,
                        "layout_detection": include_layout_analysis,
                    },
                },
            }
        finally:
            if prepared.is_temp and os.path.exists(prepared.path):
                os.unlink(prepared.path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


def _extract_simple(
    pipeline, image_data: Optional[str], file_name: Optional[str],
    unwarp: bool = False, private: bool = False,
) -> Dict[str, Any]:
    try:
        session_id = _session_id()
        bucket = bucket_for(private)
        prepared = prepare_input_file(image_data, file_name, bucket=bucket)
        try:
            print(f"Processing document with PaddleOCR-VL (simple): {prepared.path}")
            pages, r2 = _persist_all(
                pipeline, prepared, session_id, file_name, image_data, unwarp, bucket=bucket
            )

            full_text = "\n".join(p["text_content"] for p in pages if p["text_content"])
            words = full_text.split()
            return {
                "success": True,
                "session_id": session_id,
                "r2": r2,
                "extracted_text": full_text,
                "word_count": len(words),
                "character_count": len(full_text.replace(" ", "")),
                "character_metrics": calculate_character_metrics(full_text),
                "total_pages": len(pages),
                "input_method": "base64" if image_data else "s3_upload",
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "mode": "simple_extraction",
                },
            }
        finally:
            if prepared.is_temp and os.path.exists(prepared.path):
                os.unlink(prepared.path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


def _crop_and_section(
    session_id: Optional[str], margin: int = 20, target_ar: Optional[float] = None,
    private: bool = False, page: Optional[int] = None,
) -> Dict[str, Any]:
    """Crop each page of a prior OCR session to its text bound and split it into
    N sections (N = round(AR/target_ar)), cuts snapped to whitespace gaps. Reads
    the session's artifacts from R2, writes crops/sections back. CPU-only.

    If `page` (1-indexed) is given, only that page is processed (lets a per-page
    caller avoid re-cropping the whole session); otherwise all pages."""
    try:
        if not session_id:
            return {"success": False, "error": "session_id is required"}
        from io import BytesIO

        from PIL import Image

        from ocr import sectioning

        ta = float(target_ar) if target_ar else sectioning.SQRT2
        margin = int(margin)
        bucket = bucket_for(private)
        page_filter = int(page) if page is not None else None
        manifest = artifacts.get_json(artifacts.manifest_key(session_id), bucket=bucket)

        pages_out = []
        for pg in manifest.get("pages", []):
            if page_filter is not None and pg.get("page") != page_filter:
                continue
            prefix = pg["prefix"]
            raw = artifacts.get_json(pg["raw_result_key"], bucket=bucket)
            img_key = next((k for k in pg.get("viz_keys", []) if "preprocessed_output" in k), None)
            if not img_key:
                raise RuntimeError(f"no preprocessed_output for page {pg.get('page')}")

            img = Image.open(BytesIO(artifacts.get_bytes(img_key, bucket=bucket))).convert("RGB")
            res = raw.get("res", raw)
            rw, rh = res.get("width"), res.get("height")
            W, H = img.size
            sx, sy = (W / rw, H / rh) if (rw and rh and (rw, rh) != (W, H)) else (1.0, 1.0)

            polys = sectioning.polygons_from_result(raw, sx, sy)
            crop, sections, info = sectioning.section_page(img, polys, margin, ta)

            crop_key = artifacts.put_pil(f"{prefix}/crop.png", crop, bucket=bucket)
            section_keys = [artifacts.put_pil(f"{prefix}/sections/section_{i:02d}.png", s, bucket=bucket)
                            for i, s in enumerate(sections, 1)]
            pages_out.append({"page": pg["page"], **info,
                              "crop_key": crop_key, "section_keys": section_keys})

        if page_filter is not None and not pages_out:
            return {"success": False,
                    "error": f"page {page_filter} not found in session {session_id}"}

        return {
            "success": True,
            "session_id": session_id,
            "page": page_filter,
            "bucket": bucket,
            "margin": margin,
            "target_ar": round(ta, 4),
            "pages": pages_out,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


@app.cls(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    scaledown_window=300,  # keep warm ~5min after last request (cold-start mitigation)
    timeout=600,
)
@modal.concurrent(max_inputs=10)
class OCRService:
    @modal.enter()
    def _boot(self):
        # Runs once per container: start FastDeploy sidecar + build pipeline.
        self.pipeline = boot()

    @modal.asgi_app(requires_proxy_auth=True)  # Modal edge auth: caller sends Modal-Key/Modal-Secret
    def web(self):
        from fastapi import FastAPI

        web_app = FastAPI(title="PaddleOCR-VL")

        @web_app.post("/extract_text_and_analyze")
        def extract_text_and_analyze(body: dict):
            return _extract_and_analyze(
                self.pipeline,
                body.get("image_data"),
                body.get("file_name"),
                body.get("output_format", "json"),
                body.get("include_character_metrics", True),
                body.get("include_layout_analysis", True),
                body.get("unwarp", False),
                body.get("private", False),
            )

        @web_app.post("/extract_text_simple")
        def extract_text_simple(body: dict):
            return _extract_simple(
                self.pipeline,
                body.get("image_data"),
                body.get("file_name"),
                body.get("unwarp", False),
                body.get("private", False),
            )

        return web_app


@app.cls(
    image=cpu_image,        # CPU-only: pure geometry, no GPU/model stack
    secrets=SECRETS,
    scaledown_window=120,
    timeout=300,
)
@modal.concurrent(max_inputs=20)
class SectionService:
    @modal.asgi_app(requires_proxy_auth=True)  # Modal edge auth: caller sends Modal-Key/Modal-Secret
    def web(self):
        from fastapi import FastAPI

        web_app = FastAPI(title="PaddleOCR-VL Sectioning")

        @web_app.post("/crop_and_section")
        def crop_and_section(body: dict):
            return _crop_and_section(
                body.get("session_id"),
                body.get("margin", 20),
                body.get("target_ar"),
                body.get("private", False),
                body.get("page"),
            )

        return web_app
