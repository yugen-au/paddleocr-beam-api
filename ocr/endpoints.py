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
from ocr.config import GPU, R2_BUCKET
from ocr.io import PreparedInput, prepare_input_file
from ocr.metrics import calculate_character_metrics
from ocr.pipeline import boot
from ocr.resources import SECRETS, VOLUMES, app, image


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]


def _persist_all(
    pipeline, prepared: PreparedInput, session_id: str,
    file_name: Optional[str], image_data: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run prediction, persist every artifact under the session prefix, write the
    manifest last. Returns (per-page summaries, r2 reference block)."""
    input_method = "base64" if image_data else "s3_upload"
    original_key = artifacts.save_original(session_id, prepared.data, prepared.ext)

    output = pipeline.predict(prepared.path)
    pages = [artifacts.persist_page(session_id, i + 1, res) for i, res in enumerate(output)]

    artifacts.save_manifest(session_id, original_key, file_name, input_method, pages)
    r2 = {
        "bucket": R2_BUCKET,
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
) -> Dict[str, Any]:
    try:
        session_id = _session_id()
        prepared = prepare_input_file(image_data, file_name)
        try:
            print(f"Processing document with PaddleOCR-VL: {prepared.path}")
            pages, r2 = _persist_all(pipeline, prepared, session_id, file_name, image_data)

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
                        "doc_unwarping": True,
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
    pipeline, image_data: Optional[str], file_name: Optional[str]
) -> Dict[str, Any]:
    try:
        session_id = _session_id()
        prepared = prepare_input_file(image_data, file_name)
        try:
            print(f"Processing document with PaddleOCR-VL (simple): {prepared.path}")
            pages, r2 = _persist_all(pipeline, prepared, session_id, file_name, image_data)

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

    @modal.asgi_app()
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
            )

        @web_app.post("/extract_text_simple")
        def extract_text_simple(body: dict):
            return _extract_simple(
                self.pipeline, body.get("image_data"), body.get("file_name")
            )

        return web_app
