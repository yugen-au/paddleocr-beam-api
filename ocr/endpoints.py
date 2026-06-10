"""Modal service. Boots the pipeline once per container (@modal.enter), then
serves the two OCR endpoints as POST routes on a FastAPI app (@modal.asgi_app).
The extraction logic is unchanged from the Beam version — only the wrapper differs.
"""
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import modal

from ocr.config import GPU
from ocr.io import prepare_input_file
from ocr.metrics import calculate_character_metrics
from ocr.pipeline import boot
from ocr.resources import SECRETS, VOLUMES, app, image
from ocr.storage import save_images_to_r2


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]


def _extract_and_analyze(
    pipeline,
    image_data: Optional[str],
    file_name: Optional[str],
    output_format: str,
    include_character_metrics: bool,
    include_layout_analysis: bool,
) -> Dict[str, Any]:
    try:
        input_path = prepare_input_file(image_data, file_name)
        temp_file_created = image_data is not None
        try:
            print(f"Processing document with PaddleOCR-VL: {input_path}")
            output = pipeline.predict(input_path)

            session_id = _session_id()
            results = []
            for page_idx, res in enumerate(output):
                result_data = {
                    "success": True,
                    "page": page_idx + 1,
                    "text_content": res.text if hasattr(res, "text") else "",
                    "structure_info": {},
                }
                if hasattr(res, "json") and res.json:
                    result_data["structure_info"]["json"] = res.json
                if output_format == "markdown" and hasattr(res, "markdown"):
                    result_data["markdown"] = res.markdown
                if include_layout_analysis and hasattr(res, "layout"):
                    result_data["layout_analysis"] = res.layout
                if include_character_metrics:
                    result_data["character_metrics"] = calculate_character_metrics(res)
                results.append(result_data)

            print(f"Saving images to R2 with session ID: {session_id}")
            cleaned_results = save_images_to_r2(results, session_id, file_name)

            return {
                "success": True,
                "results": cleaned_results,
                "total_pages": len(cleaned_results),
                "session_id": session_id,
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
            if temp_file_created and os.path.exists(input_path):
                os.unlink(input_path)
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


def _extract_simple(
    pipeline, image_data: Optional[str], file_name: Optional[str]
) -> Dict[str, Any]:
    try:
        input_path = prepare_input_file(image_data, file_name)
        temp_file_created = image_data is not None
        try:
            print(f"Processing document with PaddleOCR-VL (simple): {input_path}")
            output = pipeline.predict(input_path)

            session_id = _session_id()
            all_text, character_metrics, raw_results = [], [], []
            for page_idx, res in enumerate(output):
                if hasattr(res, "text") and res.text:
                    all_text.append(res.text)
                    character_metrics.append(calculate_character_metrics(res))
                raw_results.append({"page": page_idx + 1, "raw_data": res})

            print(f"Saving images to R2 with session ID: {session_id}")
            cleaned_raw_results = save_images_to_r2(raw_results, session_id, file_name)

            full_text = "\n".join(all_text)
            words = full_text.split()
            if character_metrics:
                avg_metrics = {
                    "average_character_count": sum(m.get("character_count", 0) for m in character_metrics) / len(character_metrics),
                    "average_word_length": sum(m.get("average_word_length", 0) for m in character_metrics) / len(character_metrics),
                    "total_lines": sum(m.get("line_count", 0) for m in character_metrics),
                }
            else:
                avg_metrics = {"note": "No character metrics available"}

            return {
                "success": True,
                "extracted_text": full_text,
                "word_count": len(words),
                "character_count": len(full_text.replace(" ", "")),
                "character_metrics": avg_metrics,
                "session_id": session_id,
                "raw_results": cleaned_raw_results,
                "input_method": "base64" if image_data else "s3_upload",
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "mode": "simple_extraction",
                },
            }
        finally:
            if temp_file_created and os.path.exists(input_path):
                os.unlink(input_path)
    except Exception as e:
        return {"success": False, "error": str(e)}


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
