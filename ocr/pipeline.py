"""Pipeline bootstrap. Called once per container from the Modal class's
@modal.enter(): start the FastDeploy sidecar, then build the PaddleOCRVL client
that runs layout/orientation/unwarp in-process and delegates VLM recognition to
the sidecar over HTTP."""
from ocr.config import VLM_BACKEND, VLM_SERVER_URL
from ocr.vlm_server import start_vlm_server


def boot():
    """Start sidecar + return a ready PaddleOCRVL pipeline client."""
    start_vlm_server()

    from paddleocr import PaddleOCRVL

    print("Building PaddleOCRVL pipeline client (FastDeploy server backend)...")
    # pipeline_version unset -> follows the installed package default (v1.6+).
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
        vl_rec_backend=VLM_BACKEND,
        vl_rec_server_url=VLM_SERVER_URL,
    )
    print("Pipeline ready.")
    return pipeline
