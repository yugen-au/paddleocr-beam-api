"""Pipeline bootstrap. boot() is the Beam on_start hook: it launches the
FastDeploy sidecar and builds the PaddleOCRVL client that delegates the VLM
stage to it. Runs once per container; the returned pipeline is read from
`context.on_start_value` in the endpoint handlers."""
from ocr.config import VLM_BACKEND, VLM_SERVER_URL
from ocr.vlm_server import start_vlm_server


def boot():
    """Beam on_start: start sidecar VLM server, return ready pipeline client."""
    start_vlm_server()

    from paddleocr import PaddleOCRVL

    print("Building PaddleOCRVL pipeline client (FastDeploy server backend)...")
    # pipeline_version left unset -> follows installed package default (v1.6+).
    # Layout/orientation/unwarp run in-process; VLM recognition is delegated.
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
        vl_rec_backend=VLM_BACKEND,
        vl_rec_server_url=VLM_SERVER_URL,
    )
    print("Pipeline ready.")
    return pipeline
