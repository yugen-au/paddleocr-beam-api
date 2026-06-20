"""Central config (Modal).

Rule: a value is an env var *iff* it varies per deploy (prod/staging, or a flag);
otherwise it's a plain constant. deploy.py is the single source of the per-env
values and injects them into `modal deploy`'s env; config reads them with a
staging fallback so a bare `modal deploy app.py` (no wrapper) still works.

Modal imports this module twice — locally at deploy time, and again in the
container at runtime. resources.py bakes the deploy-resolved env into the image
(.env) so both reads agree (matters for GPU_SUPPORTS_FA3 on H100/H200).
"""
import os

# --- Per-environment (varies per deploy; set by deploy.py, staging fallback) ---
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "staging")
R2_BUCKET = os.environ.get("R2_BUCKET", "yugen-assets-staging")
# Private counterpart in the same R2 account, for PII docs. Naming convention:
# yugen-assets* -> yugen-private-assets*. Selected per-request via bucket_for().
R2_PRIVATE_BUCKET = R2_BUCKET.replace("yugen-assets", "yugen-private-assets")
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT", "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com"
)
GPU = os.environ.get("MODAL_GPU", "L40S")


def bucket_for(private: bool) -> str:
    """R2 bucket for a request: the private (PII) bucket if `private`, else public."""
    return R2_PRIVATE_BUCKET if private else R2_BUCKET

# Single app name across environments. prod/staging isolation comes from the
# Modal *environment* (deploy.py passes `-e main|staging`), not the name; the
# staging environment's web suffix differentiates endpoint URLs. DEPLOY_ENV is
# still baked (drives R2_BUCKET + traceability), set to the environment name.
APP_NAME = "paddleocr-vl"

# --- Constants (don't vary per deploy) ----------------------------------------
# modal.Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (the R2 keys).
R2_SECRET_NAME = "r2-creds"
# R2 key prefix under which all persisted artifacts live, flat by session:
# ocr/<session_id>/{original,result.json,p0001/{raw_result.json,page.md,viz,extracted}}
ARTIFACT_ROOT = "ocr"
GPU_SUPPORTS_FA3 = GPU in {"H100", "H200", "B200"}  # A10G/L40S do not
# FastDeploy itself is baked into the base image (official prebuilt server image),
# CUDA-matched and tested — no install/version/index config needed here anymore.

# FastDeploy VLM sidecar (0.9B model served as a separate OpenAI-compatible
# process; the pipeline delegates VLM recognition to it over HTTP).
VLM_HOST = "127.0.0.1"
VLM_PORT = 8118
VLM_SERVER_URL = f"http://{VLM_HOST}:{VLM_PORT}/v1"
VLM_BACKEND = "fastdeploy-server"
VLM_MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
VLM_GPU_MEM_UTIL = 0.6          # conservative; layout models share the GPU
VLM_MAX_NUM_SEQS = 256
VLM_MAX_MODEL_LEN = 16384
VLM_BOOT_TIMEOUT = 300          # secs to wait for the sidecar /health on cold start
