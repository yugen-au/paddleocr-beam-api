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

# R2 bucket is mounted here in the container (Modal uses absolute mount paths).
MOUNT_PATH = "/r2-bucket"

# --- Per-environment (varies per deploy; set by deploy.py, staging fallback) ---
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "staging")
R2_BUCKET = os.environ.get("R2_BUCKET", "yugen-assets-staging")
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT", "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com"
)
GPU = os.environ.get("MODAL_GPU", "L40S")

# Per-env Modal app name so prod/staging are separate deployments (don't clobber).
APP_NAME = f"paddleocr-vl-{DEPLOY_ENV}"

# --- Constants (don't vary per deploy) ----------------------------------------
# modal.Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (the R2 keys).
R2_SECRET_NAME = "r2-creds"
GPU_SUPPORTS_FA3 = GPU in {"H100", "H200", "B200"}  # A10G/L40S do not

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
