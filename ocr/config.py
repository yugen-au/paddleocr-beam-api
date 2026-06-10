"""Central config (Modal).

Unlike the Beam version, there's no deploy-time/runtime env-forwarding dance:
- Runtime values (VLM_*) are plain constants baked in the code; the container
  reads them directly from this module.
- Deploy-time values (which R2 bucket) are read from env at deploy time and only
  used to build the CloudBucketMount — the container never needs them (it just
  reads/writes the mounted path).
"""
import os

# R2 bucket is mounted here in the container (Modal uses absolute mount paths).
MOUNT_PATH = "/r2-bucket"

# --- Deploy-time: which R2 bucket (prod/staging). Set by deploy.py. -----------
ENVIRONMENTS = {
    "prod":    {"R2_BUCKET": "yugen-assets"},
    "staging": {"R2_BUCKET": "yugen-assets-staging"},
}
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "staging")
R2_BUCKET = os.environ.get("R2_BUCKET", ENVIRONMENTS[DEPLOY_ENV]["R2_BUCKET"])
# R2 S3 endpoint is per-account (same for prod/staging here).
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT", "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com"
)
# modal.Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (the R2 keys).
R2_SECRET_NAME = os.environ.get("R2_SECRET_NAME", "r2-creds")

# --- Runtime constants (the container reads these directly) -------------------
# FastDeploy VLM sidecar (the 0.9B model served as a separate OpenAI-compatible
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

# GPU — Modal lineup (no consumer RTX). Override with MODAL_GPU at deploy time.
GPU = os.environ.get("MODAL_GPU", "L40S")
GPU_SUPPORTS_FA3 = GPU in {"H100", "H200", "B200"}  # A10G/L40S do not
