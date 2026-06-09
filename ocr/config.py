"""Central config. All deploy-time + runtime knobs live here.

Resource selection (GPU/CPU/memory) is resolved when `beam deploy` imports the
module, so BEAM_PROFILE must be set in the shell running the deploy, NOT as a
runtime/container env var (GPU is provisioned before the container starts).
"""
import os

# R2 uploads bucket mount (see ocr.resources)
MOUNT_PATH = "./protocols"

# --- R2 / S3 storage (deploy-time; set in the shell running `beam deploy`) ----
# Switch prod <-> staging by overriding these. R2 endpoint is per-account (holds
# the account id), so a same-account staging bucket only needs R2_BUCKET changed.
# access/secret values are Beam *secret names*, not the raw keys.
R2_BUCKET = os.environ.get("R2_BUCKET", "protocols")
R2_ENDPOINT = os.environ.get(
    "R2_ENDPOINT",
    "https://50e1f4714be505bee485af31b51492f1.r2.cloudflarestorage.com",
)
R2_REGION = os.environ.get("R2_REGION", "auto")
R2_ACCESS_KEY_SECRET = os.environ.get("R2_ACCESS_KEY_SECRET", "BEAM_S3_KEY")
R2_SECRET_KEY_SECRET = os.environ.get("R2_SECRET_KEY_SECRET", "BEAM_S3_SECRET")

# --- FastDeploy VLM server (sidecar) -----------------------------------------
# The 0.9B vision-language model is served as a separate OpenAI-compatible HTTP
# process; the pipeline (layout/orientation/unwarp) stays in-process and calls it.
VLM_HOST = "127.0.0.1"
VLM_PORT = int(os.environ.get("VLM_PORT", "8118"))
VLM_SERVER_URL = f"http://{VLM_HOST}:{VLM_PORT}/v1"
VLM_BACKEND = "fastdeploy-server"  # confirmed string from official pipeline_config_fastdeploy.yaml

# Server backend needs a concrete model name (cannot auto-"float" like the
# in-process default). Bump this (or set VLM_MODEL_NAME) when adopting v1.7+.
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B")

# Leave VRAM headroom: the in-process layout/orientation models share the GPU
# with the FastDeploy server. 0.6 is conservative for 24GB (RTX4090).
VLM_GPU_MEM_UTIL = float(os.environ.get("VLM_GPU_MEM_UTIL", "0.6"))
VLM_MAX_NUM_SEQS = int(os.environ.get("VLM_MAX_NUM_SEQS", "256"))
VLM_MAX_MODEL_LEN = int(os.environ.get("VLM_MAX_MODEL_LEN", "16384"))

# Seconds to wait for the server to become healthy on cold start.
VLM_BOOT_TIMEOUT = int(os.environ.get("VLM_BOOT_TIMEOUT", "240"))

# --- Deploy-time resource profile --------------------------------------------
#   BEAM_PROFILE=cost     beam deploy app.py:extract_text_and_analyze   # cheap $/page
#   BEAM_PROFILE=latency  beam deploy app.py:extract_text_and_analyze   # fast per-request
PROFILES = {
    "cost":    {"gpu": "RTX4090", "cpu": 4, "memory": "16Gi"},
    "latency": {"gpu": "H100",    "cpu": 8, "memory": "32Gi"},
}
PROFILE_NAME = os.environ.get("BEAM_PROFILE", "cost")
PROFILE = PROFILES[PROFILE_NAME]

# Hopper/Blackwell support FlashAttention 3; Ada (RTX4090) does not.
GPU_SUPPORTS_FA3 = PROFILE["gpu"] in {"H100", "H200", "B200"}
