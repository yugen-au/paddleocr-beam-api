"""Central config. This module only *reads* env vars — their values live in
`deploy.py` (the single source of truth). There are intentionally NO defaults:
every var in REQUIRED_VARS must be supplied.

deploy.py sets them in the deploy process (so they're present when `beam deploy`
imports this at deploy time) and forwards them into the container via the
endpoints' `env_vars=RUNTIME_ENV` (Beam does NOT propagate the deploy shell's
environment to the runtime container, so without this the container would use
nothing / crash).

Values that never vary (host, backend name, region, mount path) are plain
constants here rather than masquerading as env vars.
"""
import os

# --- Fixed constants (not configurable) --------------------------------------
MOUNT_PATH = "./protocols"           # in-container R2 mount dir
VLM_HOST = "127.0.0.1"               # sidecar is always local
VLM_BACKEND = "fastdeploy-server"    # coupled to the code path
R2_REGION = "auto"                   # R2 is always "auto"

# --- Required env vars (values defined in deploy.py) -------------------------
REQUIRED_VARS = (
    "BEAM_PROFILE",
    "BEAM_DEPLOY_ENV",
    "VLM_PORT",
    "VLM_MODEL_NAME",
    "VLM_GPU_MEM_UTIL",
    "VLM_MAX_NUM_SEQS",
    "VLM_MAX_MODEL_LEN",
    "VLM_BOOT_TIMEOUT",
    "R2_BUCKET",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_SECRET",
    "R2_SECRET_KEY_SECRET",
)
_missing = [v for v in REQUIRED_VARS if v not in os.environ]
if _missing:
    raise RuntimeError(
        "Missing required env vars: " + ", ".join(_missing) + ". "
        "Deploy via `python deploy.py <prod|staging>` (it sets and forwards them)."
    )

# Forwarded verbatim into the container (endpoints pass env_vars=RUNTIME_ENV) so
# the runtime import + boot() see the same values chosen at deploy time.
RUNTIME_ENV = {v: os.environ[v] for v in REQUIRED_VARS}

# --- Typed views -------------------------------------------------------------
# Resource profile (drives GPU/CPU/memory + FA3).
PROFILES = {
    "cost":    {"gpu": "RTX4090", "cpu": 4, "memory": "16Gi"},
    "latency": {"gpu": "H100",    "cpu": 8, "memory": "32Gi"},
}
PROFILE_NAME = os.environ["BEAM_PROFILE"]
PROFILE = PROFILES[PROFILE_NAME]
GPU_SUPPORTS_FA3 = PROFILE["gpu"] in {"H100", "H200", "B200"}

# Environment name (prod/staging) — suffixes the deployment names so each env is a
# distinct Beam deployment in the same workspace.
DEPLOY_ENV = os.environ["BEAM_DEPLOY_ENV"]

# FastDeploy VLM server (sidecar).
VLM_PORT = int(os.environ["VLM_PORT"])
VLM_SERVER_URL = f"http://{VLM_HOST}:{VLM_PORT}/v1"
VLM_MODEL_NAME = os.environ["VLM_MODEL_NAME"]
VLM_GPU_MEM_UTIL = float(os.environ["VLM_GPU_MEM_UTIL"])
VLM_MAX_NUM_SEQS = int(os.environ["VLM_MAX_NUM_SEQS"])
VLM_MAX_MODEL_LEN = int(os.environ["VLM_MAX_MODEL_LEN"])
VLM_BOOT_TIMEOUT = int(os.environ["VLM_BOOT_TIMEOUT"])

# R2 / S3 storage. *_SECRET values are Beam secret *names*, not raw keys.
R2_BUCKET = os.environ["R2_BUCKET"]
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY_SECRET = os.environ["R2_ACCESS_KEY_SECRET"]
R2_SECRET_KEY_SECRET = os.environ["R2_SECRET_KEY_SECRET"]
