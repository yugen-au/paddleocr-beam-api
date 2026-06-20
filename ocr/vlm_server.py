"""FastDeploy VLM sidecar lifecycle.

FastDeploy is baked into the image at build time (see resources.py), so boot()
only launches the server and polls /health — no runtime install. The
paddleocr/paddlex CLIs are at /usr/local/bin (on PATH) in the vendor image.
"""
import glob
import os
import subprocess
import time
import urllib.error
import urllib.request

from ocr.config import (
    GPU_SUPPORTS_FA3,
    VLM_BOOT_TIMEOUT,
    VLM_GPU_MEM_UTIL,
    VLM_HOST,
    VLM_MAX_MODEL_LEN,
    VLM_MAX_NUM_SEQS,
    VLM_MODEL_NAME,
    VLM_PORT,
)

_BACKEND_CONFIG_PATH = "/tmp/vlm_server_config.yaml"
_SERVER_CWD = "/tmp/vlm"  # fastdeploy writes log/workerlog.* relative to cwd
_server_proc = None


def _write_backend_config() -> str:
    with open(_BACKEND_CONFIG_PATH, "w") as f:
        f.write(f"gpu-memory-utilization: {VLM_GPU_MEM_UTIL}\n")
        f.write(f"max-num-seqs: {VLM_MAX_NUM_SEQS}\n")
        f.write(f"max-model-len: {VLM_MAX_MODEL_LEN}\n")
    return _BACKEND_CONFIG_PATH


def _dump_worker_logs() -> None:
    """Surface fastdeploy worker logs (the real error when the engine fails to
    launch worker processes) — they don't go to the parent's stdout."""
    paths = sorted(glob.glob(os.path.join(_SERVER_CWD, "log", "workerlog*")))
    if not paths:
        print(f"[vlm] no worker logs under {_SERVER_CWD}/log")
        return
    for p in paths:
        try:
            with open(p, errors="replace") as f:
                tail = f.read()[-4000:]
            print(f"\n===== {p} (tail) =====\n{tail}")
        except OSError as e:
            print(f"[vlm] could not read {p}: {e}")


def _wait_for_health() -> None:
    url = f"http://{VLM_HOST}:{VLM_PORT}/health"
    deadline = time.monotonic() + VLM_BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if _server_proc is not None and _server_proc.poll() is not None:
            _dump_worker_logs()
            raise RuntimeError(
                f"VLM server exited during startup (code {_server_proc.returncode})"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print("FastDeploy VLM server is healthy.")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    _dump_worker_logs()
    raise TimeoutError(f"FastDeploy VLM server not healthy within {VLM_BOOT_TIMEOUT}s")


def start_vlm_server() -> subprocess.Popen:
    """Launch the FastDeploy server (deps baked in image), block until healthy."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        return _server_proc

    cfg = _write_backend_config()
    cmd = [
        "paddleocr", "genai_server",
        "--model_name", VLM_MODEL_NAME,
        "--host", VLM_HOST,
        "--port", str(VLM_PORT),
        "--backend", "fastdeploy",
        "--backend_config", cfg,
    ]
    env = os.environ.copy()
    if GPU_SUPPORTS_FA3:
        env["FLAGS_flash_attn_version"] = "3"  # Hopper/Blackwell only

    os.makedirs(_SERVER_CWD, exist_ok=True)
    print(f"Starting FastDeploy VLM server: {' '.join(cmd)}")
    _server_proc = subprocess.Popen(cmd, env=env, cwd=_SERVER_CWD)
    _wait_for_health()
    return _server_proc
