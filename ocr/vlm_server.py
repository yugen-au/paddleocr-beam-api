"""FastDeploy VLM sidecar lifecycle.

The base image has no fastdeploy, and `install_genai_server_deps` imports paddle
(needs the GPU driver — present at runtime, not at image build), so we install it
here in boot() rather than in the image. Then launch the server and poll /health.
The paddleocr/paddlex CLIs are at /usr/local/bin (on PATH) in the vendor image.
"""
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
_server_proc = None


def _write_backend_config() -> str:
    with open(_BACKEND_CONFIG_PATH, "w") as f:
        f.write(f"gpu-memory-utilization: {VLM_GPU_MEM_UTIL}\n")
        f.write(f"max-num-seqs: {VLM_MAX_NUM_SEQS}\n")
        f.write(f"max-model-len: {VLM_MAX_MODEL_LEN}\n")
    return _BACKEND_CONFIG_PATH


def _install_genai_deps() -> None:
    print("Installing FastDeploy genai server deps (runtime; needs GPU)...")
    subprocess.run(["paddleocr", "install_genai_server_deps", "fastdeploy"], check=True)


def _wait_for_health() -> None:
    url = f"http://{VLM_HOST}:{VLM_PORT}/health"
    deadline = time.monotonic() + VLM_BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if _server_proc is not None and _server_proc.poll() is not None:
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
    raise TimeoutError(f"FastDeploy VLM server not healthy within {VLM_BOOT_TIMEOUT}s")


def start_vlm_server() -> subprocess.Popen:
    """Install genai deps, launch the FastDeploy server, block until healthy."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        return _server_proc

    _install_genai_deps()
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

    print(f"Starting FastDeploy VLM server: {' '.join(cmd)}")
    _server_proc = subprocess.Popen(cmd, env=env)
    _wait_for_health()
    return _server_proc
