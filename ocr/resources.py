"""Modal infrastructure: App, image, persistent volumes, R2 mount, secret.

Base = the OFFICIAL prebuilt PaddleOCR-VL genai server image (fastdeploy-cuda-12.6
2.3.0 + paddleocr/paddlex baked in). It has the FastDeploy serving stack AND the
pipeline, all maintainer-tested and CUDA-matched — so we do ZERO installs. This
replaced self-installing fastdeploy-gpu from the arch-specific index, whose wheel
was CUDA-mismatched and crashed the serving worker at KV-cache init on L40S.
Runs as user `paddleocr` (uid 1000), image's own python 3.10.
"""
import modal

from ocr.config import (
    APP_NAME,
    DEPLOY_ENV,
    GPU,
    MOUNT_PATH,
    R2_BUCKET,
    R2_ENDPOINT,
    R2_SECRET_NAME,
)

app = modal.App(APP_NAME)  # paddleocr-vl-{prod,staging} -> separate deployments

# Official VLM-server image (fastdeploy backend, nvidia-gpu). `add_python=None`:
# use the image's own python (has paddle/paddleocr/fastdeploy). No installs —
# `add_local_python_source` just ships our `ocr` package in. `.env` bakes the
# deploy-resolved config so config.py reads identical values at runtime (Modal
# re-imports it in the container, where the deploy shell's env is absent).
image = (
    modal.Image.from_registry(
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-nvidia-gpu",
        add_python=None,
    )
    # The server image has base paddleocr/paddlex but NOT the pipeline extras
    # (layout/orientation/unwarp). Add the doc-parser extra via uv, unpinned — let
    # it resolve whatever it wants (pinning to <3.5 gave a paddlex whose model
    # registry didn't know PaddleOCR-VL-1.6-0.9B -> "Unknown model").
    # unsafe-best-match: consider all (trusted) indexes, as elsewhere.
    .uv_pip_install(
        "paddleocr[doc-parser]",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({
        "MODAL_GPU": GPU,
        "DEPLOY_ENV": DEPLOY_ENV,
        "R2_BUCKET": R2_BUCKET,
        "R2_ENDPOINT": R2_ENDPOINT,
    })
    .add_local_python_source("ocr")
)

# Persistent caches so cold starts don't re-download models.
model_cache = modal.Volume.from_name("paddleocr-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("paddleocr-hf-cache", create_if_missing=True)

# R2 (S3-compatible). Secret must hold AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
r2_secret = modal.Secret.from_name(R2_SECRET_NAME)
r2_mount = modal.CloudBucketMount(
    bucket_name=R2_BUCKET,
    bucket_endpoint_url=R2_ENDPOINT,  # required for R2
    secret=r2_secret,
)

# Mounts attached to the service (path -> volume/mount).
VOLUMES = {
    "/home/paddleocr/.paddlex/official_models": model_cache,
    "/home/paddleocr/.cache/huggingface": hf_cache,
    MOUNT_PATH: r2_mount,
}
SECRETS = [r2_secret]
