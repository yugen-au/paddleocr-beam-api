"""Modal infrastructure: App, image, persistent volumes, R2 mount, secret.

The vendor PaddleOCR-VL image runs as ROOT on Modal (validated), using the
image's own Python 3.10 (which has paddle + paddleocr). FastDeploy is NOT in the
base image — it's installed at runtime in boot() (needs the GPU, which is only
attached at runtime, not at build).
"""
import modal

from ocr.config import (
    DEPLOY_ENV,
    GPU,
    MOUNT_PATH,
    R2_BUCKET,
    R2_ENDPOINT,
    R2_SECRET_NAME,
)

app = modal.App("paddleocr-vl")

# Pinned vendor image, run as-is (root, image's python). `pip install -U paddleocr`
# floats to >=3.6.0 for VL-1.6 (pure-python, safe). `add_local_python_source`
# ships our `ocr` package into the container.
image = (
    modal.Image.from_registry(
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl@sha256:e2b525b8fb8ac5711eac667d574dbcf5516a2e6a5437a416357ce64ba1b81a58"
    )
    .run_commands(
        'pip install -U "paddleocr[doc-parser]"',  # >=3.6.0 for VL-1.6
        "pip install fastapi",                       # for @modal.asgi_app (no-op if present)
    )
    # Bake the deploy-resolved config so config.py reads identical values at
    # runtime (Modal re-imports it in the container, where the deploy shell's env
    # is absent). Required for GPU_SUPPORTS_FA3 to reflect the real GPU.
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
