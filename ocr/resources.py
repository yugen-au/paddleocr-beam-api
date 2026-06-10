"""Beam infrastructure objects: image, persistent volumes, R2 bucket."""
import beam

from ocr.config import (
    MOUNT_PATH,
    R2_ACCESS_KEY_SECRET,
    R2_BUCKET,
    R2_ENDPOINT,
    R2_REGION,
    R2_SECRET_KEY_SECRET,
)

# Official PaddleOCR-VL image, PINNED BY DIGEST (not :latest). Floating the base
# silently broke the build once the vendor image moved to a non-root user; pin it.
# paddleocr is still floated below (with the cache-bust) for model-version updates.
image = (
    beam.Image(
        base_image="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl@sha256:e2b525b8fb8ac5711eac667d574dbcf5516a2e6a5437a416357ce64ba1b81a58"
    )
    .add_commands([
        # cache-bust: bump to force a rebuild + refresh the floating paddleocr below
        'echo "build: 2026-06-09"',
        # floor-pin paddlepaddle (CUDA 12.6 index); float paddleocr to newest
        'pip install -U "paddlepaddle-gpu>=3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/',
        'pip install -U "paddleocr[doc-parser]"',
        # FastDeploy accelerated VLM backend (version chosen to match paddleocr)
        'paddleocr install_genai_server_deps fastdeploy',
    ])
)
# Use the base image's own Python (3.10, already has paddle) rather than letting Beam
# bootstrap its own. This skips Beam's apt-based Python install step, which fails under
# the base image's non-root user. Safe here: our installs are shell BuildSteps
# (add_commands), not pip BuildSteps, so Beam skips the Python setup entirely.
image.ignore_python = True

# Persistent caches so cold starts don't re-download models.
# .paddlex: layout/orientation/unwarp models. HF cache: the VLM weights pulled by genai_server.
model_cache = beam.Volume(
    name="paddleocr-models",
    mount_path="/home/paddleocr/.paddlex/official_models",
)
hf_cache = beam.Volume(
    name="paddleocr-hf-cache",
    mount_path="/home/paddleocr/.cache/huggingface",
)

# Cloudflare R2 (S3-compatible). access_key/secret_key are Beam secret *names*.
# All values env-driven (ocr.config) so prod <-> staging is a deploy-time switch.
uploads_bucket = beam.CloudBucket(
    name=R2_BUCKET,
    mount_path=MOUNT_PATH,
    config=beam.CloudBucketConfig(
        access_key=R2_ACCESS_KEY_SECRET,
        secret_key=R2_SECRET_KEY_SECRET,
        endpoint=R2_ENDPOINT,
        region=R2_REGION,
    ),
)

VOLUMES = [model_cache, hf_cache, uploads_bucket]
