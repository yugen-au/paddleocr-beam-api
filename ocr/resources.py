"""Beam infrastructure objects: image, persistent volumes, R2 bucket."""
import beam

from ocr.config import MOUNT_PATH

# Official PaddleOCR-VL image; float paddleocr/paddlepaddle + add FastDeploy backend.
image = (
    beam.Image(
        base_image="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest"
    )
    .add_commands([
        # cache-bust: bump this string to force Beam to rebuild and pull latest versions
        'echo "build: 2026-06-09"',
        # float: floor-pin paddlepaddle (CUDA 12.6 index), float paddleocr to newest
        'pip install -U "paddlepaddle-gpu>=3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/',
        'pip install -U "paddleocr[doc-parser]"',
        # FastDeploy accelerated VLM backend (version chosen to match paddleocr)
        'paddleocr install_genai_server_deps fastdeploy',
    ])
)

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
uploads_bucket = beam.CloudBucket(
    name="protocols",  # must match the actual R2 bucket name
    mount_path=MOUNT_PATH,
    config=beam.CloudBucketConfig(
        access_key="BEAM_S3_KEY",
        secret_key="BEAM_S3_SECRET",
        endpoint="https://50e1f4714be505bee485af31b51492f1.r2.cloudflarestorage.com",
        region="auto",
    ),
)

VOLUMES = [model_cache, hf_cache, uploads_bucket]
