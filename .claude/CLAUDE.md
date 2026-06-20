# PaddleOCR-VL Beam API

## Overview
GPU-accelerated OCR API using PaddleOCR-VL, deployed on Beam.cloud. Extracts text + analyzes document structure (orientation, unwarping, layout). Two endpoints: full analysis (`extract_text_and_analyze`) and fast simple extraction (`extract_text_simple`). Dual input: base64 or Cloudflare R2 file upload. Extracted images saved back to R2, replaced with URL refs.

## Tech Stack
- Python 3.11
- Beam.cloud (serverless GPU deploy/serve)
- PaddleOCR-VL on PaddlePaddle-GPU 3.2.1 (CUDA 12.6)
- Cloudflare R2 (S3-compatible object storage via `beam.CloudBucket`)
- GPU: RTX 4090

## Architecture
- Modular `ocr/` package; thin `app.py` re-exports endpoints so `beam deploy app.py:<fn>` still works.
  - `ocr/config.py`: PROFILE (deploy-time GPU/CPU/mem), VLM server config, constants
  - `ocr/resources.py`: image (FastDeploy deps), volumes, R2 bucket, `VOLUMES`
  - `ocr/vlm_server.py`: FastDeploy sidecar subprocess launch + `/health` poll
  - `ocr/pipeline.py`: `boot()` (Beam `on_start`) — starts sidecar, builds PaddleOCRVL client
  - `ocr/storage.py` (R2 image save), `ocr/io.py` (input prep), `ocr/metrics.py` (char metrics)
  - `ocr/endpoints.py`: two `@beam.endpoint` fns; pipeline read from `context.on_start_value`
- **Inference = FastDeploy sidecar (Option A).** VLM (0.9B) served as separate OpenAI-compatible
  process on `127.0.0.1:8118`; pipeline runs layout/orientation/unwarp in-process, delegates VLM
  recognition over HTTP (`vl_rec_backend="fastdeploy-server"`). Started once per container in `boot()`.
- Base image: official PaddleOCR-VL image; paddleocr/paddlepaddle floated (`-U`), `:latest` tag.
- Model caches (persistent volumes): `.paddlex/official_models` + HF cache `~/.cache/huggingface`
  (HF cache is required so the genai server doesn't re-download VLM weights every cold start).
- Local env via uv: `uv sync` (dev: beam-client + paddleocr stubs + pytest/ruff) or `uv sync --no-dev` (deploy-only). Source of truth: `pyproject.toml` + `uv.lock`. venv is `.venv` (beam-env retired).
- Deploy: `uv run python deploy.py <main|staging>` (`--serve` ephemeral dev, `--gpu` override, `--dry-run`).
- prod/staging = Modal **environments** (`main` + `staging`) in the one `yugen-au` workspace — selected by `-e`, single app name `paddleocr-vl`. `main` is active, so deploy.py always passes `-e`. Same R2 creds; only the bucket differs per env.
- `deploy.py` owns per-env values (`ENVIRONMENTS`: DEPLOY_ENV + R2_BUCKET); `resources.py` `.env()` bakes them into the image so `config.py` reads the same values in-container. `config.py` env reads have a staging fallback (bare import works for tests).
- GPU via `MODAL_GPU` (default L40S). Cold-start: `@app.cls(scaledown_window=300)`.
- R2 access is all boto3 (`ocr/artifacts.py`), no CloudBucketMount (per-container + can't set Content-Type). Bucket chosen per-request by `bucket_for(private)`: public (`yugen-assets*`) vs private PII (`yugen-private-assets*`). Artifacts under `ocr/<session_id>/`.
- Secrets via Modal secret `r2-creds` (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
- Both web endpoints are private (`@modal.asgi_app(requires_proxy_auth=True)`): callers must send `Modal-Key`/`Modal-Secret` proxy-auth headers; Modal's edge rejects others before the container. Tokens managed in the Modal dashboard.

## Open / verify-on-deploy
- VLM server `/health` path (assumed) and exact GPU strings beyond H100/RTX4090
- Whether base image already bundles FastDeploy (may skip `install_genai_server_deps`)
- Beam keep-warm param for cold-start mitigation (spiky `latency` profile)
- FastDeploy server model is set by `VLM_MODEL_NAME` (can't auto-float like in-process backend)
- Option B (dedicated always-warm VLM server Pod) deferred — better for high-volume business

## MCP Servers
### Base (always included)
- Core shared tools (fetch, context7, serena, graphiti, taskmaster, sequential-thinking)
- Configured in global `~/.claude.json`. No project `.mcp.json` (base-only).

## Development Notes
- Deploy: `beam deploy app.py:extract_text_and_analyze` (or `:extract_text_simple`)
- Local serve: `beam serve app.py:extract_text_and_analyze`
- Full-analysis endpoint timeout: 600s. Simple endpoint: default.
- R2 bucket/endpoint env-driven via deploy.py: prod `yugen-assets`, staging `yugen-assets-staging` (same Cloudflare account + creds, differ by bucket).
- No requirements.txt — deps declared in `beam.Image` build commands.

## Testing
- Command: `uv run pytest`
- Runner: pytest
- Note: `ocr.config` requires env vars (no defaults) — tests must set them (conftest fixture).
- Note: no test suite yet. `test_beam.py` is a Beam sandbox connectivity check, not pytest.

## Graphiti Group ID
- group_id: `paddleocr-beam-api` (codebase memory storage)
