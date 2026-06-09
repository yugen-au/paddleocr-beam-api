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
- Deploy via `python deploy.py <prod|staging>` (cross-platform; sets env + runs `beam deploy`).
  - `deploy.py` is the single source of env-var VALUES: `SHARED` (VLM knobs) + `ENVIRONMENTS` (per-env: profile + R2).
  - `config.py` has NO defaults — reads `os.environ[...]` required; raises a pointed error if unset. Not importable standalone (tests must set env). Raw `beam deploy app.py:...` no longer works.
  - Beam does NOT propagate the deploy shell env to the container, so `config.RUNTIME_ENV` is forwarded via `@endpoint(env_vars=RUNTIME_ENV)` — required for `VLM_*` to actually take effect at runtime (`boot()` runs in-container).
- Resource profiles via `BEAM_PROFILE` (cost=RTX4090 / latency=H100), resolved at deploy time.
- R2 bucket mounted at `MOUNT_PATH` (`./protocols`, a local dir); uploads read from mount, output images -> `images/<name>_<session>/`.
- Secrets via Beam secret names: `BEAM_S3_KEY`, `BEAM_S3_SECRET`.

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
- Command: `pytest`
- Runner: pytest
- Note: no test suite yet. `test_beam.py` is a Beam sandbox connectivity check, not pytest.

## Graphiti Group ID
- group_id: `paddleocr-beam-api` (codebase memory storage)
