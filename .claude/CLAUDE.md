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
- Single module `app.py`. Two `@beam.endpoint` functions share helpers.
- Base image: official PaddleOCR-VL Docker image + paddlepaddle-gpu pip install.
- Model caching: persistent `beam.Volume` mounted at `/home/paddleocr/.paddlex/official_models`.
- `pipeline` global lazy-initialized once per container (`initialize_pipeline`).
- R2 bucket mounted at `./protocols`; uploads read from mount, output images written to `images/<name>_<session>/`.
- Recursive traversal (`save_images_to_r2`) finds PIL Images in result tree, persists to R2, swaps for URL dicts.
- Secrets via Beam env: `BEAM_S3_KEY`, `BEAM_S3_SECRET`.

## MCP Servers
### Base (always included)
- Core shared tools (fetch, context7, serena, graphiti, taskmaster, sequential-thinking)
- Configured in global `~/.claude.json`. No project `.mcp.json` (base-only).

## Development Notes
- Deploy: `beam deploy app.py:extract_text_and_analyze` (or `:extract_text_simple`)
- Local serve: `beam serve app.py:extract_text_and_analyze`
- Full-analysis endpoint timeout: 600s. Simple endpoint: default.
- R2 bucket name `protocols` must match actual bucket; endpoint `r2.cloudflarestorage.com`.
- No requirements.txt — deps declared in `beam.Image` build commands.

## Testing
- Command: `pytest`
- Runner: pytest
- Note: no test suite yet. `test_beam.py` is a Beam sandbox connectivity check, not pytest.

## Graphiti Group ID
- group_id: `paddleocr-beam-api` (codebase memory storage)
