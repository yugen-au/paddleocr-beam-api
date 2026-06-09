"""Beam deploy entrypoint.

Endpoints live in `ocr.endpoints` and are re-exported here so existing
`beam deploy app.py:<fn>` commands keep working after the modular refactor.

Deploy:
  BEAM_PROFILE=cost     beam deploy app.py:extract_text_and_analyze   # cheap $/page (RTX4090)
  BEAM_PROFILE=latency  beam deploy app.py:extract_text_and_analyze   # fast per-request (H100)
  beam deploy app.py:extract_text_simple
"""
from ocr.endpoints import extract_text_and_analyze, extract_text_simple

__all__ = ["extract_text_and_analyze", "extract_text_simple"]


if __name__ == "__main__":
    print("PaddleOCR-VL Beam API (FastDeploy sidecar, modular)")
    print("Deploy:")
    print("  BEAM_PROFILE=cost beam deploy app.py:extract_text_and_analyze")
    print("  BEAM_PROFILE=latency beam deploy app.py:extract_text_and_analyze")
    print("  beam deploy app.py:extract_text_simple")
