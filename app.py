"""Modal deploy entrypoint.

`modal deploy app.py` (or `modal serve app.py`) discovers the `app` object here.
Importing `ocr.endpoints` registers the service classes on the app:
  OCRService     (GPU)  -> /extract_text_and_analyze, /extract_text_simple
  SectionService (CPU)  -> /crop_and_section

Deploy via the wrapper (sets the R2 bucket per environment):
  python deploy.py staging
  python deploy.py prod
  python deploy.py staging --serve     # ephemeral dev
"""
from ocr.endpoints import OCRService, SectionService  # noqa: F401  (register @app.cls)
from ocr.resources import app

__all__ = ["app", "OCRService", "SectionService"]
