"""Modal deploy entrypoint.

`modal deploy app.py` (or `modal serve app.py`) discovers the `app` object here.
Importing `ocr.endpoints` registers the OCRService class on the app.

Deploy via the wrapper (sets the R2 bucket per environment):
  python deploy.py staging
  python deploy.py prod
  python deploy.py staging --serve     # ephemeral dev
"""
from ocr.endpoints import OCRService  # noqa: F401  (registers the @app.cls)
from ocr.resources import app

__all__ = ["app", "OCRService"]
