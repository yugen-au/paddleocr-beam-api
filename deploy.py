#!/usr/bin/env python3
"""Cross-platform deploy wrapper for Modal. Sets the per-environment config
(R2 bucket + GPU) then runs `modal deploy app.py -e <env>`. Run with `uv run`.

  uv run python deploy.py staging
  uv run python deploy.py main                    # production
  uv run python deploy.py staging --serve         # ephemeral dev (modal serve)
  uv run python deploy.py main --gpu A10G
  uv run python deploy.py staging --dry-run

prod/staging isolation = Modal *environments* (main + staging) in the one
`yugen-au` workspace, selected with `-e`. The app keeps a single name
(`paddleocr-vl`); the staging environment's web suffix differentiates URLs.
`-e` is ALWAYS passed: `main` is the active env, so a bare deploy would silently
hit production.

R2 credentials are NOT here — they live in the Modal secret `r2-creds`
(`modal secret create r2-creds AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...`).
"""
import argparse
import os
import subprocess
import sys

_R2_ENDPOINT = "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com"
# Keyed by Modal environment name. DEPLOY_ENV is baked to match (traceability).
ENVIRONMENTS = {
    "main":    {"DEPLOY_ENV": "main",    "R2_BUCKET": "yugen-assets",         "R2_ENDPOINT": _R2_ENDPOINT},
    "staging": {"DEPLOY_ENV": "staging", "R2_BUCKET": "yugen-assets-staging", "R2_ENDPOINT": _R2_ENDPOINT},
}


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy PaddleOCR-VL to Modal.")
    p.add_argument("environment", choices=sorted(ENVIRONMENTS))
    p.add_argument("--gpu", help="Override MODAL_GPU (A10G/L40S/A100/H100/H200)")
    p.add_argument("--serve", action="store_true", help="modal serve (ephemeral dev) instead of deploy")
    p.add_argument("--dry-run", action="store_true", help="Print actions without running")
    args = p.parse_args()

    cfg = dict(ENVIRONMENTS[args.environment])
    if args.gpu:
        cfg["MODAL_GPU"] = args.gpu
    env = {**os.environ, **cfg, "PYTHONUTF8": "1"}  # PYTHONUTF8 avoids cp1252 crash on Windows

    verb = "serve" if args.serve else "deploy"
    cmd = ["modal", verb, "app.py", "-e", args.environment]
    print(f"Environment : {args.environment}  (modal -e {args.environment})")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print(f"Action      : {' '.join(cmd)}")

    if args.dry_run:
        return

    try:
        result = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        sys.exit("`modal` CLI not found. Run `uv sync` then `uv run python deploy.py ...`.")
    if result.returncode != 0:
        sys.exit(f"modal {verb} failed (exit {result.returncode})")


if __name__ == "__main__":
    main()
