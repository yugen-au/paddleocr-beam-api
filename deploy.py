#!/usr/bin/env python3
"""Cross-platform deploy script. Run with the system Python (no deps).

Sets the deploy-time environment (non-secret R2 config + resource profile) then
invokes `beam deploy` as a subprocess that inherits it. This keeps deploy-time
config out of the app while avoiding shell-specific (bash vs PowerShell) syntax.

  python deploy.py prod
  python deploy.py staging
  python deploy.py staging --profile latency
  python deploy.py prod --endpoint analyze        # just one endpoint
  python deploy.py staging --dry-run

NOTE: actual R2 credentials are NOT here. They live in Beam's secret manager,
referenced by the *_SECRET *names* below (`beam secret create <name> <value>`).
"""
import argparse
import os
import subprocess
import sys

# Same across all environments. These are the VLM tuning knobs; config.py reads
# them (no defaults there) and they're forwarded into the container at runtime.
SHARED = {
    "VLM_PORT": "8118",
    "VLM_MODEL_NAME": "PaddleOCR-VL-1.6-0.9B",
    "VLM_GPU_MEM_UTIL": "0.6",        # conservative for 24GB; bump for H100/latency
    "VLM_MAX_NUM_SEQS": "256",
    "VLM_MAX_MODEL_LEN": "16384",
    "VLM_BOOT_TIMEOUT": "240",
}

# Per-environment config — only what actually differs between prod and staging.
ENVIRONMENTS = {
    "prod": {
        "BEAM_PROFILE": "cost",
        "R2_BUCKET": "yugen-assets",
        "R2_ENDPOINT": "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com",
        "R2_ACCESS_KEY_SECRET": "BEAM_S3_KEY",
        "R2_SECRET_KEY_SECRET": "BEAM_S3_SECRET",
    },
    # Mirrors prod (same Cloudflare account + creds); only the bucket differs.
    "staging": {
        "BEAM_PROFILE": "cost",
        "R2_BUCKET": "yugen-assets-staging",
        "R2_ENDPOINT": "https://9d7bee7c1c5f0c0206e497f750384ae3.r2.cloudflarestorage.com",
        "R2_ACCESS_KEY_SECRET": "BEAM_S3_KEY",
        "R2_SECRET_KEY_SECRET": "BEAM_S3_SECRET",
    },
}

# Deployable endpoints: short name -> beam target (module:function).
ENDPOINTS = {
    "analyze": "app.py:extract_text_and_analyze",
    "simple": "app.py:extract_text_simple",
}
DEFAULT_ENDPOINTS = ["analyze", "simple"]

# Beam workspace (context) to deploy into. Separate axis from the R2 config above:
# this is WHERE the endpoint runs/bills, not which bucket it uses. Passed via
# `beam -c <ctx>` so deploys don't depend on the globally-selected default context.
BEAM_CONTEXT = "yugen-au"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy PaddleOCR-VL endpoints to Beam.")
    parser.add_argument("environment", choices=sorted(ENVIRONMENTS), help="Target environment")
    parser.add_argument(
        "--endpoint", "-e", action="append", choices=sorted(ENDPOINTS),
        help="Endpoint(s) to deploy. Repeatable. Default: " + ", ".join(DEFAULT_ENDPOINTS),
    )
    parser.add_argument("--profile", choices=["cost", "latency"], help="Override BEAM_PROFILE")
    parser.add_argument("--context", default=BEAM_CONTEXT, help=f"Beam context/workspace (default: {BEAM_CONTEXT})")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without deploying")
    args = parser.parse_args()

    cfg = {**SHARED, **ENVIRONMENTS[args.environment]}
    if args.profile:
        cfg["BEAM_PROFILE"] = args.profile
    cfg["BEAM_DEPLOY_ENV"] = args.environment  # suffixes the deployment name (prod/staging separation)
    targets = args.endpoint or DEFAULT_ENDPOINTS

    if "REPLACE_ME" in cfg["R2_BUCKET"]:
        sys.exit(
            f"R2_BUCKET for '{args.environment}' is unset. "
            "Edit ENVIRONMENTS in deploy.py before deploying."
        )

    env = {**os.environ, **cfg}

    print(f"Context     : {args.context}")
    print(f"Environment : {args.environment}")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print(f"Endpoints   : {', '.join(targets)}")

    if args.dry_run:
        print("\n[dry-run] would run:")
        for t in targets:
            print(f"  beam -c {args.context} deploy {ENDPOINTS[t]}")
        return

    for t in targets:
        ref = ENDPOINTS[t]
        print(f"\n>>> beam -c {args.context} deploy {ref}")
        try:
            result = subprocess.run(["beam", "-c", args.context, "deploy", ref], env=env)
        except FileNotFoundError:
            sys.exit("`beam` CLI not found on PATH. Install/activate it and retry.")
        if result.returncode != 0:
            sys.exit(f"Deploy failed for {ref} (exit {result.returncode})")

    print("\nDone.")


if __name__ == "__main__":
    main()
