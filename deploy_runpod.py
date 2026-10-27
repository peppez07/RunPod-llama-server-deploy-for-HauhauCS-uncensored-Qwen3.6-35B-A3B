#!/usr/bin/env python3
"""Deploy the project's Prism/Bonsai Docker image to RunPod (Python 3.10+)."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from serve_model import DEFAULT_FILE, DEFAULT_REPO, DEFAULT_REVISION

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "ghcr.io/peppez07/bonsai-llama-server-cuda:latest"


def load_env(path=Path(".env")):
    """Simple KEY=value files; environment variables take precedence."""
    values = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                raise ValueError("Expected KEY=value in .env")
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[key.strip()] = value
    return {**values, **os.environ}


def pod_config(env):
    if env.get("BUILD_FROM_SOURCE", "").lower() in ("1", "true", "yes"):
        raise ValueError("BUILD_FROM_SOURCE was removed; build Dockerfile.llama-server first")
    model_repo = env.get("HF_REPO", DEFAULT_REPO)
    pod_env = {
        "HF_TOKEN": env.get("HF_TOKEN", ""),
        "LLAMA_API_KEY": env.get("LLAMA_API_KEY", ""),
        "HF_REPO": model_repo,
        "HF_REVISION": env.get("HF_REVISION") or (DEFAULT_REVISION if model_repo == DEFAULT_REPO else "main"),
        "GGUF_FILE": env.get("GGUF_FILE", DEFAULT_FILE),
        "MODEL_ALIAS": env.get("MODEL_ALIAS", "bonsai-heretic"),
        "CTX_SIZE": env.get("CTX_SIZE", "32768"),
        "PARALLEL": env.get("PARALLEL", "1"),
        "N_GPU_LAYERS": env.get("N_GPU_LAYERS", "99"),
        "REASONING_EFFORT": env.get("REASONING_EFFORT", "medium"),
    }
    config = {
        "name": env.get("POD_NAME", "bonsai-heretic-llamaserver"),
        "imageName": env.get("LLAMA_SERVER_IMAGE", DEFAULT_IMAGE),
        "computeType": "GPU",
        "gpuTypeIds": [x.strip() for x in env.get(
            "GPU_TYPE_IDS", "NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation"
        ).split(",") if x.strip()],
        "gpuCount": 1,
        "containerDiskInGb": int(env.get("CONTAINER_DISK_GB", "150")),
        "volumeInGb": int(env.get("VOLUME_GB", "30")),
        "volumeMountPath": "/workspace",
        "ports": ["8000/http"],
        "env": pod_env,
        "dockerStartCmd": ["python3", "/app/serve_model.py"],
    }
    if env.get("NETWORK_VOLUME_ID"):
        config["networkVolumeId"] = env["NETWORK_VOLUME_ID"]
    if env.get("CONTAINER_REGISTRY_AUTH_ID"):
        config["containerRegistryAuthId"] = env["CONTAINER_REGISTRY_AUTH_ID"]
    return config


def redacted(value):
    if isinstance(value, dict):
        return {k: ("<set>" if v else "") if
                any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
                else redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redacted(v) for v in value]
    return value


def request(env, method, path, body=None):
    key = env.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise ValueError("Set RUNPOD_API_KEY in .env or the environment")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        REST_BASE + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        # Response bodies can echo pod env/credentials; do not print them.
        raise RuntimeError(f"RunPod returned HTTP {exc.code} for {method} {path}") from None


def wait_ready(pod_id, timeout):
    endpoint = f"https://{pod_id}-8000.proxy.runpod.net"
    deadline = time.monotonic() + timeout
    print(f"Waiting for {endpoint}/health", flush=True)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=10) as response:
                if response.status == 200 and json.load(response).get("status") == "ok":
                    print(f"READY: {endpoint}/v1")
                    return 0
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    print("Timed out. Inspect the pod's container logs; the pod is still running.", file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    launch = subs.add_parser("launch")
    launch.add_argument("--dry-run", action="store_true", help="Print redacted config without calling RunPod")
    for name in ("status", "wait", "terminate"):
        command = subs.add_parser(name)
        command.add_argument("pod_id")
        if name == "wait":
            command.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        env = load_env()
        if args.command == "launch":
            config = pod_config(env)
            if args.dry_run:
                print(json.dumps(redacted(config), indent=2))
                return 0
            for name in ("HF_TOKEN", "LLAMA_API_KEY"):
                if not config["env"][name].strip():
                    raise ValueError(f"Set {name} in .env or the environment before launch")
            # One creation request. Avoid automatic retries that can allocate
            # duplicate billable pods when a response is lost.
            result = request(env, "POST", "/pods", config)
            pod_id = result.get("id")
            if not pod_id:
                raise RuntimeError("RunPod returned no pod ID; check the console before retrying")
            print(f"Pod created: {pod_id}")
            print(f"Run: python deploy_runpod.py wait {pod_id}")
            print(f"API: https://{pod_id}-8000.proxy.runpod.net/v1")
        elif args.command == "status":
            print(json.dumps(redacted(request(env, "GET", f"/pods/{args.pod_id}")), indent=2))
        elif args.command == "wait":
            return wait_ready(args.pod_id, args.timeout)
        else:
            request(env, "DELETE", f"/pods/{args.pod_id}")
            print("Pod terminated. Its local volume/cache is deleted.")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

