#!/usr/bin/env python3
"""Spin up a RunPod A6000 pod serving HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
via the official llama.cpp server image (ghcr.io/ggml-org/llama.cpp:server-cuda).

The image already contains a CUDA-built llama-server, so cold boot is just:
image pull + model download + model load (~3-4 min on first run, ~1 min after).

Usage:
    # Reads RUNPOD_API_KEY and HF_TOKEN from .env
    python deploy_runpod.py launch
    python deploy_runpod.py status <pod_id>
    python deploy_runpod.py wait <pod_id>
    python deploy_runpod.py terminate <pod_id>

Default quant is Q6_K_P (~31 GB). Override with QUANT env var:
    QUANT=Q4_K_M python deploy_runpod.py launch
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_env():
    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


ENV = load_env()
RUNPOD_API_KEY = ENV.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
HF_TOKEN = ENV.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
QUANT = os.environ.get("QUANT", "Q6_K_P")

if not RUNPOD_API_KEY:
    print("ERROR: RUNPOD_API_KEY not in .env or env", file=sys.stderr)
    sys.exit(1)

REST_BASE = "https://rest.runpod.io/v1"

HF_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
GGUF_FILE = f"Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-{QUANT}.gguf"


def _request(method: str, url: str, body: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {method} {url}", file=sys.stderr)
        try:
            err = json.loads(e.read())
            print(json.dumps(err, indent=2), file=sys.stderr)
        except Exception:
            print(e.read().decode(errors="replace"), file=sys.stderr)
        raise


# llama-server's -hf flag pulls the GGUF straight from HuggingFace (the binary
# is built with CURL support). LLAMA_CACHE pins downloads to the volume so a
# pod restart doesn't re-download.
STARTUP_SCRIPT = rf"""
set -e
exec > >(tee -a /workspace/startup.log) 2>&1
echo "=== Hauhau Aggressive llama-server (prebuilt image) ==="
date

if [ -z "${{HF_TOKEN:-}}" ]; then
  echo "ERROR: HF_TOKEN not set"
  exec tail -f /dev/null
fi

export LLAMA_CACHE=/workspace/llama-cache
mkdir -p "$LLAMA_CACHE"

echo "=== Starting llama-server ==="
exec llama-server \
  -hf {HF_REPO}:{GGUF_FILE} \
  --hf-token "$HF_TOKEN" \
  --alias hauhau-aggressive \
  --host 0.0.0.0 --port 8000 \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --parallel 4 \
  --metrics
"""


POD_CONFIG = {
    "name": "hauhau-aggressive-llamaserver",
    "imageName": "ghcr.io/ggml-org/llama.cpp:server-cuda",
    # 48GB cards in priority order — inventory bounces, so list multiples.
    # A100 80GB at the end as a more-expensive but usually-available fallback.
    "gpuTypeIds": [
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA L40S",
        "NVIDIA RTX A6000",
        "NVIDIA L40",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
    ],
    "gpuCount": 1,
    "containerDiskInGb": 20,
    # 50 GB volume: 31 GB model + ~10 GB image extracted + headroom.
    # Larger volumes restrict which machines can be allocated.
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "ports": ["8000/http"],
    "env": {
        "HF_TOKEN": HF_TOKEN or "",
    },
    "dockerStartCmd": ["bash", "-c", STARTUP_SCRIPT],
}


def cmd_launch():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)
    print(f"Quant: {QUANT}  ->  {GGUF_FILE}")
    print(f"Image: {POD_CONFIG['imageName']}")
    print(f"Creating pod '{POD_CONFIG['name']}':")
    masked = {**POD_CONFIG, "env": {k: "<set>" if v else "" for k, v in POD_CONFIG["env"].items()}}
    masked["dockerStartCmd"] = ["bash", "-c", "<startup script>"]
    print(json.dumps(masked, indent=2))
    print()
    # RunPod inventory churns second-by-second. Two failure modes:
    #   1. 500 "no instances currently available" on POST.
    #   2. POST returns a pod_id but the pod sits unallocated (machine: {}).
    # Both need retry. We accept a pod only if a machine attaches within 75s.
    pod_id = None
    for attempt in range(40):
        try:
            result = _request("POST", f"{REST_BASE}/pods", POD_CONFIG)
        except urllib.error.HTTPError as e:
            if e.code != 500:
                raise
            print(f"[attempt {attempt+1}] no inventory; retry in 10s", flush=True)
            time.sleep(10)
            continue
        candidate = result.get("id")
        print(f"[attempt {attempt+1}] pod {candidate} created; waiting for machine...", flush=True)
        for _ in range(15):  # 15 * 5s = 75s
            time.sleep(5)
            status = _request("GET", f"{REST_BASE}/pods/{candidate}")
            # Use machineId — RunPod sets this at allocation but often leaves
            # the embedded `machine` object empty in REST responses.
            if status.get("machineId"):
                pod_id = candidate
                print(f"  -> allocated on machine {status['machineId']} (cost ${status.get('costPerHr')}/hr)", flush=True)
                break
        if pod_id:
            break
        print(f"  -> never allocated; terminating and retrying", flush=True)
        try:
            _request("DELETE", f"{REST_BASE}/pods/{candidate}")
        except Exception:
            pass
        time.sleep(5)
    if not pod_id:
        print("Gave up after 40 attempts.", file=sys.stderr)
        sys.exit(1)
    if pod_id:
        print(f"\nPod created: {pod_id}")
        print(f"Watch with: python deploy_runpod.py wait {pod_id}")
        print(f"Endpoint will be: https://{pod_id}-8000.proxy.runpod.net")
    return result


def cmd_status(pod_id: str):
    result = _request("GET", f"{REST_BASE}/pods/{pod_id}")
    print(json.dumps(result, indent=2))
    runtime = result.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if p.get("privatePort") == 8000 and p.get("type") == "http":
            endpoint = f"https://{pod_id}-{p['privatePort']}.proxy.runpod.net"
            print(f"\n8000/http -> {endpoint}")
            break
    return result


def cmd_terminate(pod_id: str):
    print(f"Terminating pod {pod_id}...")
    _request("DELETE", f"{REST_BASE}/pods/{pod_id}")
    print("OK")


def cmd_wait_ready(pod_id: str, max_wait_s: int = 900):
    endpoint = f"https://{pod_id}-8000.proxy.runpod.net"
    print(f"Polling {endpoint}/v1/models (up to {max_wait_s}s)...")
    start = time.time()
    while time.time() - start < max_wait_s:
        elapsed = int(time.time() - start)
        try:
            req = urllib.request.Request(f"{endpoint}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print(f"\n[+{elapsed}s] READY: {endpoint}")
                    print(json.loads(resp.read()))
                    return endpoint
        except Exception as e:
            print(f"[+{elapsed}s] not ready: {type(e).__name__}", end="\r")
        time.sleep(15)
    print(f"\nTimed out after {max_wait_s}s")
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "launch":
        cmd_launch()
    elif cmd == "status":
        cmd_status(sys.argv[2])
    elif cmd == "wait":
        cmd_wait_ready(sys.argv[2])
    elif cmd == "terminate":
        cmd_terminate(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
