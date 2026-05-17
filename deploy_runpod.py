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
MMPROJ_FILE = "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"


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


# Build llama.cpp from source on nvidia/cuda base. Caches the build on
# /workspace so subsequent pod boots skip the compile entirely (~5 min once,
# ~30s thereafter). llama-server downloads the GGUF via -hf at startup.
STARTUP_SCRIPT = rf"""
exec > >(tee -a /workspace/startup.log) 2>&1
set -u
echo "=== Hauhau Aggressive llama-server (source build) ==="
date

# sshd for debugging — PUBLIC_KEY auto-injected by RunPod
if [ ! -f /workspace/.ssh-ready ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    openssh-server git build-essential cmake curl libcurl4-openssl-dev
  mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
  [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  sed -i "s/#\?PermitRootLogin .*/PermitRootLogin prohibit-password/" /etc/ssh/sshd_config
  sed -i "s/#\?PasswordAuthentication .*/PasswordAuthentication no/" /etc/ssh/sshd_config
  touch /workspace/.ssh-ready
fi
/usr/sbin/sshd -e &

if [ -z "${{HF_TOKEN:-}}" ]; then
  echo "ERROR: HF_TOKEN not set"
  exec tail -f /dev/null
fi

# Build llama.cpp with CUDA (cached on /workspace)
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  echo "Building llama.cpp from source..."
  cd /workspace
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
  # nproc returns host CPUs (often 128) regardless of the container's vCPU
  # quota, which would over-parallelize and thrash. Read the cgroup quota.
  JOBS=$(awk '{{q=$1; p=$2; if (q=="max") q=p; printf "%d\n", (q+p-1)/p}}' \
    /sys/fs/cgroup/cpu.max 2>/dev/null) || JOBS=8
  echo "Building with -j $JOBS"
  cmake --build build --config Release -j "$JOBS" --target llama-server
else
  echo "llama-server already built — skipping compile."
fi

# Download the GGUF + mmproj via curl. llama-server's -hf path is unreliable
# because its built-in CURL was compiled without OpenSSL detection on the
# old base-image build path (the new GHA-built image fixes this).
MODEL_DIR=/workspace/model
mkdir -p "$MODEL_DIR"
MODEL_FILE="$MODEL_DIR/{GGUF_FILE}"
MMPROJ_PATH="$MODEL_DIR/{MMPROJ_FILE}"
for SPEC in "{GGUF_FILE}" "{MMPROJ_FILE}"; do
  DEST="$MODEL_DIR/$SPEC"
  if [ ! -s "$DEST" ]; then
    echo "Downloading $SPEC..."
    curl -L --fail -H "Authorization: Bearer $HF_TOKEN" \
      -o "$DEST" \
      "https://huggingface.co/{HF_REPO}/resolve/main/$SPEC"
  else
    echo "$SPEC already present ($(du -h "$DEST" | cut -f1))."
  fi
done

echo "=== Starting llama-server ==="
# --jinja: required for HauhauCS chat-template rendering (per model card)
# --mmproj: enables vision input via the multimodal projector
/workspace/llama.cpp/build/bin/llama-server \
  --model "$MODEL_FILE" \
  --mmproj "$MMPROJ_PATH" \
  --alias hauhau-aggressive \
  --host 0.0.0.0 --port 8000 \
  --n-gpu-layers 999 \
  --ctx-size 65536 \
  --parallel 4 \
  --jinja \
  --metrics 2>&1 | tee -a /workspace/llama-server.log
echo "=== llama-server exited with code $? — parking container for SSH debug ==="
exec tail -f /dev/null
"""


POD_CONFIG = {
    "name": "hauhau-aggressive-llamaserver",
    "imageName": "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
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
    "ports": ["8000/http", "22/tcp"],
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
