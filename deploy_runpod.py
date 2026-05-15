#!/usr/bin/env python3
"""Spin up a RunPod A6000 pod serving HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
via llama.cpp's llama-server (OpenAI-compatible API on port 8000).

Usage:
    # Reads RUNPOD_API_KEY and HF_TOKEN from .env
    python deploy_runpod.py launch
    python deploy_runpod.py status <pod_id>
    python deploy_runpod.py logs <pod_id>
    python deploy_runpod.py wait <pod_id>
    python deploy_runpod.py terminate <pod_id>

Default quant is Q6_K_P (~31 GB). Override with QUANT env var, e.g.
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
GQL_URL = "https://api.runpod.io/graphql"

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
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {method} {url}", file=sys.stderr)
        try:
            err = json.loads(e.read())
            print(json.dumps(err, indent=2), file=sys.stderr)
        except Exception:
            print(e.read().decode(errors="replace"), file=sys.stderr)
        raise


# Container startup: install llama.cpp, download the GGUF, exec llama-server.
# All state lands on /workspace so pod restarts don't re-download.
STARTUP_SCRIPT = rf"""
exec > >(tee -a /workspace/startup.log) 2>&1
echo "=== Hauhau Aggressive llama-server deployment ==="
date

# sshd for debugging even if llama-server fails
if [ ! -f /workspace/.ssh-ready ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    openssh-server git build-essential cmake curl python3 python3-pip libcurl4-openssl-dev
  mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh
  [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  sed -i "s/#\?PermitRootLogin .*/PermitRootLogin prohibit-password/" /etc/ssh/sshd_config
  sed -i "s/#\?PasswordAuthentication .*/PasswordAuthentication no/" /etc/ssh/sshd_config
  touch /workspace/.ssh-ready
fi
/usr/sbin/sshd -e

if [ -z "${{HF_TOKEN:-}}" ]; then
  echo "ERROR: HF_TOKEN not set"
  exec tail -f /dev/null
fi

# Build llama.cpp with CUDA support (cached on volume)
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  echo "Building llama.cpp with CUDA..."
  cd /workspace
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
  cmake --build build --config Release -j --target llama-server
else
  echo "llama-server already built."
fi

# Download the GGUF (single file)
MODEL_DIR=/workspace/model
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/{GGUF_FILE}" ]; then
  echo "Downloading {GGUF_FILE} from {HF_REPO}..."
  pip install --quiet --upgrade huggingface_hub
  hf download {HF_REPO} {GGUF_FILE} \
    --local-dir "$MODEL_DIR" --token "$HF_TOKEN" \
    || huggingface-cli download {HF_REPO} {GGUF_FILE} \
       --local-dir "$MODEL_DIR" --token "$HF_TOKEN"
else
  echo "Model present, skipping download."
fi

echo "=== Starting llama-server ==="
exec /workspace/llama.cpp/build/bin/llama-server \
  --model "$MODEL_DIR/{GGUF_FILE}" \
  --alias hauhau-aggressive \
  --host 0.0.0.0 --port 8000 \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --parallel 4 \
  --threads 8 \
  --metrics \
  || {{
    echo "llama-server failed — parking container for SSH debug"
    exec tail -f /dev/null
  }}
"""


POD_CONFIG = {
    "name": "hauhau-aggressive-llamaserver",
    "imageName": "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
    "gpuTypeIds": ["NVIDIA RTX A6000"],
    "gpuCount": 1,
    "containerDiskInGb": 40,
    "volumeInGb": 80,
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
    print(f"Creating pod '{POD_CONFIG['name']}':")
    masked = {**POD_CONFIG, "env": {k: "<set>" if v else "" for k, v in POD_CONFIG["env"].items()}}
    masked["dockerStartCmd"] = ["bash", "-c", "<startup script>"]
    print(json.dumps(masked, indent=2))
    print()
    result = _request("POST", f"{REST_BASE}/pods", POD_CONFIG)
    print(json.dumps(result, indent=2))
    pod_id = result.get("id")
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


def cmd_logs(pod_id: str):
    body = {
        "query": "query($podId: String!) { pod(input: { podId: $podId }) { id logs { logs } } }",
        "variables": {"podId": pod_id},
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(GQL_URL, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))


def cmd_terminate(pod_id: str):
    print(f"Terminating pod {pod_id}...")
    result = _request("DELETE", f"{REST_BASE}/pods/{pod_id}")
    print(json.dumps(result, indent=2) if result else "OK")


def cmd_wait_ready(pod_id: str, max_wait_s: int = 2700):
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
    print(f"\nTimed out after {max_wait_s}s — check logs")
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
    elif cmd == "logs":
        cmd_logs(sys.argv[2])
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
