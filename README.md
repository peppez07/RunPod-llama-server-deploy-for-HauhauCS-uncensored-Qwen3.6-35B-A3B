# uncensored

Deploy [`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)
(or any GGUF model) to a RunPod GPU pod behind an OpenAI-compatible HTTP API.
Multimodal-ready out of the box.

```
              ┌──────────────────────────────────┐
              │ Your browser, OpenWebUI, API,    │
              │ or anything that talks OpenAI v1 │
              └────────────────┬─────────────────┘
                               │ https
                               ▼
            https://<podId>-8000.proxy.runpod.net/v1
                               │
                               ▼
       ┌───────────────────────────────────────────┐
       │ RunPod pod · 48 GB VRAM · llama-server    │
       │   · model + mmproj on /workspace          │
       └───────────────────────────────────────────┘
```

## Quick start (~9 min)

```bash
git clone https://github.com/bobby-beckmann/uncensored.git
cd uncensored
cp .env.example .env             # fill in RUNPOD_API_KEY and HF_TOKEN
python3 deploy_runpod.py launch  # prints a pod id
python3 deploy_runpod.py wait <pod_id>
```

When `wait` reports `READY`, open `https://<pod_id>-8000.proxy.runpod.net/`
in a browser for the built-in chat UI (with sampling knobs in the sidebar
and drag-and-drop image input), or point any OpenAI client at
`https://<pod_id>-8000.proxy.runpod.net/v1`.

```bash
python3 deploy_runpod.py terminate <pod_id>   # stop the meter
```

Cost is roughly **$0.77/hr** on the default GPU (RTX 6000 Ada, 48 GB).
The `/workspace` volume is destroyed with the pod, so a fresh launch
re-downloads the model (~6 min over 80 MB/s).

## What you get on the pod

- `llama-server` from llama.cpp, built with CUDA.
- `Q6_K_P` quant by default (~31 GB on disk). Override: `QUANT=Q4_K_M python3 deploy_runpod.py launch`.
- `--jinja` chat template (required by this model card).
- `--mmproj` loaded so vision input works.
- Built-in chat UI at `/`, OpenAI-compatible API at `/v1`.
- `--ctx-size 65536` split across `--parallel 4` slots (16 K context per concurrent user; bump in `deploy_runpod.py` if you need more).

## How it boots fast

By default the pod pulls a **prebuilt CUDA image** from this repo's
GitHub Container Registry:
[`ghcr.io/bobby-beckmann/llama-server-cuda:latest`](https://github.com/bobby-beckmann/uncensored/pkgs/container/llama-server-cuda).

The image is rebuilt automatically on every push to `main` via
[`.github/workflows/build-llama-server-image.yml`](.github/workflows/build-llama-server-image.yml).
It bakes only the binary — no model — so the same image works for any
GGUF you want to swap in.

If you need a fresh build (testing a custom llama.cpp branch, recovering
from a broken image, etc.), fall back to compiling on the pod:

```bash
BUILD_FROM_SOURCE=1 python3 deploy_runpod.py launch
```

This boots `nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04`, clones llama.cpp,
compiles it on the pod (~25 min cold), then runs. SSH on port 22 is
enabled in this mode for debugging.

## Swapping in a different model

No fork or file edit needed — `deploy_runpod.py` reads the model from
env vars. The prebuilt image contains only `llama-server`, so it serves
any GGUF you can name.

**Multimodal model from Hugging Face:**

```bash
HF_REPO="unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF" \
GGUF_FILE="Qwen3-VL-30B-A3B-Instruct-Q6_K.gguf" \
MMPROJ_FILE="mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf" \
MODEL_ALIAS="qwen3-vl" \
python3 deploy_runpod.py launch
```

**Text-only model:** set `MMPROJ_FILE=` empty — the script will skip both
the projector download and the `--mmproj` flag automatically.

```bash
HF_REPO="bartowski/Llama-3.3-70B-Instruct-GGUF" \
GGUF_FILE="Llama-3.3-70B-Instruct-Q4_K_M.gguf" \
MMPROJ_FILE="" \
MODEL_ALIAS="llama-3.3-70b" \
python3 deploy_runpod.py launch
```

Bigger models need a bigger card — bump the `gpuTypeIds` priority list
in `deploy_runpod.py` (e.g. lead with `NVIDIA A100 80GB PCIe`) if a
48 GB card won't hold your quant.

If you want your own GHCR image (you're forking and don't want to depend
on `bobby-beckmann/llama-server-cuda`):

1. Push to your fork — the GHA workflow builds on every commit to `main`.
2. After the first build, flip `llama-server-cuda` in your packages page
   to **public** (Package settings → Danger zone → Change visibility).
3. Run with `LLAMA_SERVER_IMAGE=ghcr.io/<you>/llama-server-cuda:latest python3 deploy_runpod.py launch`.

## Other clients

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://<pod_id>-8000.proxy.runpod.net/v1",
    api_key="not-needed",
)
r = client.chat.completions.create(
    model="hauhau-aggressive",
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
```

[Open WebUI](https://github.com/open-webui/open-webui) on your laptop:

```bash
docker run -d --name openwebui -p 3000:8080 \
  -e OPENAI_API_BASE_URL=https://<pod_id>-8000.proxy.runpod.net/v1 \
  -e OPENAI_API_KEY=anything \
  -v openwebui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

## Knobs

| Env var               | Default                                            | What it does                                          |
|-----------------------|----------------------------------------------------|-------------------------------------------------------|
| `RUNPOD_API_KEY`      | (required)                                         | Auth for the RunPod REST API.                         |
| `HF_TOKEN`            | (required)                                         | Hugging Face read token; used by the pod to download. |
| `HF_REPO`             | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-...`          | Hugging Face repo to pull the GGUF from.              |
| `GGUF_FILE`           | `Qwen3.6-35B-A3B-Uncensored-...-{QUANT}.gguf`      | Exact filename inside that repo.                      |
| `MMPROJ_FILE`         | `mmproj-Qwen3.6-...-f16.gguf`                      | Vision projector filename. Empty = text-only model.   |
| `MODEL_ALIAS`         | `hauhau-aggressive`                                | `--alias` shown to OpenAI clients as the model id.    |
| `QUANT`               | `Q6_K_P`                                           | Quant tier; substituted into the default `GGUF_FILE`. |
| `POD_NAME`            | `<MODEL_ALIAS>-llamaserver`                        | Pod name shown in the RunPod dashboard.               |
| `BUILD_FROM_SOURCE`   | unset                                              | Set to `1` to compile llama.cpp on the pod instead.   |
| `LLAMA_SERVER_IMAGE`  | `ghcr.io/bobby-beckmann/llama-server-cuda:latest`  | Override the prebuilt image (e.g. to your fork).      |

## Quirks worth knowing

- **Thinking mode is on by default.** Responses stream the model's
  reasoning to `delta.reasoning_content`, then the final answer to
  `delta.content`. A client that only consumes `content` will see empty
  deltas for a second or two. Prepend `/no_think` to disable per-turn,
  or pass `enable_thinking=false` via the chat template extension.
- **Image input.** The built-in UI accepts drag-and-drop and encodes as
  a base64 data URL. HTTPS image URLs work only on the prebuilt-image
  path (llama.cpp's embedded CURL needs OpenSSL, which the prebuilt
  Dockerfile installs but the source-build apt block doesn't).
- **Inventory bounces.** RunPod can be momentarily out of any given GPU.
  `deploy_runpod.py launch` has a built-in retry loop that submits
  every 10 s until a machine sticks, falling through the priority list
  RTX 6000 Ada → L40S → A6000 → L40 → A100.

## Layout

| Path                                              | What it is                                                |
|---------------------------------------------------|-----------------------------------------------------------|
| `deploy_runpod.py`                                | Pod lifecycle: `launch`, `status`, `wait`, `terminate`.   |
| `Dockerfile.llama-server`                         | Multi-stage CUDA build of `llama-server`.                 |
| `.github/workflows/build-llama-server-image.yml`  | Builds and pushes the image on every commit to `main`.    |
| `.env.example`                                    | Template — copy to `.env` and fill in.                    |
