# uncensored

Deploy [`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)
(or any GGUF model) on a RunPod GPU pod, behind an OpenAI-compatible HTTP API.
Boots multimodal-ready out of the box.

```
                ┌─────────────────────────────────┐
                │ Your browser / OpenWebUI / API  │
                └────────────┬────────────────────┘
                             │ https
                             ▼
            https://<podId>-8000.proxy.runpod.net/v1
                             │
                             ▼
       ┌─────────────────────────────────────────┐
       │ RunPod pod · 48 GB VRAM · llama-server  │
       │ · model + mmproj on /workspace          │
       └─────────────────────────────────────────┘
```

## What you get on the pod

- `llama-server` from llama.cpp, built with CUDA
- Q6_K_P quant by default (~31 GB; override with `QUANT=`)
- `--jinja` chat template (required by the model card)
- `--mmproj` loaded → vision input works
- Built-in chat UI at `/`, OpenAI-compatible API at `/v1`

## One-time setup

1. **Accounts.** [RunPod](https://www.runpod.io/) account with credit; a [Hugging Face](https://huggingface.co/) token (any read-scope token is fine).
2. **Clone and configure.**
   ```bash
   git clone https://github.com/bobby-beckmann/uncensored.git
   cd uncensored
   cp .env.example .env
   # fill in RUNPOD_API_KEY and HF_TOKEN
   ```
3. **(Optional) Use the prebuilt image.** This repo's GitHub Actions builds a CUDA-ready `llama-server` image and pushes it to `ghcr.io/bobby-beckmann/llama-server-cuda:latest`. Once the package is public, future pods can skip the in-pod compile. (The current `deploy_runpod.py` builds from source on boot — slower first run but no auth needed.)

## Launch a pod

```bash
python3 deploy_runpod.py launch
# Pod allocated: xxxxxxxxx  cost $0.77/hr
# Endpoint will be: https://xxxxxxxxx-8000.proxy.runpod.net

python3 deploy_runpod.py wait xxxxxxxxx
# polls until /v1/models returns 200 (~10–25 min cold boot)
```

Open `https://xxxxxxxxx-8000.proxy.runpod.net/` to chat in the built-in UI, or point any OpenAI client at `https://xxxxxxxxx-8000.proxy.runpod.net/v1`.

## Stop a pod

```bash
python3 deploy_runpod.py terminate xxxxxxxxx
```

Billing stops immediately. The `/workspace` volume is destroyed with the pod, so next launch re-downloads the model (~6 min over 80 MB/s).

## Common changes

- **Different quant.** `QUANT=Q4_K_M python3 deploy_runpod.py launch` (saves ~10 GB VRAM at lower quality).
- **Different model.** Edit `HF_REPO`, `GGUF_FILE`, and `MMPROJ_FILE` near the top of `deploy_runpod.py`. The deploy script is generic — only the file names are model-specific.
- **Cheaper / faster GPU.** Edit `gpuTypeIds` in `POD_CONFIG`. Default tries RTX 6000 Ada → L40S → A6000 → L40 → A100 80GB in priority order. Inventory fluctuates; the script has a built-in retry loop that re-submits until allocation succeeds.
- **Larger context.** Bump `--ctx-size` in the startup script (currently 65536, model supports 262144).

## Other clients

The model alias is `hauhau-aggressive`. Any OpenAI-compatible client works:

```python
from openai import OpenAI
client = OpenAI(base_url="https://<podId>-8000.proxy.runpod.net/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="hauhau-aggressive",
    messages=[{"role": "user", "content": "hi"}],
)
print(r.choices[0].message.content)
```

For [Open WebUI](https://github.com/open-webui/open-webui) on your laptop:

```bash
docker run -d --name openwebui -p 3000:8080 \
  -e OPENAI_API_BASE_URL=https://<podId>-8000.proxy.runpod.net/v1 \
  -e OPENAI_API_KEY=anything \
  -v openwebui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

## Quirks worth knowing

- **Thinking mode is on by default.** Responses stream the model's reasoning to `delta.reasoning_content`, then the final answer to `delta.content`. If you only consume `content` you might see empty deltas for the first second or two. To disable, prepend `/no_think` to your prompt or pass `enable_thinking=false` via your chat template extension.
- **`-hf` direct-download is broken on the source-build path.** llama.cpp's embedded CURL was compiled without OpenSSL detection, so `llama-server -hf user/repo:file` fails with `HTTPS is not supported`. The deploy script works around this with `curl` for the initial download. The prebuilt GHCR image installs `libssl-dev`, so once you switch to it `-hf` works.
- **Image URLs in chat must be base64 data URLs**, *not* HTTPS links, for the same reason as above. The built-in chat UI does this transparently when you drag in an image.
- **The first build is slow.** ~25 min in-pod to compile llama.cpp from source. The artifacts cache to `/workspace` so a *restart* of the same pod is fast — but a *new* pod (different volume) rebuilds. Use the prebuilt GHCR image to skip this.

## Files

- `deploy_runpod.py` — pod lifecycle: `launch`, `status`, `wait`, `terminate`
- `Dockerfile.llama-server` — multi-stage CUDA build of llama-server
- `.github/workflows/build-llama-server-image.yml` — builds + pushes the image on every commit to main
- `.env.example` — template; copy to `.env` and fill in
