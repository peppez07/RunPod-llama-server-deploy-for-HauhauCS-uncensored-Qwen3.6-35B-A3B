# Ternary Bonsai on Docker and RunPod

Run [OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF)
as an OpenAI-compatible API on port 8000.

This model needs the [Prism ML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp).
The Dockerfile builds that fork at a pinned commit, including its ternary kernels
and Hadamard transforms. Stock llama.cpp images and the previous HauhauCS image
are not suitable substitutes for this model.

The default file is `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf`
(7,206,168,928 bytes), pinned to model revision
`5ab50fd49410941a8d4f97912303b1c7d458ffa2`.
The runtime is pinned to Prism commit
`c96baf4abe24d11a50f326e40b6223b687449d63`.
Defaults follow the model card: 32K context, Jinja, medium reasoning,
temperature 1.0, top-p 0.95, top-k 20. One parallel slot avoids multiplying
the initial memory requirement. This setup serves text; it does not download a vision projector.

## 1. Configure credentials

Download/clone this repository and open a terminal in its root. Copy the example:

```sh
cp .env.example .env
```

On Windows PowerShell use `Copy-Item .env.example .env`.

Edit `.env` locally:

```dotenv
HF_TOKEN=hf_your_read_token
LLAMA_API_KEY=your_long_random_inference_password
RUNPOD_API_KEY=your_runpod_key
```

Create a Hugging Face **read** token at
[Hugging Face token settings](https://huggingface.co/settings/tokens), with access
to the model repository. The startup process passes it directly to
`hf_hub_download(token=...)`; a separate `hf auth login` command is unnecessary.
It does not save a login token to the model volume.
Credentials are runtime settings, never Docker build arguments or image layers.
`.env` is excluded from Git and the Docker build context.

`LLAMA_API_KEY` protects inference requests and is separate from the Hugging Face token.
It is required by the RunPod launcher. Compose binds only to localhost and allows
it to be empty. `RUNPOD_API_KEY` is only needed by the RunPod launcher and is
not passed to the inference container.

## 2A. Run Docker locally

Prerequisites: a Linux x86-64 NVIDIA GPU host, or Windows with Docker Desktop's
WSL2 Linux GPU support; an NVIDIA driver compatible with CUDA 12.8;
Docker Engine/Desktop, Compose **2.30+**, and GPU access configured.
On Linux install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Check GPU access before building:

```sh
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
docker compose up --build -d
docker compose logs -f llama-server
```

The first build compiles the CUDA runtime; allow substantial time and disk space
(at least 30 GB free for the build/cache and model). The first startup downloads
the model; later startups reuse the named `model-cache` volume. The Hub maintains
incomplete downloads separately and finalizes them atomically. Invalid GGUF headers
fail startup instead of loading an error page.

```sh
curl http://localhost:8000/health
```

Wait for HTTP 200 and `{"status":"ok"}`. Downloading/loading can take several minutes.
The browser UI is available at `http://localhost:8000` if the runtime's prebuilt UI
assets are available. Use your `LLAMA_API_KEY` when prompted.
To stop without deleting the cache: `docker compose down`.
`docker compose down -v` also deletes downloaded models.

The default build targets Ampere/Ada GPUs (SM 80/86/89), including A6000/A100/L40S.
The model file size is not total VRAM usage: context, working buffers and runtime
allocations need additional space. Lower `CTX_SIZE` (for example 8192) or reduce
`N_GPU_LAYERS` if loading runs out of memory.
To support another GPU, rebuild with a matching `CUDA_ARCHITECTURES` argument.

## 2B. Build the image and deploy on RunPod

1. Enable Actions in this fork if GitHub shows they are disabled.
2. Merge the setup changes into `main`, or run **Actions → Build Bonsai Prism CUDA image → Run workflow**
   on the branch you want to build. Leave the runtime ref blank to use the pinned commit.
3. Wait for the workflow to pass. It tests the Python code, builds the image, checks
   startup fails cleanly without a token, and publishes
   `ghcr.io/peppez07/bonsai-llama-server-cuda:latest` plus a commit SHA tag.
   Pull requests build/test without publishing.
4. Make the GHCR package public in GitHub package settings, or configure a RunPod
   registry credential and set `CONTAINER_REGISTRY_AUTH_ID` in `.env`.
   The Hugging Face token authenticates model downloads; GHCR has its own image-pull credentials.
5. Prefer the published commit SHA tag in `LLAMA_SERVER_IMAGE` for repeatable deployment.

With Python **3.10+** installed (no Python packages needed for the launcher):

```sh
python deploy_runpod.py launch --dry-run
python deploy_runpod.py launch
python deploy_runpod.py wait POD_ID
python deploy_runpod.py status POD_ID
```

Use `py -3` instead of `python` on Windows if needed.
Launching allocates a billable GPU pod. The launcher makes one creation request
and prints its ID; it does not retry ambiguous failures and risk duplicate pods.
Check the RunPod console before retrying a failed launch.

The pod uses this project's Docker startup command, exposes `8000/http`,
and caches models under `/workspace/hf-cache` on a 30 GB volume.
RunPod serves the API at:

```text
https://POD_ID-8000.proxy.runpod.net/v1
```

`wait` polls the public `/health` route, returning a nonzero exit code after
30 minutes if the server is not ready. A timeout does not stop billing.
Inspect container logs in RunPod for download, driver, or model-load failures.
Status and dry-run output redact credential fields.

When finished:

```sh
python deploy_runpod.py terminate POD_ID
```

Termination deletes the pod and its pod-local volume/cache.
To retain the cache across pod deletion, provision a compatible RunPod network
volume and set `NETWORK_VOLUME_ID`. Its location constrains available GPUs.

## 3. Send a request

Save the following as `request.json`:

```json
{
  "model": "bonsai-heretic",
  "messages": [{"role": "user", "content": "Explain how a rainbow forms."}],
  "max_tokens": 512,
  "stream": false
}
```

For local Docker:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_LLAMA_API_KEY" \
  --data-binary @request.json
```

On Windows use `curl.exe` and put the command on one line.
For RunPod, replace `http://localhost:8000` with your pod's HTTPS proxy URL.
Never send the Hugging Face or RunPod token as your inference API password.

## Configuration

All launcher settings are read from `.env`, with exported environment variables
taking precedence. Use simple `KEY=value` entries, optionally quoted; shell
substitution and inline comments are not supported. Compose also reads `.env`.

| Setting | Purpose / default |
| --- | --- |
| `HF_TOKEN` | Hugging Face read token |
| `LLAMA_API_KEY` | Your inference API password |
| `HF_REPO` | OS-Software Bonsai Heretic repository |
| `HF_REVISION` | Pinned model commit; change alongside the repository/file |
| `GGUF_FILE` | Exact PQ2_0 filename |
| `MODEL_ALIAS` | `bonsai-heretic` |
| `CTX_SIZE` | `32768` |
| `PARALLEL` | `1` |
| `N_GPU_LAYERS` | `99`; lower for partial CPU offload |
| `REASONING_EFFORT` | `medium` |
| `LLAMA_SERVER_IMAGE` | Your published Prism image (RunPod only) |
| `GPU_TYPE_IDS` | Comma-separated RunPod GPU types |
| `VOLUME_GB` | `30` (RunPod only) |
| `NETWORK_VOLUME_ID` | Optional persistent RunPod network volume |
| `CONTAINER_REGISTRY_AUTH_ID` | Optional RunPod private registry credential |

For the smaller packing, set
`GGUF_FILE=Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf`.
It also requires the Prism runtime.
When using `docker run` directly, the startup supports `HF_TOKEN_FILE` pointing
to a mounted token file, as an alternative to `HF_TOKEN`.
The old `QUANT`, `MMPROJ_FILE` and pod-side `BUILD_FROM_SOURCE` paths are replaced
by exact model filenames and a single Docker build/runtime.

## Validation

```sh
python -m unittest discover -s tests -v
python deploy_runpod.py launch --dry-run
```

Tests use mocked downloads and RunPod calls; they do not allocate GPUs or download
the 7.2 GB model. The Docker build and GPU inference still need validation on the
target host. A passing health check proves loading; a successful chat request
checks generation.

References: [model usage](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF#usage),
[Prism runtime](https://github.com/PrismML-Eng/llama.cpp),
[Hub downloads](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).

