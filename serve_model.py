#!/usr/bin/env python3
"""Authenticate, cache the selected GGUF, and exec Prism llama-server."""
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF"
DEFAULT_REVISION = "5ab50fd49410941a8d4f97912303b1c7d458ffa2"
DEFAULT_FILE = "Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf"


def server_command(model, env):
    return [
        "/usr/local/bin/llama-server", "--model", str(model),
        "--alias", env.get("MODEL_ALIAS", "bonsai-heretic"),
        "--host", "0.0.0.0", "--port", env.get("PORT", "8000"),
        "--n-gpu-layers", env.get("N_GPU_LAYERS", "99"),
        "--ctx-size", env.get("CTX_SIZE", "32768"),
        "--parallel", env.get("PARALLEL", "1"),
        "--jinja", "--reasoning", "on",
        "--reasoning-effort", env.get("REASONING_EFFORT", "medium"),
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--metrics",
    ]


def prepare_model(env):
    # Explicit authentication avoids persisting a login credential to the volume.
    token = env.get("HF_TOKEN", "").strip()
    if env.get("HF_TOKEN_FILE"):
        token = Path(env["HF_TOKEN_FILE"]).read_text().strip()
    if not token:
        raise ValueError("Set HF_TOKEN or HF_TOKEN_FILE to a Hugging Face read token")
    repo = env.get("HF_REPO", DEFAULT_REPO)
    revision = env.get("HF_REVISION") or (DEFAULT_REVISION if repo == DEFAULT_REPO else "main")
    filename = env.get("GGUF_FILE", DEFAULT_FILE)
    if not filename.endswith(".gguf"):
        raise ValueError("GGUF_FILE must name a .gguf file")
    from huggingface_hub import hf_hub_download
    print(f"Downloading/checking {repo}/{filename} at {revision}", flush=True)
    model = Path(hf_hub_download(
        repo_id=repo, filename=filename, revision=revision, token=token,
        cache_dir=env.get("HF_HUB_CACHE", "/workspace/hf-cache"),
    ))
    # The Hub handles incomplete downloads and atomic cache finalization.
    with model.open("rb") as handle:
        if handle.read(4) != b"GGUF" or model.stat().st_size < 24:
            raise ValueError("Invalid GGUF header; clear its cache and retry")
    return model


def main():
    health = None
    try:
        if os.environ.get("RUNPOD_SERVERLESS", "").lower() in ("1", "true", "yes"):
            app_port = int(os.environ.get("PORT", "8000"))
            health_port = int(os.environ.get("PORT_HEALTH", "8001"))
            if app_port == health_port:
                raise ValueError("PORT_HEALTH must differ from PORT")
            # Start before the model download so RunPod sees initializing (204).
            # A separate process survives exec when Python becomes llama-server.
            health_env = {k: v for k, v in os.environ.items()
                          if not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD"))}
            health = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("runpod_health.py"))],
                env=health_env,
            )
        model = prepare_model(os.environ)
        command = server_command(model, os.environ)
        child_env = dict(os.environ)
        for name in ("HF_TOKEN", "HF_TOKEN_FILE", "HUGGING_FACE_HUB_TOKEN"):
            child_env.pop(name, None)
        if health is not None and health.poll() is not None:
            raise RuntimeError("RunPod health process exited before model startup")
        print(f"Model ready; starting Prism llama-server on port {os.environ.get('PORT', '8000')}", flush=True)
        os.execve(command[0], command, child_env)
    except Exception as exc:
        if health is not None:
            health.terminate()
            health.wait(timeout=5)
        # HTTP exception text may contain signed URLs; only emit its type.
        print(f"Startup failed ({type(exc).__name__}). Check token permissions, "
              "HF_REPO/HF_REVISION/GGUF_FILE, network access and free disk space.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
