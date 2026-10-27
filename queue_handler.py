"""Forward RunPod Queue jobs to the local Prism HTTP server."""
import json
import os
import time
import urllib.error
import urllib.request


def make_payload(job_input):
    if not isinstance(job_input, dict):
        raise ValueError("input must be an object")
    messages = job_input.get("messages")
    if messages is None:
        prompt = job_input.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide input.prompt or input.messages")
        messages = [{"role": "user", "content": prompt}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    payload = {
        "model": os.environ.get("MODEL_ALIAS", "bonsai-heretic"),
        "messages": messages,
        "max_tokens": job_input.get("max_tokens", 512),
        "stream": False,
        # Avoid spending a short simple request's budget on reasoning by default.
        "chat_template_kwargs": {"enable_thinking": bool(job_input.get("enable_thinking", False))},
    }
    for key in ("temperature", "top_p", "top_k", "seed", "stop", "response_format"):
        if key in job_input:
            payload[key] = job_input[key]
    return payload


def handler(job):
    try:
        payload = make_payload(job.get("input"))
    except (ValueError, AttributeError) as error:
        return {"error": str(error)}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LLAMA_API_KEY"]
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]
        output = {
            "text": choice["message"].get("content", ""),
            "model": result.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage", {}),
        }
        if choice["message"].get("reasoning_content"):
            output["reasoning_content"] = choice["message"]["reasoning_content"]
        return output
    except urllib.error.HTTPError as error:
        return {"error": f"Prism server returned HTTP {error.code}"}
    except (urllib.error.URLError, TimeoutError):
        return {"error": "Prism server unavailable or generation timed out"}


def main():
    import runpod
    # Do not take jobs until the local model is ready.
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health"
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if json.load(response).get("status") == "ok":
                    break
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Prism did not become ready within 30 minutes")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
