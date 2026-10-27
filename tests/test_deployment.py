import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import deploy_runpod as deploy
import serve_model as serve


class DeploymentTests(unittest.TestCase):
    def test_dotenv_all_settings_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('# Comment\nHF_TOKEN="file-token"\nCTX_SIZE=4096\nexport PARALLEL=2\n')
            with patch.dict(os.environ, {"HF_TOKEN": "environment-token"}, clear=True):
                env = deploy.load_env(path)
            self.assertEqual(env["HF_TOKEN"], "environment-token")
            self.assertEqual(deploy.pod_config(env)["env"]["CTX_SIZE"], "4096")
            self.assertEqual(deploy.pod_config(env)["env"]["PARALLEL"], "2")

    def test_payload_uses_bonsai_image_and_passes_runtime_auth(self):
        config = deploy.pod_config({"HF_TOKEN": "hf-test", "LLAMA_API_KEY": "api-test"})
        self.assertIn("peppez07/bonsai-", config["imageName"])
        self.assertEqual(config["containerDiskInGb"], 150)
        self.assertEqual(deploy.pod_config({"CONTAINER_DISK_GB": "180"})["containerDiskInGb"], 180)
        self.assertEqual(config["env"]["HF_TOKEN"], "hf-test")
        self.assertEqual(config["env"]["HF_REVISION"], serve.DEFAULT_REVISION)
        self.assertEqual(config["dockerStartCmd"], ["python3", "/app/serve_model.py"])

    def test_dry_run_is_offline_and_redacted(self):
        output = io.StringIO()
        with patch.object(deploy, "load_env", return_value={
            "HF_TOKEN": "hf-private", "LLAMA_API_KEY": "inference-private"
        }), patch.object(deploy, "request") as request, contextlib.redirect_stdout(output):
            self.assertEqual(deploy.main(["launch", "--dry-run"]), 0)
        request.assert_not_called()
        self.assertNotIn("hf-private", output.getvalue())
        self.assertNotIn("inference-private", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["env"]["HF_TOKEN"], "<set>")

    def test_public_pod_requires_api_key_before_spending(self):
        with patch.object(deploy, "load_env", return_value={"HF_TOKEN": "hf-test"}), \
                patch.object(deploy, "request") as request, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main(["launch"]), 1)
        request.assert_not_called()

    def test_launch_does_not_retry_ambiguous_network_failure(self):
        env = {"HF_TOKEN": "hf-test", "LLAMA_API_KEY": "api-test"}
        with patch.object(deploy, "load_env", return_value=env), \
                patch.object(deploy, "request", side_effect=TimeoutError) as request, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deploy.main(["launch"]), 1)
        self.assertEqual(request.call_count, 1)

    def test_redacts_status_credentials_recursively(self):
        result = deploy.redacted({"env": {"HF_TOKEN": "secret", "LLAMA_API_KEY": "secret"},
                                 "nested": [{"password": "secret"}], "id": "pod"})
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(result["id"], "pod")

    def test_download_passes_token_revision_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"GGUF" + bytes(32))
            token_file = Path(directory) / "token"
            token_file.write_text("hf-from-file\n")
            download = Mock(return_value=str(model))
            hub = types.SimpleNamespace(hf_hub_download=download)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = serve.prepare_model({"HF_TOKEN_FILE": str(token_file), "HF_HUB_CACHE": directory})
            self.assertEqual(result, model)
            self.assertEqual(download.call_args.kwargs["token"], "hf-from-file")
            self.assertEqual(download.call_args.kwargs["revision"], serve.DEFAULT_REVISION)
            self.assertEqual(download.call_args.kwargs["cache_dir"], directory)

    def test_invalid_or_partial_download_cannot_start_server(self):
        for content in (b"", b"GGUF", b"<html>401 unauthorized</html>"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "bad.gguf"
                model.write_bytes(content)
                hub = types.SimpleNamespace(hf_hub_download=Mock(return_value=str(model)))
                with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    serve.prepare_model({"HF_TOKEN": "hf-test"})

    def test_startup_failure_does_not_expose_exception_or_exec(self):
        with patch.object(serve, "prepare_model", side_effect=RuntimeError("hf-secret")), \
                patch.object(serve.os, "execve") as execute, \
                contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(serve.main(), 1)
        execute.assert_not_called()
        self.assertNotIn("hf-secret", error.getvalue())

    def test_exec_removes_hf_credential_and_preserves_api_auth(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf-test", "HF_TOKEN_FILE": "/secret",
                                    "LLAMA_API_KEY": "api-test"}, clear=True), \
                patch.object(serve, "prepare_model", return_value=Path("/cache/model.gguf")), \
                patch.object(serve.os, "execve") as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            serve.main()
        binary, command, env = execute.call_args.args
        self.assertEqual(binary, command[0])
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HF_TOKEN_FILE", env)
        self.assertEqual(env["LLAMA_API_KEY"], "api-test")
        self.assertNotIn("hf-test", command)
        self.assertIn("--jinja", command)


if __name__ == "__main__":
    unittest.main()
