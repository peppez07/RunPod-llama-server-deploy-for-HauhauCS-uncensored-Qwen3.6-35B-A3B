import contextlib
import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import runpod_health as health
import serve_model as serve


class HealthTests(unittest.TestCase):
    def test_ready_only_after_successful_model_load(self):
        response = Mock(status=200)
        response.read.return_value = json.dumps({"status": "ok"}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock()
        with patch.object(health.urllib.request, "urlopen", return_value=response):
            self.assertEqual(health.readiness_status("http://localhost/health"), 200)

    def test_loading_or_downloading_is_initializing(self):
        errors = [urllib.error.URLError("not listening"),
                  urllib.error.HTTPError("http://localhost", 503, "loading", {}, None)]
        for error in errors:
            with self.subTest(error=error), patch.object(
                    health.urllib.request, "urlopen", side_effect=error):
                self.assertEqual(health.readiness_status("http://localhost/health"), 204)

    def test_auth_error_is_not_reported_ready(self):
        error = urllib.error.HTTPError("http://localhost", 401, "unauthorized", {}, None)
        with patch.object(health.urllib.request, "urlopen", side_effect=error):
            self.assertEqual(health.readiness_status("http://localhost/health"), 503)

    def test_failed_download_stops_health_process(self):
        with patch.dict(os.environ, {"RUNPOD_SERVERLESS": "1", "HF_TOKEN": "private"}, clear=True), \
                patch.object(serve.subprocess, "Popen") as process, \
                patch.object(serve, "prepare_model", side_effect=ValueError), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(serve.main(), 1)
        process.return_value.terminate.assert_called_once()
        process.return_value.wait.assert_called_once()
        self.assertNotIn("HF_TOKEN", process.call_args.kwargs["env"])

    def test_serverless_health_survives_exec_and_uses_configured_port(self):
        with patch.dict(os.environ, {"RUNPOD_SERVERLESS": "1", "PORT": "8080",
                                    "PORT_HEALTH": "8081"}, clear=True), \
                patch.object(serve.subprocess, "Popen") as process, \
                patch.object(serve, "prepare_model", return_value=Path("/model.gguf")), \
                patch.object(serve.os, "execve") as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            process.return_value.poll.return_value = None
            serve.main()
        process.return_value.terminate.assert_not_called()
        command = execute.call_args.args[1]
        self.assertEqual(command[command.index("--port") + 1], "8080")


if __name__ == "__main__":
    unittest.main()
