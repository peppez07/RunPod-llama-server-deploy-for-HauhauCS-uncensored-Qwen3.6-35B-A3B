import io
import json
import os
import unittest
import urllib.error
from unittest.mock import MagicMock, Mock, patch

import queue_handler as queue
import serve_model as serve


class QueueTests(unittest.TestCase):
    def test_supervisor_stops_consumer_if_model_server_exits(self):
        server = Mock()
        server.poll.return_value = 1
        worker = Mock()
        worker.poll.return_value = None
        with patch.object(serve.subprocess, "Popen", side_effect=[server, worker]), \
                patch.object(serve.signal, "signal"), self.assertRaises(RuntimeError):
            serve.run_queue(["llama-server"], {})
        worker.terminate.assert_called_once()
        worker.wait.assert_called_once()
        server.terminate.assert_not_called()

    def test_auto_detect_queue_and_allow_http_override(self):
        env = {"RUNPOD_WEBHOOK_GET_JOB": "https://api.runpod.ai/v2/id/job-take/worker"}
        self.assertTrue(serve.queue_enabled(env))
        self.assertFalse(serve.queue_enabled({**env, "RUNPOD_ENDPOINT_TYPE": "LOAD_BALANCER"}))
        self.assertFalse(serve.queue_enabled({}))

    def test_prompt_conversion_and_bounded_sync_output(self):
        result = queue.make_payload({"prompt": "write Hello World", "max_tokens": 64})
        self.assertEqual(result["messages"][0]["content"], "write Hello World")
        self.assertEqual(result["max_tokens"], 64)
        self.assertFalse(result["stream"])
        self.assertFalse(result["chat_template_kwargs"]["enable_thinking"])

    def test_chat_messages_and_reasoning_opt_in(self):
        messages = [{"role": "user", "content": "Hello"}]
        payload = queue.make_payload({"messages": messages, "enable_thinking": True})
        self.assertEqual(payload["messages"], messages)
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])

    def test_bad_input_does_not_call_backend(self):
        with patch.object(queue.urllib.request, "urlopen") as request:
            self.assertIn("error", queue.handler({"input": {"prompt": ""}}))
        request.assert_not_called()

    def test_http_generation_is_returned_as_queue_output(self):
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps({
            "model": "bonsai-heretic", "choices": [
                {"message": {"content": "Hello World"}, "finish_reason": "stop"}
            ], "usage": {"completion_tokens": 3}
        }).encode())
        with patch.dict(os.environ, {"LLAMA_API_KEY": "private"}, clear=True), \
                patch.object(queue.urllib.request, "urlopen", return_value=response) as request:
            result = queue.handler({"input": {"prompt": "write Hello World"}})
        self.assertEqual(result["text"], "Hello World")
        self.assertNotIn("private", json.dumps(result))
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], "bonsai-heretic")

    def test_server_error_is_reported_without_body_or_credentials(self):
        error = urllib.error.HTTPError("http://localhost", 500, "private", {}, None)
        with patch.object(queue.urllib.request, "urlopen", side_effect=error):
            result = queue.handler({"input": {"prompt": "Hello"}})
        self.assertEqual(result, {"error": "Prism server returned HTTP 500"})


if __name__ == "__main__":
    unittest.main()
