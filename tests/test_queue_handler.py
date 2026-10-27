import io
import json
import os
import unittest
import urllib.error
from unittest.mock import MagicMock, Mock, patch

import queue_handler as queue
import serve_model as serve


class QueueTests(unittest.TestCase):
    def test_default_output_budget_and_explicit_client_limit(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(queue.make_payload({"prompt": "Hello"})["max_tokens"], 8192)
            self.assertEqual(queue.make_payload({"prompt": "Hello", "max_tokens": 512})["max_tokens"], 512)
        with patch.dict(os.environ, {"MAX_OUTPUT_TOKENS": "4096"}, clear=True):
            self.assertEqual(queue.make_payload({"prompt": "Hello"})["max_tokens"], 4096)

    def test_invalid_output_budget_is_rejected_before_network_call(self):
        for value in (0, -1, True, "8192", 1.5):
            with self.subTest(value=value), patch.object(queue.urllib.request, "urlopen") as request:
                self.assertIn("error", queue.handler({"input": {"prompt": "Hello", "max_tokens": value}}))
                request.assert_not_called()

    def test_token_limit_exhaustion_is_explicit_in_queue_output(self):
        result = {"model": "bonsai-heretic", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "unfinished"},
            "finish_reason": "length"}], "usage": {"completion_tokens": 32}}
        for body in ({"prompt": "Hello", "max_tokens": 32},
                     {"openai_input": {"messages": [{"role": "user", "content": "Hello"}],
                                       "max_tokens": 32}}):
            response = MagicMock()
            response.__enter__.return_value = io.BytesIO(json.dumps(result).encode())
            with self.subTest(body=body), patch.object(queue.urllib.request, "urlopen", return_value=response):
                output = queue.handler({"input": body})
            self.assertTrue(output["truncated"])
            self.assertIn("incomplete", output["warning"])

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

    def test_openai_contract_preserves_tools_and_tool_result_messages(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ]}, {"role": "tool", "tool_call_id": "call_1", "content": "file contents"}]
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        payload = queue.make_payload({"openai_input": {
            "messages": messages, "tools": tools, "tool_choice": "auto",
            "max_completion_tokens": 8192, "stream": True
        }})
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertFalse(payload["stream"])

    def test_openai_tool_response_is_not_reduced_to_text(self):
        result = {"model": "bonsai-heretic", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}], "usage": {}}
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(result).encode())
        with patch.object(queue.urllib.request, "urlopen", return_value=response):
            actual = queue.handler({"input": {"openai_input": {"messages": [
                {"role": "user", "content": "Read the file"}]}}})
        self.assertEqual(actual, result)

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
