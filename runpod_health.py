"""RunPod Load Balancer /ping adapter, on a separate health port."""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def readiness_status(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 if response.status == 200 and json.load(response).get("status") == "ok" else 503
    except urllib.error.HTTPError as error:
        return 204 if error.code == 503 else 503
    except (urllib.error.URLError, TimeoutError):
        return 204  # Downloading or loading; llama-server is not listening yet.
    except (ValueError, AttributeError):
        return 503


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/ping":
            self.send_error(404)
            return
        status = readiness_status(self.server.backend_url)
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    app_port = int(os.environ.get("PORT", "8000"))
    health_port = int(os.environ.get("PORT_HEALTH", "8001"))
    if app_port == health_port:
        raise ValueError("PORT_HEALTH must differ from PORT for the /ping adapter")
    server = ThreadingHTTPServer(("0.0.0.0", health_port), HealthHandler)
    server.backend_url = f"http://127.0.0.1:{app_port}/health"
    print(f"RunPod readiness: /ping on port {health_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
