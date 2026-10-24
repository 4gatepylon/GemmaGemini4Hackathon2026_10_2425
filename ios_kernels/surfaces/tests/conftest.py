import socket
import threading
import time

import pytest
import uvicorn

from surfaces.openai_server.server import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    """Start a real uvicorn server for a runner in a background thread; yields its base URL."""
    servers = []

    def start(runner) -> str:
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(runner), host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        servers.append(server)
        return f"http://127.0.0.1:{port}/v1"

    yield start
    for s in servers:
        s.should_exit = True
