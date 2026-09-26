import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from privacy_and_grokking.artifacts.cache import (
    get_artifact_bytes,
    get_artifact_json,
    get_artifact_path,
)


class MockMLflowHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if "training_config.json" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = json.dumps({"test_param": 123, "optimizer": {"name": "AdamW"}}).encode("utf-8")
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockMLflowHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_artifact_cache_local_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("privacy_and_grokking.artifacts.cache.get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("privacy_and_grokking.artifacts.local.get_cache_dir", lambda: tmp_path)

    run_id = "test_run_123"
    artifact_rel = "subfolder/config.json"
    target = tmp_path / "runs" / run_id / artifact_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"cached": True}), encoding="utf-8")

    # Should hit local cache without contacting any server
    path = get_artifact_path(run_id, artifact_rel, tracking_uri="http://invalid.domain:9999")
    assert path == target
    assert path.is_file()

    data = get_artifact_json(run_id, artifact_rel, tracking_uri="http://invalid.domain:9999")
    assert data == {"cached": True}

    raw = get_artifact_bytes(run_id, artifact_rel, tracking_uri="http://invalid.domain:9999")
    assert b"cached" in raw


def test_artifact_cache_download_and_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_server: str):
    monkeypatch.setattr("privacy_and_grokking.artifacts.cache.get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("privacy_and_grokking.artifacts.local.get_cache_dir", lambda: tmp_path)

    run_id = "remote_run_456"
    artifact_rel = "training_config.json"
    expected_cache_file = tmp_path / "runs" / run_id / artifact_rel

    assert not expected_cache_file.exists()

    # Fetch from mock server
    path = get_artifact_path(run_id, artifact_rel, tracking_uri=mock_server)
    assert path == expected_cache_file
    assert expected_cache_file.is_file()

    data = json.loads(expected_cache_file.read_text(encoding="utf-8"))
    assert data["test_param"] == 123

    # Subsequent call with invalid URI must succeed because of disk cache
    cached_path = get_artifact_path(run_id, artifact_rel, tracking_uri="http://invalid.domain:9999")
    assert cached_path == expected_cache_file
