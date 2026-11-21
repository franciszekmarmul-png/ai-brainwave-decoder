from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.data import DATASETS, PROCESSED_DIR
from src.live import MAX_TIME_LIMIT, MODELS, RunRequest, run

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SOURCE_FILES = ("train.py", "eegnet.py", "baselines.py", "live.py", "data.py")


class Run:
    def __init__(self, request: RunRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.events: list[dict] = []
        self.finished = False
        self.stop_requested = False
        self._changed = threading.Condition()

    def emit(self, event: dict) -> None:
        with self._changed:
            self.events.append(event)
            if event["type"] in ("done", "error"):
                self.finished = True
            self._changed.notify_all()

    def follow(self):
        index = 0
        while True:
            with self._changed:
                if index >= len(self.events) and not self.finished:
                    self._changed.wait(timeout=15)
                pending = self.events[index:]
                index += len(pending)
                done = self.finished and index >= len(self.events)
            if not pending and not done:
                yield None
            yield from pending
            if done:
                return

    def start(self) -> None:
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            run(self.request, self.emit, lambda: self.stop_requested)
        except Exception as err:
            traceback.print_exc()
            self.emit({"type": "error", "message": str(err)})


class Runs:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        self._active: Run | None = None

    def start(self, request: RunRequest) -> Run:
        request.check()
        with self._lock:
            if self._active and not self._active.finished:
                raise RuntimeError("a model is already training; stop it or wait for it to finish")
            current = Run(request)
            self._runs[current.id] = current
            self._active = current
        current.start()
        return current

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)


def options() -> dict:
    return {
        "max_time_limit": MAX_TIME_LIMIT,
        "models": [{"key": key, "name": name} for key, name in MODELS.items()],
        "datasets": [
            {
                "key": spec.key,
                "title": spec.title,
                "channels": spec.n_channels,
                "classes": list(spec.classes),
                "people": list(spec.subjects),
                "chance": spec.chance,
                "downloaded": (PROCESSED_DIR / f"{spec.key}.npz").exists(),
            }
            for spec in DATASETS.values()
        ],
    }


def make_handler(runs: Runs):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BrainwaveDecoder"

        def log_message(self, format: str, *args) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/options":
                return self._json(options())
            if path.startswith("/api/source/"):
                return self._source(path.removeprefix("/api/source/"))
            if path.startswith("/api/runs/") and path.endswith("/events"):
                return self._events(path.split("/")[3])
            return self._static(path)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/runs":
                return self._start()
            if path.startswith("/api/runs/") and path.endswith("/stop"):
                current = runs.get(path.split("/")[3])
                if current is None:
                    return self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
                current.stop_requested = True
                return self._json({"stopping": True})
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _start(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                request = RunRequest(str(body["dataset"]), int(body["person"]), str(body["model"]), float(body["time_limit"]))
                current = runs.start(request)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
                return self._json({"error": str(err)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as err:
                return self._json({"error": str(err)}, HTTPStatus.CONFLICT)
            self._json({"id": current.id}, HTTPStatus.CREATED)

        def _events(self, run_id: str) -> None:
            current = runs.get(run_id)
            if current is None:
                return self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in current.follow():
                    chunk = ": keep-alive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _source(self, name: str) -> None:
            if name not in SOURCE_FILES:
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            self._send((ROOT / "src" / name).read_bytes(), "text/plain; charset=utf-8")

        def _static(self, path: str) -> None:
            target = (WEB_DIR / (path.lstrip("/") or "index.html")).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(WEB_DIR) or not target.is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
                kind += "; charset=utf-8"
            self._send(target.read_bytes(), kind)

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload).encode(), "application/json", status)

        def _send(self, body: bytes, kind: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(Runs()))
    server.daemon_threads = True
    return server
