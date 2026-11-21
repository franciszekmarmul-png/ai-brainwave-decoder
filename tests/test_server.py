import json
import threading
import urllib.error
import urllib.request

import pytest

from src import server


@pytest.fixture
def app():
    httpd = server.serve(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_options_list_every_dataset_and_model(app):
    status, kind, body = get(f"{app}/api/options")
    options = json.loads(body)
    assert status == 200 and kind == "application/json"
    assert [d["key"] for d in options["datasets"]] == ["bnci2014_001", "bnci2014_004", "physionet_mi"]
    assert [m["key"] for m in options["models"]] == ["eegnet", "fbcsp", "csp_lda"]


def test_the_page_and_its_assets_are_served(app):
    assert b"AI Brainwave Decoder" in get(f"{app}/")[2]
    assert get(f"{app}/app.js")[1].startswith(("application/javascript", "text/javascript"))


def test_source_is_limited_to_the_listed_files(app):
    assert b"class EEGNet" in get(f"{app}/api/source/eegnet.py")[2]
    with pytest.raises(urllib.error.HTTPError):
        get(f"{app}/api/source/server.py")


def test_files_outside_the_web_folder_cannot_be_read(app):
    with pytest.raises(urllib.error.HTTPError):
        get(f"{app}/../app.py")
    with pytest.raises(urllib.error.HTTPError):
        get(f"{app}/%2e%2e/app.py")


def test_bad_run_requests_get_a_clear_error(app):
    status, body = post(f"{app}/api/runs", {"dataset": "bnci2014_001", "person": 42, "model": "eegnet", "time_limit": 60})
    assert status == 400 and "person" in body["error"]


def test_only_one_run_trains_at_a_time(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_run(request, emit, should_stop):
        started.set()
        release.wait(5)
        emit({"type": "done"})

    monkeypatch.setattr(server, "run", slow_run)
    runs = server.Runs()
    request = server.RunRequest("bnci2014_001", 1, "eegnet", 60)
    first = runs.start(request)
    started.wait(5)
    with pytest.raises(RuntimeError, match="already training"):
        runs.start(request)
    release.set()
    assert [e["type"] for e in first.follow()] == ["done"]
