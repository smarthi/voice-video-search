"""visual-embed / caption FastAPI services and their HTTP clients.

No model download: every route handler calls into `_embed_qwen_images`,
`_embed_qwen_texts`, `_qwen_caption`, or `_qwen_rewrite`, and every test here
monkeypatches exactly that function to a fast fake -- the same seam
`test_embed.py` and `test_ocr.py` already use to test routing without a GPU.

Two layers are tested:
  1. The FastAPI app in isolation (`fastapi.testclient.TestClient`), which
     exercises real multipart/JSON parsing and response serialization
     without a real socket.
  2. The plain HTTP client functions (`embed_client.py` / `caption_client.py`)
     against a *real* server bound to 127.0.0.1 on an ephemeral port -- these
     call `httpx.post` with a real URL, so a mock transport would not catch a
     wire-format mismatch the way an actual round trip does.
"""

from __future__ import annotations

import io
import socket
import threading
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vvs.media import caption as caption_module
from vvs.media.caption_client import CaptionServiceError, remote_caption_image, remote_expand_query_terms
from vvs.retrieve import embed as embed_module
from vvs.retrieve.embed_client import EmbedServiceError, remote_embed_images, remote_embed_texts
from vvs.services.caption_server import app as caption_app
from vvs.services.visual_embed_server import app as visual_embed_app


def _tiny_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 40, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_tiny_jpeg(path) -> None:
    Image.new("RGB", (8, 8), color=(10, 200, 90)).save(path, format="JPEG")


class _Monkeypatch:
    """Same shim tests/test_corpus.py uses, for scripts run without pytest."""

    def __init__(self) -> None:
        self._saved: list[tuple[object, str, object]] = []

    def setattr(self, target, name, value) -> None:
        self._saved.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, value in reversed(self._saved):
            setattr(target, name, value)


# --- server route tests (TestClient, no real socket) -----------------------


def test_visual_embed_health_and_info_do_not_need_a_model() -> None:
    client = TestClient(visual_embed_app)
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert info["model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert info["dim"] > 0


def test_visual_embed_images_route(monkeypatch) -> None:
    fake = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(embed_module, "_embed_qwen_images", lambda paths, **_kw: fake[: len(paths)])
    client = TestClient(visual_embed_app)
    jpeg = _tiny_jpeg_bytes()
    resp = client.post(
        "/embed/images",
        files=[("files", ("a.jpg", jpeg, "image/jpeg")), ("files", ("b.jpg", jpeg, "image/jpeg"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 2
    assert np.allclose(body["vectors"], fake)


def test_visual_embed_images_route_rejects_empty_upload() -> None:
    client = TestClient(visual_embed_app)
    resp = client.post("/embed/images", files=[])
    assert resp.status_code in (400, 422)  # FastAPI 422s a missing required field


def test_visual_embed_texts_route(monkeypatch) -> None:
    fake = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    monkeypatch.setattr(embed_module, "_embed_qwen_texts", lambda texts, **_kw: fake)
    client = TestClient(visual_embed_app)
    resp = client.post("/embed/texts", json={"texts": ["a query"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 3
    assert np.allclose(body["vectors"], fake)


def test_caption_route(monkeypatch) -> None:
    monkeypatch.setattr(
        caption_module, "_qwen_caption", lambda image: ("a red square", ["square"], "fake-model")
    )
    client = TestClient(caption_app)
    resp = client.post("/caption", files={"file": ("f.jpg", _tiny_jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"caption": "a red square", "objects": ["square"], "model": "fake-model"}


def test_expand_query_route(monkeypatch) -> None:
    monkeypatch.setattr(caption_module, "_qwen_rewrite", lambda prompt: "running water\nstream")
    client = TestClient(caption_app)
    resp = client.post("/expand_query", json={"query": "river", "max_variants": 2})
    assert resp.status_code == 200
    assert resp.json()["variants"] == ["running water", "stream"]


def test_expand_query_route_rejects_empty_query() -> None:
    client = TestClient(caption_app)
    resp = client.post("/expand_query", json={"query": "  "})
    assert resp.status_code == 400


# --- real server, real socket: the actual client functions -----------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_app_in_background(app, port: int):
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("test server did not start in time")
    return server, thread


def _stop(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=5.0)


def test_embed_client_round_trips_against_a_real_server(tmp_path) -> None:
    fake = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mp = _Monkeypatch()
    mp.setattr(embed_module, "_embed_qwen_images", lambda paths, **_kw: fake[: len(paths)])
    mp.setattr(embed_module, "_embed_qwen_texts", lambda texts, **_kw: fake[: len(texts)])
    port = _free_port()
    server, thread = _run_app_in_background(visual_embed_app, port)
    try:
        p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
        _write_tiny_jpeg(p1)
        _write_tiny_jpeg(p2)
        vecs = remote_embed_images([p1, p2], f"http://127.0.0.1:{port}")
        assert np.allclose(vecs, fake)
        vecs2 = remote_embed_texts(["q1", "q2"], f"http://127.0.0.1:{port}")
        assert np.allclose(vecs2, fake)
    finally:
        _stop(server, thread)
        mp.undo()


def test_embed_client_raises_on_an_unreachable_service() -> None:
    try:
        remote_embed_images(["/dev/null"], "http://127.0.0.1:1", timeout_s=2.0)
        raise AssertionError("expected EmbedServiceError")
    except EmbedServiceError:
        pass


def test_caption_client_round_trips_against_a_real_server(tmp_path) -> None:
    mp = _Monkeypatch()
    mp.setattr(caption_module, "_qwen_caption", lambda image: ("a caption", ["obj"], "fake-model"))
    mp.setattr(caption_module, "_qwen_rewrite", lambda prompt: "alt phrase one\nalt phrase two")
    port = _free_port()
    server, thread = _run_app_in_background(caption_app, port)
    try:
        img_path = tmp_path / "still.jpg"
        _write_tiny_jpeg(img_path)
        cap, objects, model = remote_caption_image(img_path, f"http://127.0.0.1:{port}")
        assert (cap, objects, model) == ("a caption", ["obj"], "fake-model")
        variants = remote_expand_query_terms("river", f"http://127.0.0.1:{port}", max_variants=2)
        assert variants == ["alt phrase one", "alt phrase two"]
    finally:
        _stop(server, thread)
        mp.undo()


def test_caption_client_raises_on_an_unreachable_service() -> None:
    try:
        remote_expand_query_terms("river", "http://127.0.0.1:1", timeout_s=2.0)
        raise AssertionError("expected CaptionServiceError")
    except CaptionServiceError:
        pass


# --- wiring inside embed.py / caption.py ------------------------------------


def test_remote_qwen_ok_requires_both_url_and_the_default_model(monkeypatch) -> None:
    from vvs.config import Settings

    cfg_off = Settings(visual_embed_url=None)
    assert embed_module._remote_qwen_ok(settings=cfg_off) is False

    cfg_on = Settings(visual_embed_url="http://embed:8001")
    assert embed_module._remote_qwen_ok(settings=cfg_on) is True

    # An explicit non-default model must not be proxied to a service that
    # only ever loads the default weights.
    cfg_8b = Settings(visual_embed_url="http://embed:8001", visual_model="Qwen/Qwen3-VL-Embedding-8B")
    assert embed_module._remote_qwen_ok(settings=cfg_8b) is False


def test_caption_image_uses_the_remote_client_when_configured(monkeypatch) -> None:
    from vvs.config import Settings
    from vvs.media import caption_client as caption_client_module

    calls: list[str] = []
    monkeypatch.setattr(
        caption_module,
        "load_settings",
        lambda: Settings(caption_service_url="http://caption:8002", scene_caption_model="qwen_vl"),
    )
    monkeypatch.setattr(
        caption_client_module,
        "remote_caption_image",
        lambda path, url, **_kw: calls.append(url) or ("remote caption", [], "remote-model"),
    )
    result = caption_module.caption_image("/does/not/matter.jpg")
    assert result == ("remote caption", [], "remote-model")
    assert calls == ["http://caption:8002"]


def test_expand_query_terms_uses_the_remote_client_when_configured(monkeypatch) -> None:
    from vvs.config import Settings
    from vvs.media import caption_client as caption_client_module

    monkeypatch.setattr(caption_module, "_qwen_expand_disabled", None)
    monkeypatch.setattr(
        caption_module,
        "load_settings",
        lambda: Settings(caption_service_url="http://caption:8002", query_expand=True, query_expand_variants=2),
    )
    monkeypatch.setattr(
        caption_client_module,
        "remote_expand_query_terms",
        lambda query, url, **_kw: ["remote variant"],
    )
    assert caption_module.expand_query_terms("river") == ["remote variant"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
        kwargs = {}
        if "monkeypatch" in params:
            kwargs["monkeypatch"] = _Monkeypatch()
        if "tmp_path" in params:
            import tempfile
            from pathlib import Path

            kwargs["tmp_path"] = Path(tempfile.mkdtemp())
        fn(**kwargs)
        if "monkeypatch" in kwargs:
            kwargs["monkeypatch"].undo()
    print("ok")
