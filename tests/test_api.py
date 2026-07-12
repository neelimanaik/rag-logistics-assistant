"""API tests using FastAPI's TestClient.

The assistant is overridden with a fake, so these run with no model and no index
(proving the API layer is decoupled from the RAG engine).
"""
from fastapi.testclient import TestClient

from app import app, get_assistant
from src.api import rate_limit as rl


class _FakeAssistant:
    def ask(self, question, request_id=None):
        return (
            "answer text",
            [{"document": "d.pdf", "section": "S", "pages": "1 - 1"}],
            "HIGH",
        )

    def ask_stream(self, question, request_id=None):
        yield {"type": "metadata", "confidence": "HIGH", "citations": []}
        yield {"type": "token", "text": "hi"}
        yield {"type": "done"}


def _client_with_fake():
    app.dependency_overrides[get_assistant] = lambda: _FakeAssistant()
    return TestClient(app)


def _token(client):
    resp = client.post("/token", data={"username": "analyst", "password": "demo"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def teardown_function():
    app.dependency_overrides.clear()
    rl.reset()


def test_health_is_public():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_login_success_returns_token():
    client = TestClient(app)
    resp = client.post("/token", data={"username": "analyst", "password": "demo"})
    assert resp.status_code == 200
    assert resp.json()["token_type"] == "bearer"
    assert resp.json()["access_token"]


def test_login_bad_credentials_is_401():
    client = TestClient(app)
    resp = client.post("/token", data={"username": "analyst", "password": "wrong"})
    assert resp.status_code == 401


def test_ask_requires_auth():
    client = _client_with_fake()
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 401


def test_ask_returns_answer_with_token():
    client = _client_with_fake()
    token = _token(client)
    resp = client.post(
        "/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer text"
    assert body["confidence"] == "HIGH"
    assert body["citations"][0]["document"] == "d.pdf"


def test_ask_stream_with_token_emits_sse():
    client = _client_with_fake()
    token = _token(client)
    resp = client.post(
        "/ask/stream",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "data:" in resp.text
    assert "metadata" in resp.text and "done" in resp.text


def test_rate_limit_returns_429_after_max(monkeypatch):
    monkeypatch.setattr(rl, "MAX_REQUESTS", 2)
    rl.reset()
    client = _client_with_fake()
    headers = {"Authorization": f"Bearer {_token(client)}"}

    # First MAX_REQUESTS calls succeed...
    for _ in range(2):
        ok = client.post("/ask", json={"question": "hi"}, headers=headers)
        assert ok.status_code == 200

    # ...the next one is rate-limited.
    limited = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert limited.status_code == 429
