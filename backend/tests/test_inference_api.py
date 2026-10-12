"""Integration tests crossing the HTTP boundary for in-memory model inference."""

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_demo_token
from app.main import app

client = TestClient(app)


@pytest.fixture
def operator_headers():
    token = create_demo_token(role="operator")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers():
    token = create_demo_token(role="admin")
    return {"Authorization": f"Bearer {token}"}


def test_models_list_authenticated(operator_headers):
    resp = client.get("/api/v1/models", headers=operator_headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) >= 2
    assert any(m["id"] == "bitnet-b1.58-2b-i2s" for m in data)


def test_model_load_and_resident_status(operator_headers):
    resp = client.post(
        "/api/v1/models/load",
        json={"model_id": "bitnet-b1.58-2b-i2s", "prewarm_tensors": True},
        headers=operator_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_loaded"] is True
    assert resp.json()["memory_mb"] > 0


def test_completions_endpoint_latency_and_tokens(operator_headers):
    payload = {
        "model": "bitnet-b1.58-2b-i2s",
        "prompt": "Evaluate BitNet b1.58 inference speed",
        "max_tokens": 16,
        "temperature": 0.5,
        "activation": "relu_sqr",
    }
    resp = client.post("/api/v1/completions", json=payload, headers=operator_headers)
    assert resp.status_code == 200
    res = resp.json()

    assert res["object"] == "text_completion"
    assert len(res["choices"]) == 1
    assert len(res["choices"][0]["text"]) > 0
    assert res["usage"]["completion_tokens"] > 0
    assert res["latency_ms"] > 0
    assert res["tokens_per_second"] > 0


def test_chat_completions_endpoint(operator_headers):
    payload = {
        "model": "bitnet-b1.58-2b-i2s",
        "messages": [
            {"role": "system", "content": "You are a BitNet specialist."},
            {"role": "user", "content": "Explain Squared ReLU FFN graph."},
        ],
        "max_tokens": 20,
    }
    resp = client.post("/api/v1/chat/completions", json=payload, headers=operator_headers)
    assert resp.status_code == 200
    res = resp.json()

    assert res["object"] == "chat.completion"
    choice = res["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0


def test_quantize_endpoint_metrics(operator_headers):
    payload = {
        "rows": 32,
        "cols": 32,
        "seed": 100,
        "lut_group_size": 3,
    }
    resp = client.post("/api/v1/quantize", json=payload, headers=operator_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["rows"] == 32
    assert data["cols"] == 32
    assert data["compression_ratio"] > 1.0
    assert data["cosine_similarity"] > 0.8
    assert len(data["sample_weights"]) > 0


def test_unload_model_admin_role_enforced(operator_headers, admin_headers):
    # Operator cannot unload -> 403
    op_resp = client.post(
        "/api/v1/models/unload?model_id=bitnet-b1.58-small-lut",
        headers=operator_headers,
    )
    assert op_resp.status_code == 403

    # Admin can unload -> 200
    admin_resp = client.post(
        "/api/v1/models/unload?model_id=bitnet-b1.58-small-lut",
        headers=admin_headers,
    )
    assert admin_resp.status_code == 200
    assert "unloaded successfully" in admin_resp.json()["message"]
