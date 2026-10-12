"""Tests covering benchmark execution, compiler diagnostics, and advisory service."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.security import create_demo_token
from app.main import app
from app.models.database import db
from app.services.benchmark_service import benchmark_service
from app.services.diagnostics_service import diagnostics_service
from app.services.llm_advisory import llm_advisory_service

client = TestClient(app)


@pytest.fixture
def operator_headers():
    token = create_demo_token(role="operator")
    return {"Authorization": f"Bearer {token}"}


def test_run_perplexity_benchmark_service():
    res = benchmark_service.run_perplexity_benchmark(num_layers=10)
    assert res["relu_sqr_perplexity"] < res["silu_perplexity"]
    assert res["perplexity_gap"] > 0


def test_run_quantization_benchmark_service():
    res = benchmark_service.run_quantization_benchmark(dim=64)
    assert res["compression_ratio"] > 1.0
    assert res["cosine_similarity"] > 0.8
    assert res["total_elements"] == 4096


def test_run_throughput_benchmark_service():
    res = benchmark_service.run_throughput_benchmark(dim=64)
    assert res["additions_count"] > 0
    assert res["speedup_5ary_vs_3ary"] > 0


def test_evaluate_compiler_matrix_clang20_hazard():
    res = diagnostics_service.evaluate_compiler_matrix(
        compiler_name="clang",
        compiler_version="20.1.0",
        target_arch="x86_64",
        cflags=["-O3", "-Wunreachable-code-break"],
    )
    assert res.safe_to_compile is False
    codes = [h.issue_code for h in res.hazards]
    assert "CLANG20-CONST-CORRECTNESS" in codes
    assert "CMAKE-UNRECOGNIZED-FLAG" in codes


def test_evaluate_compiler_matrix_arm_hazard():
    res = diagnostics_service.evaluate_compiler_matrix(
        compiler_name="gcc",
        compiler_version="13.2.0",
        target_arch="aarch64",
        cflags=["-O3"],
    )
    codes = [h.issue_code for h in res.hazards]
    assert "ARM-NEON-TYPE-CONVERSION" in codes


def test_audit_logs_pagination_and_filter(operator_headers):
    # Log known actions
    db.log_audit("user-test-1", "TEST_ACTION_A", "res1", "SUCCESS")
    db.log_audit("user-test-2", "TEST_ACTION_B", "res2", "FAILURE")

    resp = client.get(
        "/api/v1/audit/logs?page=1&page_size=10&action=TEST_ACTION_A",
        headers=operator_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert all(item["action"] == "TEST_ACTION_A" for item in data["data"])


def test_llm_advisory_offline_deterministic():
    """Verify deterministic offline advisory produces report when no API key is provided."""
    import asyncio

    summary = {
        "metric": "Perplexity",
        "relu_sqr_ppl": "17.1",
        "silu_ppl": "99.8",
    }
    report = asyncio.run(llm_advisory_service.generate_advisory_report(summary))
    assert report.is_advisory_only is True
    assert report.provider == "offline-deterministic"
    assert "Squared ReLU" in report.analysis_text
    assert len(report.grounded_findings) > 0


def test_llm_advisory_mocked_provider_request(monkeypatch):
    """Verify provider adapter path with mocked network response."""
    import asyncio

    monkeypatch.setattr("app.core.config.settings.llm_api_key", "test-synthetic-key")
    monkeypatch.setattr("app.core.config.settings.llm_provider", "openai-compatible")

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json = lambda: {
        "choices": [{"message": {"content": "Mocked expert BitNet evaluation."}}]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        summary = {"metric": "Throughput"}
        report = asyncio.run(llm_advisory_service.generate_advisory_report(summary))
        assert report.provider == "openai-compatible"
        assert report.analysis_text == "Mocked expert BitNet evaluation."
