import json
import socket
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app.ai.career_guide import AIUpstreamError, CareerGuideAIService
from app.config import settings
from app.main import app


def test_ai_health_reports_configuration_without_secret(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "", raising=False)
    response = TestClient(app).get("/api/ai/career-guide/health")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "ok"
    assert data["ai_configured"] is False
    assert data["ai_status"] == "unavailable"
    assert "key" not in json.dumps(data).lower()


def test_primary_retry_then_fallback_model(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    monkeypatch.setattr(settings, "AI_CIRCUIT_BREAKER_FAILURES", 99, raising=False)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    service = CareerGuideAIService()
    attempts = []

    def fake_request(model, payload):
        attempts.append(model)
        if model == "primary-model":
            raise AIUpstreamError("UPSTREAM_5XX", "temporary", status_code=503, retryable=True)
        return "Gemini fallback answer"

    monkeypatch.setattr(service, "_request_model", fake_request)
    result = service.generate_response({"target_role": {"name": "Full Stack Developer"}}, "hi", [], request_id="test")
    assert attempts == ["primary-model", "primary-model", "fallback-model"]
    assert result["answer"] == "Gemini fallback answer"
    assert result["model_used"] == "fallback-model"
    assert result["fallback_used"] is True


def test_primary_rate_limit_then_fallback_success_keeps_ai_ready(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    monkeypatch.setattr(settings, "AI_CIRCUIT_BREAKER_FAILURES", 3, raising=False)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    service = CareerGuideAIService()
    attempts = []

    def fake_request(model, payload):
        attempts.append(model)
        if model == "primary-model":
            raise AIUpstreamError("RATE_LIMIT", "Gemini HTTP 429", status_code=429, retryable=True, retry_after_seconds=37)
        return "Gemini fallback answer"

    monkeypatch.setattr(service, "_request_model", fake_request)
    result = service.generate_response({}, "hi", [], request_id="rate-limit-fallback")
    health = service.health()
    assert attempts == ["primary-model", "fallback-model"]
    assert result["model_used"] == "fallback-model"
    assert result["fallback_used"] is True
    assert health["ai_status"] == "ready"
    assert health["consecutive_failures"] == 0
    assert health["model_statuses"]["primary-model"]["status"] == "rate_limited"
    assert health["model_statuses"]["fallback-model"]["status"] == "ready"


def test_non_retryable_auth_error_does_not_try_fallback(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    service = CareerGuideAIService()
    attempts = []

    def fake_request(model, payload):
        attempts.append(model)
        raise AIUpstreamError("AUTH_ERROR", "bad key", status_code=403, retryable=False)

    monkeypatch.setattr(service, "_request_model", fake_request)
    with pytest.raises(AIUpstreamError) as exc:
        service.generate_response({}, "hi", [], request_id="test")
    assert exc.value.category == "AUTH_ERROR"
    assert attempts == ["primary-model"]


def test_timeout_and_empty_response_are_structured(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    service = CareerGuideAIService()
    timeout = socket.timeout("slow")
    assert isinstance(timeout, socket.timeout)
    with pytest.raises(AIUpstreamError) as exc:
        raise AIUpstreamError("TIMEOUT", "Gemini request timed out.", retryable=True)
    assert exc.value.category == "TIMEOUT"
    with pytest.raises(AIUpstreamError) as empty:
        raise AIUpstreamError("INVALID_RESPONSE", "Gemini returned an empty answer.", retryable=True)
    assert empty.value.category == "INVALID_RESPONSE"


def test_circuit_breaker_marks_degraded(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    monkeypatch.setattr(settings, "AI_CIRCUIT_BREAKER_FAILURES", 3, raising=False)
    monkeypatch.setattr(settings, "AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 45, raising=False)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    service = CareerGuideAIService()

    def fail(model, payload):
        raise AIUpstreamError("UPSTREAM_5XX", "temporary", status_code=503, retryable=True)

    monkeypatch.setattr(service, "_request_model", fail)
    for index in range(3):
        with pytest.raises(AIUpstreamError):
            service.generate_response({}, "hi", [], request_id=f"test-{index}")
    assert service.health()["ai_status"] == "degraded"
    assert service.health()["consecutive_failures"] == 3


def test_expired_cooldown_allows_recovery_probe(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    service = CareerGuideAIService()
    service.consecutive_failures = 6
    service.degraded_until = 1.0
    attempts = []

    def succeed(model, payload):
        attempts.append(model)
        return "OK"

    monkeypatch.setattr(service, "_request_model", succeed)
    result = service.generate_response({}, "Reply only with OK", [], request_id="recovery")
    assert result["answer"] == "OK"
    assert attempts == ["primary-model"]
    assert service.health()["ai_status"] == "ready"
    assert service.health()["consecutive_failures"] == 0
    assert service.health()["last_ai_error_category"] is None


def test_success_resets_stale_failure_state(monkeypatch):
    monkeypatch.setattr(settings, "AI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "PRIMARY_LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "fallback-model", raising=False)
    service = CareerGuideAIService()
    service.consecutive_failures = 2
    service.last_ai_failure_category = "NETWORK_ERROR"

    monkeypatch.setattr(service, "_request_model", lambda model, payload: "Recovered")
    service.generate_response({}, "hi", [], request_id="success-reset")
    assert service.health()["consecutive_failures"] == 0
    assert service.health()["ai_status"] == "ready"
    assert service.health()["last_ai_error_category"] is None


def test_404_is_model_not_found_not_circuit_unavailable():
    service = CareerGuideAIService()
    assert service._http_category(404) == "MODEL_NOT_FOUND"


def test_chat_route_preserves_upstream_status_code(monkeypatch):
    from app.routes import ai as ai_routes

    def fail(*args, **kwargs):
        raise AIUpstreamError("RATE_LIMIT", "Gemini HTTP 429", status_code=429, retryable=True, model="primary-model")

    monkeypatch.setattr(ai_routes, "get_current_user", lambda request, ascentra: {"id": "u1", "profile": {}})
    monkeypatch.setattr(ai_routes.career_guide_ai, "generate_response", fail)
    response = TestClient(app).post(
        "/api/ai/career-guide/chat",
        json={"message": "hi", "history": [], "context": {}},
    )
    assert response.status_code == 429
    assert response.json()["detail"]["error"] == "RATE_LIMIT"
