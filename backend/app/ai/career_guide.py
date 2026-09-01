import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ai.prompts import SYSTEM_PROMPT
from app.ai.domain_scope import OUT_OF_SCOPE, build_scope_redirect, classify_domain_scope


RETRYABLE_CATEGORIES = {"RATE_LIMIT", "MODEL_UNAVAILABLE", "UPSTREAM_5XX", "TIMEOUT", "NETWORK_ERROR", "INVALID_RESPONSE"}
NON_RETRYABLE_HTTP = {400: "LOCAL_SERVER_ERROR", 401: "AUTH_ERROR", 403: "AUTH_ERROR", 404: "MODEL_NOT_FOUND"}


@dataclass
class AIUpstreamError(Exception):
    category: str
    message: str
    status_code: Optional[int] = None
    retryable: bool = False
    model: Optional[str] = None
    retry_after_seconds: Optional[float] = None

    def __str__(self) -> str:
        return self.message


class CareerGuideAIService:
    def __init__(self):
        self.last_ai_success_at: Optional[float] = None
        self.last_ai_failure_at: Optional[float] = None
        self.last_ai_failure_category: Optional[str] = None
        self.consecutive_failures = 0
        self.degraded_until = 0.0
        self.model_statuses: Dict[str, Dict[str, Any]] = {}

    @property
    def is_configured(self) -> bool:
        return self.config_status()["configured"]

    def config_status(self) -> Dict[str, Any]:
        return {
            "configured": bool(settings.AI_API_KEY and settings.AI_API_KEY.strip() and not settings.AI_API_KEY.startswith("YOUR_")),
            "primary_model": str(settings.PRIMARY_LLM_MODEL or "").strip(),
            "fallback_model": str(settings.FALLBACK_LLM_MODEL or "").strip(),
            "client_ready": urllib.request is not None,
        }

    def health(self) -> Dict[str, Any]:
        cfg = self.config_status()
        configured = bool(cfg["configured"] and cfg["primary_model"] and cfg["fallback_model"] and cfg["client_ready"])
        self._refresh_circuit()
        cooldown_remaining = max(0.0, self.degraded_until - time.time())
        status = "unavailable" if not configured else ("degraded" if cooldown_remaining > 0 else "ready")
        model_statuses = {}
        for model in self._configured_models():
            model_statuses[model] = self.model_statuses.get(model, {"status": "unknown", "last_error_category": None, "last_checked_at": None})
        return {
            "ai_configured": configured,
            "primary_model": cfg["primary_model"],
            "fallback_model": cfg["fallback_model"],
            "ai_status": status,
            "model_statuses": model_statuses,
            "cooldown_remaining_seconds": int(cooldown_remaining),
            "last_ai_error_category": self.last_ai_failure_category,
            "last_ai_success_at": self.last_ai_success_at,
            "last_ai_failure_at": self.last_ai_failure_at,
            "consecutive_failures": self.consecutive_failures,
        }

    def startup_summary(self) -> Dict[str, str]:
        h = self.health()
        return {
            "configured": "yes" if h["ai_configured"] else "no",
            "primary_model": h["primary_model"] or "missing",
            "fallback_model": h["fallback_model"] or "missing",
        }

    def generate_response(
        self,
        student_context: Dict[str, Any],
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        scope_decision = classify_domain_scope(user_message, history or [], student_context)
        if scope_decision.scope == OUT_OF_SCOPE:
            return {
                "answer": build_scope_redirect(user_message, student_context),
                "model_used": "scope_guardrail",
                "fallback_used": False,
                "latency_ms": int((time.time() - started) * 1000),
                "scope": scope_decision.scope,
            }

        if not self.is_configured:
            raise AIUpstreamError("CONFIG_ERROR", "ASCENTRA AI is not configured.", retryable=False)
        self._refresh_circuit()
        if time.time() < self.degraded_until:
            raise AIUpstreamError("MODEL_UNAVAILABLE", "ASCENTRA AI is cooling down after repeated upstream failures.", retryable=True)

        payload = self._build_payload(student_context, user_message, history or [], scope_decision.instruction)
        errors: List[AIUpstreamError] = []
        models = self._configured_models()
        schedule = [(models[0], 1, 0), (models[0], 2, 0.6)]
        if len(models) > 1:
            schedule.append((models[1], 1, 1.2))

        skip_retry_models = set()
        for model, attempt, delay in schedule:
            if model in skip_retry_models:
                continue
            if delay:
                time.sleep(delay + random.uniform(0, 0.18))
            attempt_started = time.time()
            try:
                answer = self._request_model(model, payload)
                self._record_model_success(model)
                self._record_success()
                fallback_used = model != models[0]
                print(
                    f"[CareerGuideAI] request_id={request_id or '-'} endpoint=/api/ai/career-guide/chat "
                    f"model={model} status=success category=OK latency_ms={int((time.time()-attempt_started)*1000)} "
                    f"fallback_used={'yes' if fallback_used else 'no'}"
                )
                return {
                    "answer": answer,
                    "model_used": model,
                    "fallback_used": fallback_used,
                    "latency_ms": int((time.time() - started) * 1000),
                    "scope": scope_decision.scope,
                }
            except AIUpstreamError as error:
                error.model = model
                errors.append(error)
                self._record_model_failure(model, error)
                if error.category == "RATE_LIMIT" and (error.retry_after_seconds is None or error.retry_after_seconds > 2):
                    skip_retry_models.add(model)
                print(
                    f"[CareerGuideAI] request_id={request_id or '-'} endpoint=/api/ai/career-guide/chat "
                    f"model={model} status=failure category={error.category} latency_ms={int((time.time()-attempt_started)*1000)} "
                    f"fallback_used={'yes' if model != models[0] else 'no'}"
                )
                if not error.retryable:
                    break

        last = errors[-1] if errors else AIUpstreamError("LOCAL_SERVER_ERROR", "ASCENTRA AI request failed.", retryable=True)
        self._record_failure(last.category)
        raise AIUpstreamError(last.category, "ASCENTRA AI is temporarily unavailable.", status_code=last.status_code, retryable=last.retryable, model=last.model, retry_after_seconds=last.retry_after_seconds)

    def probe(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        result = self.generate_response({"diagnostic": "connectivity_probe"}, "Reply with OK.", [], request_id)
        return {"ok": result["answer"].strip().upper().startswith("OK"), **result}

    def _build_payload(self, student_context: Dict[str, Any], user_message: str, history: List[Dict[str, Any]], scope_instruction: Optional[str] = None) -> Dict[str, Any]:
        compact_context = self._trim_context(student_context)
        system_instruction = SYSTEM_PROMPT + "\n\nASCENTRA STUDENT CONTEXT:\n" + json.dumps(compact_context, separators=(",", ":"))
        if scope_instruction:
            system_instruction += "\n\nSCOPE ROUTING NOTE:\n" + scope_instruction
        contents = []
        for msg in history[-12:]:
            role = "user" if msg.get("role") in ["user", "student"] else "model"
            text = str(msg.get("content") or msg.get("message") or "")[:1800]
            if text:
                contents.append({"role": role, "parts": [{"text": text}]})
        contents.append({"role": "user", "parts": [{"text": user_message[:4000]}]})
        return {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800},
        }

    def _trim_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raw = json.dumps(context, default=str)
        if len(raw) <= settings.AI_MAX_CONTEXT_CHARS:
            return context
        trimmed = json.loads(raw)
        resume = trimmed.get("resume_evidence") or trimmed.get("resume") or {}
        if isinstance(resume, dict):
            parsed = resume.get("parsed") or {}
            if isinstance(parsed, dict) and "text" in parsed:
                parsed["text"] = str(parsed["text"])[:settings.AI_MAX_RESUME_TEXT_CHARS]
            if "text" in resume:
                resume["text"] = str(resume["text"])[:settings.AI_MAX_RESUME_TEXT_CHARS]
            if "projects" in resume and isinstance(resume["projects"], list):
                resume["projects"] = resume["projects"][:6]
        if isinstance(trimmed.get("role_compatibility"), list):
            trimmed["role_compatibility"] = trimmed["role_compatibility"][:8]
        if isinstance(trimmed.get("company_opportunities"), list):
            trimmed["company_opportunities"] = trimmed["company_opportunities"][:12]
        return trimmed

    def _configured_models(self) -> List[str]:
        models = [settings.PRIMARY_LLM_MODEL, settings.FALLBACK_LLM_MODEL]
        clean = []
        for model in models:
            model = str(model or "").strip()
            if model and model not in clean:
                clean.append(model)
        return clean

    def _request_model(self, model: str, payload: Dict[str, Any]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.AI_API_KEY}"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=settings.AI_TIMEOUT_SECONDS) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            category = self._http_category(error.code)
            retry_after = self._retry_after_seconds(error.headers.get("Retry-After"), raw)
            message = f"Gemini HTTP {error.code}"
            if category == "RATE_LIMIT" and retry_after:
                message = f"{message}; retry_after_seconds={int(retry_after)}"
            raise AIUpstreamError(category, message, error.code, category in RETRYABLE_CATEGORIES, retry_after_seconds=retry_after)
        except TimeoutError:
            raise AIUpstreamError("TIMEOUT", "Gemini request timed out.", retryable=True)
        except socket.timeout:
            raise AIUpstreamError("TIMEOUT", "Gemini request timed out.", retryable=True)
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", error)
            category = "TIMEOUT" if isinstance(reason, socket.timeout) else "NETWORK_ERROR"
            raise AIUpstreamError(category, f"Gemini network request failed: {type(reason).__name__}: {reason}", retryable=True)
        except json.JSONDecodeError:
            raise AIUpstreamError("INVALID_RESPONSE", "Gemini returned invalid JSON.", retryable=True)

        candidates = resp_json.get("candidates") if isinstance(resp_json, dict) else None
        if not candidates:
            raise AIUpstreamError("INVALID_RESPONSE", "Gemini returned no candidates.", retryable=True)
        finish_reason = candidates[0].get("finishReason")
        if finish_reason and finish_reason not in {"STOP", "MAX_TOKENS"}:
            raise AIUpstreamError("INVALID_RESPONSE", f"Gemini finish reason was {finish_reason}.", retryable=finish_reason != "SAFETY")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not answer:
            raise AIUpstreamError("INVALID_RESPONSE", "Gemini returned an empty answer.", retryable=True)
        return answer

    def _http_category(self, status_code: int) -> str:
        if status_code == 429:
            return "RATE_LIMIT"
        if status_code in {500, 502, 503, 504}:
            return "UPSTREAM_5XX"
        return NON_RETRYABLE_HTTP.get(status_code, "LOCAL_SERVER_ERROR")

    def _record_success(self) -> None:
        self.last_ai_success_at = time.time()
        self.last_ai_failure_category = None
        self.consecutive_failures = 0
        self.degraded_until = 0.0
        for model in self._configured_models():
            if self.model_statuses.get(model, {}).get("last_error_category") != "RATE_LIMIT":
                continue
            self.model_statuses[model]["status"] = "rate_limited"

    def _record_failure(self, category: str) -> None:
        self._refresh_circuit()
        self.last_ai_failure_at = time.time()
        self.last_ai_failure_category = category
        if category in RETRYABLE_CATEGORIES:
            self.consecutive_failures += 1
            if self.consecutive_failures >= settings.AI_CIRCUIT_BREAKER_FAILURES:
                self.degraded_until = time.time() + settings.AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS
        else:
            self.consecutive_failures = 0

    def _record_model_success(self, model: str) -> None:
        self.model_statuses[model] = {"status": "ready", "last_error_category": None, "last_checked_at": time.time()}

    def _record_model_failure(self, model: str, error: AIUpstreamError) -> None:
        status = "rate_limited" if error.category == "RATE_LIMIT" else ("unavailable" if error.category in {"MODEL_NOT_FOUND", "AUTH_ERROR"} else "degraded")
        self.model_statuses[model] = {
            "status": status,
            "last_error_category": error.category,
            "last_status_code": error.status_code,
            "retry_after_seconds": error.retry_after_seconds,
            "last_checked_at": time.time(),
        }

    def _refresh_circuit(self) -> None:
        if self.degraded_until and time.time() >= self.degraded_until:
            self.degraded_until = 0.0
            self.consecutive_failures = 0

    def _retry_after_seconds(self, header: Optional[str], body: str) -> Optional[float]:
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        try:
            parsed = json.loads(body or "{}")
            for detail in parsed.get("error", {}).get("details", []):
                retry_delay = detail.get("retryDelay") if isinstance(detail, dict) else None
                if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                    return float(retry_delay[:-1])
        except (ValueError, TypeError, AttributeError):
            return None
        return None


career_guide_ai = CareerGuideAIService()
