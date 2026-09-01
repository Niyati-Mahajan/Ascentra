from fastapi import APIRouter, Depends, HTTPException, Request, Cookie
import base64
import json
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.api.auth import get_current_user
from app.services.student_context import student_context_builder
from app.ai.career_guide import career_guide_ai
from app.ai.career_guide import AIUpstreamError
from app.services.ai_guide_service import ai_guide_service

router = APIRouter(tags=["career_guide_ai"])

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []
    context: Optional[Dict[str, Any]] = {}

@router.get("/ai/career-guide/health")
def health_check():
    health = career_guide_ai.health()
    return {
        "backend": "ok",
        "service": "career-guide",
        "provider": "gemini" if health["ai_configured"] else "none",
        **health,
    }

@router.post("/ai/career-guide/probe")
def ai_probe(request: Request, ascentra: Optional[str] = Cookie(None)):
    try:
        forwarded = request.headers.get("x-ascentra-user")
        if forwarded and request.client and request.client.host in {"127.0.0.1", "::1"}:
            user = json.loads(base64.b64decode(forwarded).decode("utf-8"))
            if not user.get("id"):
                raise ValueError("Missing authenticated user")
        else:
            get_current_user(request, ascentra)
    except Exception:
        raise HTTPException(status_code=401, detail={"ok": False, "error": "AUTH_ERROR"})
    try:
        return {"ok": True, **career_guide_ai.probe()}
    except AIUpstreamError as error:
        raise HTTPException(status_code=503, detail={"ok": False, "error": error.category, "retryable": error.retryable})

@router.post("/ai/career-guide/chat")
@router.post("/ai/guide")
@router.post("/career-guide")
def career_guide_chat(req: ChatRequest, request: Request, ascentra: Optional[str] = Cookie(None)):
    msg = req.message.strip()
    if not msg or len(msg) > 4000:
        raise HTTPException(status_code=400, detail="A message is required.")

    # Node validates the session, then forwards the authenticated profile only
    # over the loopback proxy. Direct callers must use FastAPI authentication.
    try:
        forwarded = request.headers.get("x-ascentra-user")
        if forwarded and request.client and request.client.host in {"127.0.0.1", "::1"}:
            user = json.loads(base64.b64decode(forwarded).decode("utf-8"))
            if not user.get("id"):
                raise ValueError("Missing authenticated user")
        else:
            user = get_current_user(request, ascentra)
    except Exception:
        raise HTTPException(status_code=401, detail="Signed out")

    detected_intent = ai_guide_service.detect_intent(msg, req.history)
    has_history = bool(req.history)
    has_context = bool(req.context)
    student_ctx = student_context_builder.build_ascentra_context(
        user,
        msg,
        req.history,
        req.context,
        detected_intent,
    )
    section_names = student_ctx.get("selected_context_sections", [])

    request_id = request.headers.get("x-request-id") or "-"
    try:
        print(f"[CareerGuideAI] request_id={request_id} endpoint=/api/ai/career-guide/chat status=start intent={detected_intent} sections={section_names} history_count={len(req.history or [])} live_context={has_context}")
        result = career_guide_ai.generate_response(student_ctx, msg, req.history, request_id=request_id)
        response_source = "scope_guardrail" if result.get("model_used") == "scope_guardrail" else "gemini"
        print(f"[CareerGuideAI] request_id={request_id} endpoint=/api/ai/career-guide/chat status=complete intent={detected_intent} scope={result.get('scope', 'unknown')} model={result.get('model_used')} fallback_used={'yes' if result.get('fallback_used') else 'no'}")
        return {
            "answer": result["answer"],
            "intent": detected_intent,
            "response_source": response_source,
            "context_sections": section_names,
            "detectedTargetRole": (student_ctx.get("target_role") or {}).get("id") or (student_ctx.get("target_role") or {}).get("name"),
            "model_used": result.get("model_used"),
            "fallback_used": result.get("fallback_used"),
            "evaluations": {
                "careerDirectionDelta": 15,
                "skillConfidenceDelta": 15,
                "knowledgeSignalDelta": 15,
                "experienceDelta": 15
            }
        }
    except AIUpstreamError as error:
        print(f"[CareerGuideAI] request_id={request_id} endpoint=/api/ai/career-guide/chat status=failure category={error.category} retryable={'yes' if error.retryable else 'no'}")
        status_code = error.status_code or (503 if error.retryable or error.category in {"CONFIG_ERROR", "MODEL_UNAVAILABLE"} else 500)
        raise HTTPException(
            status_code=status_code,
            detail={"ok": False, "error": error.category, "retryable": error.retryable, "ai_status": career_guide_ai.health()["ai_status"]},
        )
    except Exception:
        print(f"[CareerGuideAI] request_id={request_id} endpoint=/api/ai/career-guide/chat status=failure category=LOCAL_SERVER_ERROR")
        raise HTTPException(status_code=500, detail={"ok": False, "error": "LOCAL_SERVER_ERROR", "retryable": False})
