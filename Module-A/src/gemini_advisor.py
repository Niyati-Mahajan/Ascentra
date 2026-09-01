"""Optional Gemini-powered advisor layer for ASCENTRA Module A."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - handled at runtime for optional feature.
    genai = None
    types = None

try:
    from .config import GEMINI_MODEL
except ImportError:
    from config import GEMINI_MODEL


def gemini_available() -> bool:
    load_dotenv()
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": prediction.get("student_id"),
        "risk_probability": prediction.get("risk_probability"),
        "risk_level": prediction.get("risk_level"),
        "decision": prediction.get("decision"),
        "risk_factors": prediction.get("risk_factors", [])[:5],
        "protective_factors": prediction.get("protective_factors", [])[:5],
        "rule_based_interventions": prediction.get("interventions", [])[:5],
    }


def _parse_gemini_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    elif "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"advisor_summary": parsed}
    except json.JSONDecodeError:
        return {"advisor_summary": text.strip()}


def generate_gemini_advice(prediction: dict[str, Any], model_name: str = GEMINI_MODEL) -> dict[str, Any]:
    """Generate advisor-facing intervention language from the model/SHAP output.

    Gemini is not used for model scoring. It only rewrites and organizes advice
    from already-computed risk probability, SHAP factors, and rule interventions.
    """
    if genai is None or types is None:
        return {
            "enabled": False,
            "source": "local_fallback",
            "error": "google-genai is not installed.",
        }

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "source": "local_fallback",
            "error": "GEMINI_API_KEY is not configured.",
        }

    prompt_payload = _compact_prediction(prediction)
    prompt = f"""
You are ASCENTRA's Academic Risk Advisor.

Use only the provided model output. Do not invent new student facts, do not mention LMS activity,
assignments, late submissions, or quiz averages, and do not override the model probability.

Return concise JSON with these keys:
- advisor_summary
- immediate_actions
- priority_interventions
- follow_up_plan
- monitoring_metrics
- escalation_condition

Every priority intervention must be linked to one of the provided risk_factors.

Model output:
{json.dumps(prompt_payload, indent=2)}
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=(
                    "You produce academic advising recommendations grounded only in supplied SHAP factors."
                ),
            ),
        )
        text = response.text or ""
        return {
            "enabled": True,
            "source": "gemini",
            "model": model_name,
            "advice": _parse_gemini_text(text),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "source": "local_fallback",
            "error": str(exc),
        }
