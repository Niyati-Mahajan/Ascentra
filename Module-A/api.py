"""Lightweight API boundary for ASCENTRA Module A.

This service is intentionally separate from the teammate's Node.js backend. The
Node server can call POST /api/academic-risk and pass through the JSON response.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import academic_risk_prediction


class StudentAcademicProfile(BaseModel):
    student_id: str | None = None
    semester: int
    attendance_percentage: float
    ca1_score: float
    ca2_score: float
    ca3_score: float
    best_2_ca_average: float
    mid_term_score: float
    previous_semester_tgpa: float | None = Field(default=None)
    academic_trend: str
    use_gemini: bool = Field(default=False)


app = FastAPI(title="ASCENTRA Module A Academic Risk API")


@app.post("/api/academic-risk")
def academic_risk_endpoint(profile: StudentAcademicProfile) -> dict[str, Any]:
    try:
        payload = profile.model_dump()
    except AttributeError:
        payload = profile.dict()
    use_gemini = bool(payload.pop("use_gemini", False))
    try:
        return academic_risk_prediction(payload, use_gemini=use_gemini)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
