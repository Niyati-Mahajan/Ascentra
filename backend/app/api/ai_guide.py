from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.api.auth import get_current_user
from app.services.ai_guide_service import ai_guide_service

router = APIRouter(tags=["ai_guide"])

class GuideRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []
    context: Optional[Dict[str, Any]] = {}

@router.post("/ai/guide/legacy")
def ask_guide(req: GuideRequest, user: dict = Depends(get_current_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="A message is required.")
        
    result = ai_guide_service.process_query(user["profile"], req.message, req.history, req.context)
    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "evaluations": {
            "careerDirectionDelta": 20 if result["intent"] == "career_direction" else 10,
            "skillConfidenceDelta": 20 if result["intent"] == "skill_gap" else 10,
            "knowledgeSignalDelta": 15,
            "experienceDelta": 15
        },
        "detectedTargetRole": result.get("target_role")
    }
