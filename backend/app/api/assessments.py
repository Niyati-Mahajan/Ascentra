from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.auth import get_current_user
from app.services.assessment_service import assessment_service
from typing import Dict, Any

router = APIRouter(tags=["assessments"])

class SubmitTestRequest(BaseModel):
    answers: Dict[str, int]

@router.get("/weekly-test")
def get_weekly_test(user: dict = Depends(get_current_user)):
    return {"questions": assessment_service.get_weekly_test(user["id"])}

@router.post("/weekly-test/submit")
def submit_weekly_test(req: SubmitTestRequest, user: dict = Depends(get_current_user)):
    result = assessment_service.submit_weekly_test(user["id"], req.answers)
    return {"ok": True, "result": result}
