from fastapi import APIRouter, Depends
from app.api.auth import get_current_user
from app.services.recommendation_service import recommendation_service
from app.ml.predictor import placement_10k_predictor
from typing import Optional

router = APIRouter(tags=["intelligence"])

@router.get("/roles")
def get_roles():
    return {
        "roles": recommendation_service.get_all_roles(),
        "provenance": "Curated technology roles knowledge base."
    }

@router.post("/intelligence/placement-readiness")
@router.get("/readiness")
def get_placement_readiness(user: dict = Depends(get_current_user)):
    result = placement_10k_predictor.predict(user["profile"])
    return result

@router.post("/intelligence/role-match")
def get_role_match(target_role: Optional[str] = None, user: dict = Depends(get_current_user)):
    match_data = recommendation_service.calculate_role_match(user["profile"], target_role)
    return match_data
