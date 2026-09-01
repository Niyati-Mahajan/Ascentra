from fastapi import APIRouter, Depends, HTTPException
from app.api.auth import get_current_user
from app.storage.json_store import store
from typing import Dict, Any

router = APIRouter(prefix="/profile", tags=["profile"])

@router.get("")
def get_profile(user: dict = Depends(get_current_user)):
    return {"profile": user["profile"]}

@router.put("")
def update_full_profile(data: Dict[str, Any], user: dict = Depends(get_current_user)):
    store.update_user_profile(user["id"], data)
    return {"ok": True}

@router.post("/update")
def update_profile_partial(data: Dict[str, Any], user: dict = Depends(get_current_user)):
    profile = user["profile"]
    
    for key in ["full_name", "degree", "branch", "university", "year", "semester", "cgpa", "backlogs", "target_role"]:
        if key in data:
            profile[key] = data[key]
        elif key == "full_name" and "fullName" in data:
            profile["full_name"] = data["fullName"]
        elif key == "branch" and "department" in data:
            profile["branch"] = data["department"]
            
    if "skills" in data:
        profile["technical_skills"].update(data["skills"])
        
    store.update_user_profile(user["id"], profile)
    return {"ok": True, "profile": profile}

@router.post("/skills")
def add_skill(skill: str, level: int = 50, user: dict = Depends(get_current_user)):
    profile = user["profile"]
    profile["technical_skills"][skill] = level
    store.update_user_profile(user["id"], profile)
    return {"ok": True, "skills": profile["technical_skills"]}

@router.delete("/skills/{skill_name}")
def delete_skill(skill_name: str, user: dict = Depends(get_current_user)):
    profile = user["profile"]
    if skill_name in profile["technical_skills"]:
        del profile["technical_skills"][skill_name]
        store.update_user_profile(user["id"], profile)
    return {"ok": True, "skills": profile["technical_skills"]}
