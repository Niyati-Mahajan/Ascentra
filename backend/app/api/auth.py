from fastapi import APIRouter, HTTPException, Response, Request, Depends, Cookie
from app.storage.json_store import store
from app.storage.schemas import UserRegister, UserLogin
from typing import Optional

router = APIRouter(prefix="/auth", tags=["auth"])

def get_current_user(request: Request, ascentra: Optional[str] = Cookie(None)):
    token = ascentra or request.cookies.get("ascentra")
    if not token:
        raise HTTPException(status_code=401, detail="Signed out")
    user = store.get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return user

@router.post("/register")
def register(data: UserRegister, response: Response):
    username = data.username.strip()
    email = data.email.strip().lower()
    
    if store.get_user_by_email(email) or store.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="An account already uses this email or username.")
        
    user = store.create_user(username, email, data.password, full_name=data.name or username)
    token = store.create_session(user["id"])
    
    max_age = 2592000 if data.remember else None
    response.set_cookie(key="ascentra", value=token, httponly=True, samesite="lax", max_age=max_age)
    
    return {
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
        "onboarding": False,
        "firstLogin": True
    }

@router.post("/login")
def login(data: UserLogin, response: Response):
    user = store.get_user_by_email(data.email)
    if not user or not store._verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        
    token = store.create_session(user["id"])
    max_age = 2592000 if data.remember else None
    response.set_cookie(key="ascentra", value=token, httponly=True, samesite="lax", max_age=max_age)
    
    return {
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
        "onboarding": bool(user["profile"].get("assessment")),
        "profile": user["profile"],
        "firstLogin": not user.get("has_logged_in", False)
    }

@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "user": {"id": current_user["id"], "username": current_user["username"], "email": current_user["email"]},
        "profile": current_user["profile"]
    }

@router.post("/logout")
def logout(response: Response, ascentra: Optional[str] = Cookie(None)):
    if ascentra:
        store.delete_session(ascentra)
    response.delete_cookie("ascentra")
    return {"ok": True}
