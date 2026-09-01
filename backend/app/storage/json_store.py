import os
import json
import uuid
import hashlib
from typing import Dict, Any, Optional, List
from app.config import settings
from app.storage.schemas import UserRecord, StudentProfile

# Migrate root data.json if exists or storage_data/users.json
ROOT_DB = "c:/Ascentra/data.json"

class JSONStore:
    def __init__(self):
        self.file_path = settings.USERS_FILE
        self.sessions: Dict[str, str] = {} # token -> user_id
        self._ensure_init()

    def _hash_password(self, password: str) -> str:
        salt = os.urandom(16).hex()
        derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        return f"{salt}:{derived}"

    def _verify_password(self, password: str, hashed: str) -> bool:
        try:
            salt, v = hashed.split(':')
            derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            return derived == v
        except Exception:
            return False

    def _ensure_init(self):
        if not os.path.exists(self.file_path):
            initial_data = {"users": []}
            if os.path.exists(ROOT_DB):
                try:
                    with open(ROOT_DB, 'r', encoding='utf-8') as f:
                        old_data = json.load(f)
                    for u in old_data.get("users", []):
                        uname = u.get("username") or u.get("name") or "student"
                        full_name = u.get("name") or uname
                        prof = u.get("profile") or {}
                        student_data = prof.get("student") or {}
                        
                        p = StudentProfile(
                            full_name=student_data.get("name") or full_name,
                            degree=student_data.get("degree") or "",
                            branch=student_data.get("department") or "",
                            university=student_data.get("university") or "",
                            semester=student_data.get("semester"),
                            cgpa=student_data.get("cgpa"),
                            backlogs=student_data.get("backlogs") or 0,
                            technical_skills=student_data.get("skills") or {},
                            projects=student_data.get("projects") or [],
                            target_role=student_data.get("target"),
                            roadmap=student_data.get("roadmap") or {},
                            resume=prof.get("resume") or {}
                        )
                        initial_data["users"].append({
                            "id": u.get("id", str(uuid.uuid4())),
                            "username": uname,
                            "email": u.get("email", "").lower(),
                            "password_hash": u.get("password", ""),
                            "profile": p.dict(),
                            "has_logged_in": u.get("hasLoggedIn", True)
                        })
                except Exception as e:
                    print("Error migrating ROOT_DB:", e)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)

    def read_all(self) -> Dict[str, Any]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_all(self, data: Dict[str, Any]):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        # Also sync to root data.json for node/frontend legacy compatibility if needed
        sync_root = {"users": []}
        for u in data.get("users", []):
            sync_root["users"].append({
                "id": u["id"],
                "username": u["username"],
                "name": u["profile"]["full_name"] or u["username"],
                "email": u["email"],
                "password": u["password_hash"],
                "profile": {
                    "student": {
                        "name": u["profile"]["full_name"] or u["username"],
                        "department": u["profile"]["branch"],
                        "university": u["profile"]["university"],
                        "degree": u["profile"]["degree"],
                        "semester": u["profile"]["semester"],
                        "cgpa": u["profile"]["cgpa"],
                        "backlogs": u["profile"]["backlogs"],
                        "target": u["profile"]["target_role"],
                        "skills": u["profile"]["technical_skills"],
                        "projects": u["profile"]["projects"],
                        "roadmap": u["profile"]["roadmap"]
                    },
                    "resume": u["profile"].get("resume", {})
                },
                "hasLoggedIn": u.get("has_logged_in", True)
            })
        with open(ROOT_DB, 'w', encoding='utf-8') as f:
            json.dump(sync_root, f, indent=2)

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        db = self.read_all()
        for u in db.get("users", []):
            if u["email"].lower() == email.lower():
                return u
        return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        db = self.read_all()
        for u in db.get("users", []):
            if u["username"].lower() == username.lower():
                return u
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        db = self.read_all()
        for u in db.get("users", []):
            if u["id"] == user_id:
                return u
        return None

    def create_user(self, username: str, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        db = self.read_all()
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        profile = StudentProfile(full_name=full_name or username).dict()
        user_rec = {
            "id": user_id,
            "username": username,
            "email": email.lower(),
            "password_hash": password_hash,
            "profile": profile,
            "has_logged_in": True
        }
        db["users"].append(user_rec)
        self.save_all(db)
        return user_rec

    def update_user_profile(self, user_id: str, profile_dict: Dict[str, Any]) -> bool:
        db = self.read_all()
        for u in db["users"]:
            if u["id"] == user_id:
                u["profile"].update(profile_dict)
                self.save_all(db)
                return True
        return False

    def create_session(self, user_id: str) -> str:
        token = os.urandom(32).hex()
        self.sessions[token] = user_id
        return token

    def get_user_from_session(self, token: str) -> Optional[Dict[str, Any]]:
        user_id = self.sessions.get(token)
        if user_id:
            return self.get_user_by_id(user_id)
        return None

    def delete_session(self, token: str):
        if token in self.sessions:
            del self.sessions[token]

store = JSONStore()
