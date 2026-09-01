"""Authenticated resume extraction and persistence routes.

This module restores the router expected by ``app.main``.  It reuses the
existing parser and JSON store; it does not train or alter any ML model.
"""
import base64
from io import BytesIO
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.api.auth import get_current_user
from app.storage.json_store import store
from app.nlp.resume_parser import (
    clean_text, extract_identity_name, extract_sections, extract_skills,
    extract_soft_skills, validate_resume_identity,
)

router = APIRouter(prefix="/resume", tags=["resume"])
MAX_BYTES = 5 * 1024 * 1024

def _extract_text(filename: str, content: bytes) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if len(content) > MAX_BYTES:
        raise HTTPException(400, "Resume must be under 5 MB.")
    try:
        if suffix == "pdf":
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        if suffix == "docx":
            from docx import Document
            return "\n".join(p.text for p in Document(BytesIO(content)).paragraphs)
    except Exception as exc:
        raise HTTPException(422, "The resume could not be read. Upload another readable PDF or DOCX.") from exc
    raise HTTPException(400, "Use a PDF or DOCX resume.")

def _parse(filename: str, content: bytes, full_name: str) -> dict:
    text = clean_text(_extract_text(filename, content))
    if len(text) < 40:
        raise HTTPException(422, "Little or no readable text was found in this file.")
    identity = extract_identity_name(text)
    valid, reason = validate_resume_identity(full_name, identity)
    if not identity:
        raise HTTPException(422, "Couldn't verify the name on this resume. Please upload a resume that clearly includes your name.")
    if not valid:
        raise HTTPException(409, "Resume doesn't match your profile. Please upload your own resume.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    projects = [line[:160] for line in lines if any(word in line.lower() for word in ("built", "developed", "implemented", "created"))][:5]
    return {"name": filename, "text": text[:30000], "identityName": identity, "skills": extract_skills(text), "soft": extract_soft_skills(text), "sections": extract_sections(text), "projects": projects}

class ExtractRequest(BaseModel):
    name: str
    data: str

@router.post("/extract")
def extract(req: ExtractRequest, user: dict = Depends(get_current_user)):
    try:
        content = base64.b64decode(req.data, validate=True)
    except Exception as exc:
        raise HTTPException(400, "Invalid file data.") from exc
    return {"text": _extract_text(req.name, content)}

@router.post("/upload")
async def upload(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content = await file.read()
    parsed = _parse(file.filename or "resume", content, user["profile"].get("full_name", ""))
    profile = user["profile"]
    previous = profile.get("resume", {})
    parsed["version"] = int(previous.get("parsed", {}).get("version", 0)) + 1
    profile["resume"] = {"name": parsed["name"], "text": parsed["text"], "parsed": parsed}
    store.update_user_profile(user["id"], profile)
    return {"ok": True, "parsed": parsed}
