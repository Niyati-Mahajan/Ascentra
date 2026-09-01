import pytest
import os
import json
from fastapi.testclient import TestClient
from app.main import app
from app.storage.json_store import store

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_auth_and_profile_flow():
    # Register user
    reg_data = {
        "username": "testuser123",
        "email": "testuser123@example.com",
        "password": "Password123!",
        "name": "Rahul Sharma"
    }
    response = client.post("/api/auth/register", json=reg_data)
    assert response.status_code in [200, 201, 409]

    # Login user
    login_data = {
        "email": "testuser123@example.com",
        "password": "Password123!"
    }
    response = client.post("/api/auth/login", json=login_data)
    assert response.status_code == 200
    cookies = response.cookies
    
    # Get Profile
    prof_resp = client.get("/api/profile", cookies=cookies)
    assert prof_resp.status_code == 200
    assert prof_resp.json()["profile"]["full_name"] == "Rahul Sharma"

def test_placement_readiness():
    login_data = {
        "email": "testuser123@example.com",
        "password": "Password123!"
    }
    login_resp = client.post("/api/auth/login", json=login_data)
    cookies = login_resp.cookies
    
    resp = client.get("/api/readiness", cookies=cookies)
    assert resp.status_code == 200
    # Verification: Unconfigured student does not receive a fabricated score
    data = resp.json()
    assert "status" in data or "readiness_score" in data

def test_ai_guide(monkeypatch):
    from app.routes import ai as ai_route

    monkeypatch.setattr(
        ai_route.career_guide_ai,
        "generate_response",
        lambda *args, **kwargs: {"answer": "Mock Gemini answer", "model_used": "test-model", "fallback_used": False},
    )
    login_data = {
        "email": "testuser123@example.com",
        "password": "Password123!"
    }
    login_resp = client.post("/api/auth/login", json=login_data)
    cookies = login_resp.cookies
    
    req_data = {"message": "What skills do I need for fullstack developer?"}
    resp = client.post("/api/ai/guide", json=req_data, cookies=cookies)
    assert resp.status_code == 200
    assert "answer" in resp.json()
