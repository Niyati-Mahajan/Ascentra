from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import auth, profile, resume, intelligence, ai_guide, assessments
from app.routes import ai as ai_routes

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# Startup API Key check
@app.on_event("startup")
def startup_event():
    from app.ai.career_guide import career_guide_ai
    ai = career_guide_ai.startup_summary()
    print(f"=== ASCENTRA BACKEND STARTUP ===")
    print(f"AI configured: {ai['configured']}")
    print(f"Primary model: {ai['primary_model']}")
    print(f"Fallback model: {ai['fallback_model']}")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ai_routes.router, prefix=settings.API_V1_STR)
app.include_router(ai_routes.router, prefix="/api")
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix="/api")
app.include_router(profile.router, prefix=settings.API_V1_STR)
app.include_router(profile.router, prefix="/api")
app.include_router(resume.router, prefix=settings.API_V1_STR)
app.include_router(resume.router, prefix="/api")
app.include_router(intelligence.router, prefix=settings.API_V1_STR)
app.include_router(intelligence.router, prefix="/api")
app.include_router(ai_guide.router, prefix=settings.API_V1_STR)
app.include_router(ai_guide.router, prefix="/api")
app.include_router(assessments.router, prefix=settings.API_V1_STR)
app.include_router(assessments.router, prefix="/api")

@app.get("/")
def root():
    return {
        "service": "Ascentra Python Backend",
        "status": "online"
    }

@app.get("/api/health")
def health_check():
    from app.ml.predictor import placement_10k_predictor
    from app.ai.career_guide import career_guide_ai
    
    ml_loaded = placement_10k_predictor.pipeline is not None
    ai = career_guide_ai.health()
    
    return {
        "status": "healthy",
        "backend": "ok",
        "service": "ascentra-backend",
        "ml_model_loaded": ml_loaded,
        "gemini_configured": ai["ai_configured"],
        **ai,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
