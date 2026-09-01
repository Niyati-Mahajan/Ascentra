import os


def _load_env_file(path: str, *, override: bool = False) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if override or key not in os.environ:
                    os.environ[key] = value.strip().strip('"').strip("'")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Support both project-level .env and backend/.env. The project-level file is
# where the Node proxy is normally configured; backend/.env can still override
# Python-only settings.
_load_env_file(os.path.join(ROOT_DIR, ".env"))
_load_env_file(os.path.join(BASE_DIR, ".env"), override=True)

try:
    from pydantic_settings import BaseSettings
except ImportError:
    class BaseSettings:
        pass

class Settings:
    PROJECT_NAME: str = "Ascentra AI/ML Career Intelligence API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    BASE_DIR: str = BASE_DIR
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    KNOWLEDGE_DATA_DIR: str = os.path.join(DATA_DIR, "knowledge")
    
    MODELS_DIR: str = os.path.join(BASE_DIR, "models", "trained")
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage_data")
    USERS_FILE: str = os.path.join(STORAGE_DIR, "users.json")
    REPORTS_FILE: str = os.path.join(STORAGE_DIR, "training_report.json")
    
    # LLM Settings
    AI_API_KEY: str = os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_API_KEY") or ""
    PRIMARY_LLM_MODEL: str = os.environ.get("PRIMARY_LLM_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    FALLBACK_LLM_MODEL: str = os.environ.get("FALLBACK_LLM_MODEL", "gemini-3.5-flash-lite")
    GEMINI_MODEL: str = PRIMARY_LLM_MODEL
    AI_TIMEOUT_SECONDS: int = int(os.environ.get("AI_TIMEOUT_SECONDS", "15"))
    AI_MAX_CONTEXT_CHARS: int = int(os.environ.get("AI_MAX_CONTEXT_CHARS", "24000"))
    AI_MAX_RESUME_TEXT_CHARS: int = int(os.environ.get("AI_MAX_RESUME_TEXT_CHARS", "4000"))
    AI_CIRCUIT_BREAKER_FAILURES: int = int(os.environ.get("AI_CIRCUIT_BREAKER_FAILURES", "3"))
    AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = int(os.environ.get("AI_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "45"))
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:4173", "http://127.0.0.1:4173", "http://localhost:5173", "http://127.0.0.1:5173"]

settings = Settings()

os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(settings.KNOWLEDGE_DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
