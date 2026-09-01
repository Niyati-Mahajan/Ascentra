# Ascentra

Self-contained placement guidance and career-roadmap application. Open `index.html` in a browser, or run a static server from this directory:

```powershell
node server.js
```

Module A academic-risk prediction stays in its own FastAPI service. Start it separately, then point the Node server at it:

```powershell
cd Module-A
python -m uvicorn api:app --host 127.0.0.1 --port 8001

cd ..
$env:MODULE_A_API_URL="http://127.0.0.1:8001"
node server.js
```

The frontend calls the Node endpoint `POST /api/academic-risk`; Node validates the academic fields and forwards them to Module A. The existing placement/career intelligence functionality remains separate.

The current implementation deliberately uses `localStorage` as its repository so the full interaction flow works without credentials or an external service. `Store` at the top of `app.js` is the single persistence boundary; replace it with authenticated API calls when integrating the supplied backend/data services.

Key calculation services live alongside it: normalized resume skill extraction, transparent role/readiness scoring, rule-based company eligibility, campus alignment, gap prioritisation, dynamic roadmap generation, what-if calculations, and the Gemini-powered ASCENTRA AI chat layer.

ASCENTRA AI uses Gemini only as the conversational reasoning layer. ASCENTRA profile data, role matching, readiness calculations, company rules, resume evidence, roadmap state, weekly checks, and Module A academic-risk results remain the source of truth.

Create `.env` or `backend/.env` from `.env.example` and provide:

```powershell
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
FALLBACK_LLM_MODEL=gemini-3.5-flash-lite
```

Run the full local app:

```powershell
# Terminal 1: Gemini-backed ASCENTRA AI
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: web app / Node proxy
cd ..
node server.js
```

Optional, only if using Academic Risk:

```powershell
# Terminal 3
cd Module-A
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

AI diagnostics:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/ai/career-guide/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

The health endpoints report configuration and runtime status without exposing secrets. The manual probe endpoint is `POST /api/ai/career-guide/probe`; use it sparingly because it makes a real Gemini call.
