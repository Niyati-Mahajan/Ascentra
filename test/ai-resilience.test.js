const assert=require('assert');
const fs=require('fs');
const path=require('path');

const server=fs.readFileSync(path.join(__dirname,'..','server.js'),'utf8');
const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');
const pyClient=fs.readFileSync(path.join(__dirname,'..','backend','app','ai','career_guide.py'),'utf8');
const pyRoutes=fs.readFileSync(path.join(__dirname,'..','backend','app','routes','ai.py'),'utf8');
const pyConfig=fs.readFileSync(path.join(__dirname,'..','backend','app','config.py'),'utf8');

const route=server.match(/if\(url\.pathname==='\/api\/career-guide'\|\|url\.pathname==='\/api\/ai\/career-guide\/chat'\)\{[\s\S]*?return proxyReq\.end\(\);\n  \}/)?.[0]||'';
assert(route.includes("if(!me)return json(res,401,{error:'Signed out'})"),'Node AI route must require an authenticated session');
assert(route.includes('timeout:aiProxyTimeoutMs()'),'Node AI proxy must have an explicit timeout');
assert(route.includes("AI_BACKEND_TIMEOUT"),'Node AI proxy must return a structured timeout code');
assert(route.includes("AI_BACKEND_UNAVAILABLE"),'Node AI proxy must return a structured backend-unavailable code');
assert(route.includes("AI_BACKEND_INVALID_RESPONSE"),'Node AI proxy must handle invalid backend JSON safely');
assert(!route.includes("response_source:'local_fallback'"),'Node AI proxy must not replace Gemini with deterministic local responses');
assert(server.includes('checkAiBackend()'),'Node startup must check AI backend reachability without blocking app startup');
assert(server.includes("url.pathname==='/api/ai/career-guide/health'&&req.method==='GET'"),'Node must proxy AI health diagnostics');

assert(pyConfig.includes('FALLBACK_LLM_MODEL'),'FastAPI config must define a fallback Gemini model');
assert(pyConfig.includes('AI_TIMEOUT_SECONDS'),'FastAPI config must define Gemini request timeout');
assert(pyConfig.includes('AI_MAX_CONTEXT_CHARS'),'FastAPI config must define context size guard');
assert(pyClient.includes('RETRYABLE_CATEGORIES'),'Gemini client must categorize retryable failures');
assert(pyClient.includes('schedule = [(models[0], 1, 0), (models[0], 2, 0.6)]'),'Gemini client must retry primary briefly');
assert(pyClient.includes('schedule.append((models[1], 1, 1.2))'),'Gemini client must try fallback model after primary retry');
assert(pyClient.includes('urllib.request.urlopen(req, timeout=settings.AI_TIMEOUT_SECONDS)'),'Gemini requests must use explicit timeout');
assert(pyClient.includes('last_ai_success_at'),'Gemini client must record last success metadata');
assert(pyClient.includes('consecutive_failures'),'Gemini client must track consecutive failures');
assert(pyClient.includes('degraded_until'),'Gemini client must implement cooldown/degraded state');
assert(pyClient.includes('INVALID_RESPONSE'),'Gemini client must validate empty or malformed responses');

assert(pyRoutes.includes('"backend": "ok"'),'AI health must report backend status');
assert(pyRoutes.includes('"ai_status"'),'AI health/errors must report AI status');
assert(pyRoutes.includes('/ai/career-guide/probe'),'FastAPI must expose a manual AI probe route');
assert(pyRoutes.includes('detail={"ok": False, "error": error.category'),'FastAPI AI errors must be structured');

assert(app.includes('AI_HISTORY_LIMIT=12'),'frontend must bound transmitted conversation history');
assert(app.includes('AI_CONTEXT_LIMIT=22000'),'frontend must enforce a context size guard');
assert(app.includes('AI_RESUME_TEXT_LIMIT=3000'),'frontend must not blindly send full raw resume text');
assert(app.includes('AbortController'),'frontend AI requests must have a timeout');
assert(app.includes("'X-Request-Id':requestId"),'frontend must send a request id');
assert(app.includes('state.guideHistory.filter(m=>!(m.error&&m.retry)).slice(-AI_HISTORY_LIMIT)'),'frontend must avoid sending retry-error cards and bound history');
assert(app.includes('findLastIndex(m=>m.error&&m.retry&&m.failedPrompt===q)'),'Retry must replace the failed assistant card');
assert(app.includes("guide({retry:true,message:window._ascentraLastPrompt})"),'Retry must resend the failed turn without duplicating the user message');
assert(app.includes('if(window._ascentraAiBusy)return'),'frontend must prevent concurrent duplicate AI requests');

console.log('ai resilience tests passed');
