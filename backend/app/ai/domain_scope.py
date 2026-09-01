import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


IN_SCOPE = "in_scope"
BORDERLINE = "borderline"
OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class ScopeDecision:
    scope: str
    reason: str
    instruction: Optional[str] = None


CAREER_TERMS = {
    "academic",
    "academic risk",
    "ascentra",
    "backend placement",
    "career",
    "cgpa",
    "company",
    "companies",
    "eligibility",
    "eligible",
    "evidence",
    "full-stack",
    "full stack",
    "gap",
    "interview",
    "interviews",
    "job",
    "placement",
    "placements",
    "portfolio",
    "prepared",
    "prepare",
    "readiness",
    "resume",
    "roadmap",
    "role",
    "roles",
    "skill",
    "skills",
    "target role",
    "what-if",
    "what if",
}

TECH_TERMS = {
    "api",
    "apis",
    "arrays",
    "backend",
    "binary tree",
    "dbms",
    "dsa",
    "frontend",
    "full stack",
    "git",
    "javascript",
    "linked list",
    "linked lists",
    "node",
    "normalization",
    "react",
    "rest",
    "sql",
    "state",
}

OUT_OF_SCOPE_TERMS = {
    "chemistry assignment",
    "cricket match",
    "essay",
    "joke",
    "poem",
    "quantum mechanics",
    "trivia",
    "vacation",
}

GENERAL_TASK_PATTERNS = [
    r"\bbuild (a|an|the|complete)\b",
    r"\bdo my\b",
    r"\bfinish my\b",
    r"\bgive me .* code\b",
    r"\bhelp me solve\b",
    r"\bimplement\b",
    r"\bleetcode\b",
    r"\bsolve\b",
    r"\bwrite .* assignment\b",
    r"\bwrite .* code\b",
    r"\bwrite my\b",
]

CONCEPT_STARTS = (
    "explain ",
    "how does ",
    "how do ",
    "teach me ",
    "what is ",
    "what are ",
)


def classify_domain_scope(
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    student_context: Optional[Dict[str, Any]] = None,
) -> ScopeDecision:
    text = _normalize(user_message)
    history_text = _normalize(" ".join(_message_text(m) for m in (history or [])[-8:]))

    if _is_diagnostic(student_context):
        return ScopeDecision(IN_SCOPE, "diagnostic")
    if not text:
        return ScopeDecision(OUT_OF_SCOPE, "empty")
    if text in {"reply only with ok", "reply with ok"}:
        return ScopeDecision(IN_SCOPE, "diagnostic_probe")
    if _has_career_context(text):
        return ScopeDecision(IN_SCOPE, "career_context")
    if _has_recent_career_context(history_text) and _is_follow_up(text):
        return ScopeDecision(IN_SCOPE, "career_follow_up")

    if _is_general_task_completion(text):
        return ScopeDecision(OUT_OF_SCOPE, "general_task_completion")
    if any(term in text for term in OUT_OF_SCOPE_TERMS):
        return ScopeDecision(OUT_OF_SCOPE, "general_topic")
    if _is_direct_coding_solution_request(text):
        return ScopeDecision(OUT_OF_SCOPE, "coding_solution")

    if _is_technical_concept_question(text):
        return ScopeDecision(
            BORDERLINE,
            "technical_concept",
            "The user is asking about a technical concept without explicit placement context. Answer briefly, explain why it matters for the target role or interviews, suggest how deeply to know it, and connect it to readiness, roadmap, or skill evidence. Do not provide a long tutorial or complete solution.",
        )

    if _is_greeting_or_short_followup(text):
        return ScopeDecision(IN_SCOPE, "conversation_management")
    return ScopeDecision(OUT_OF_SCOPE, "outside_ascentra_scope")


def build_scope_redirect(user_message: str, student_context: Optional[Dict[str, Any]] = None) -> str:
    text = _normalize(user_message)
    dsa_gap = _has_dsa_gap(student_context or {})
    gap_sentence = " DSA is visible in your ASCENTRA gaps, so linked-list practice may be useful." if dsa_gap else ""

    if "linked list" in text or "linked-list" in text:
        return (
            "Linked lists are relevant to DSA interview preparation, so I can help you plan how to practice them."
            f"{gap_sentence} For placement prep, focus on traversal, reversal, cycle detection, pointer manipulation, and time complexity. "
            "I can also help you decide whether DSA should be a current priority for your target role."
        )
    if "two sum" in text or "leetcode" in text or "solve" in text:
        return (
            "I'm focused on career and placement guidance inside ASCENTRA. I can help you choose which DSA patterns to practice, "
            "how to structure interview preparation, or whether problem-solving is a priority gap for your target role."
        )
    return (
        "I'm focused on career and placement guidance inside ASCENTRA. I can help with readiness, role fit, skill gaps, "
        "resume or project evidence, company preparation, interview strategy, academic health, roadmap planning, or ASCENTRA features."
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _message_text(message: Dict[str, Any]) -> str:
    return str(message.get("content") or message.get("message") or "")


def _is_diagnostic(student_context: Optional[Dict[str, Any]]) -> bool:
    return bool(isinstance(student_context, dict) and student_context.get("diagnostic"))


def _has_career_context(text: str) -> bool:
    if "git" in text and "developing" in text:
        return True
    if any(term in text for term in CAREER_TERMS):
        return True
    return bool(re.search(r"\b(dsa|sql|react|backend|frontend|api|apis)\b.*\b(interview|placement|placements|role|roadmap|resume|project)\b", text))


def _has_recent_career_context(history_text: str) -> bool:
    return any(term in history_text for term in CAREER_TERMS)


def _is_follow_up(text: str) -> bool:
    if len(text.split()) <= 5:
        return True
    return text.startswith(("how about ", "what about ", "and ", "also ", "tell me more", "explain that"))


def _is_general_task_completion(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in GENERAL_TASK_PATTERNS)


def _is_direct_coding_solution_request(text: str) -> bool:
    direct_coding = any(term in text for term in ["reverse a linked list", "reverse linked list", "two sum"])
    asks_for_solution = any(phrase in text for phrase in ["can you help me", "code", "give me", "solve", "write", "implementation"])
    return direct_coding and asks_for_solution


def _is_technical_concept_question(text: str) -> bool:
    if not any(term in text for term in TECH_TERMS):
        return False
    return text.startswith(CONCEPT_STARTS) or "teach me" in text or text in TECH_TERMS


def _is_greeting_or_short_followup(text: str) -> bool:
    return text in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no"} or len(text.split()) <= 2


def _has_dsa_gap(context: Dict[str, Any]) -> bool:
    gaps = []
    skill_profile = context.get("skill_profile") if isinstance(context, dict) else {}
    target_role = context.get("target_role") if isinstance(context, dict) else {}
    if isinstance(skill_profile, dict):
        gaps.extend(skill_profile.get("target_gaps") or [])
    if isinstance(target_role, dict):
        gaps.extend(target_role.get("missing_skills") or [])
    for gap in gaps:
        name = gap.get("name") or gap.get("skill") if isinstance(gap, dict) else gap
        if name and "dsa" in str(name).lower():
            return True
    return False
