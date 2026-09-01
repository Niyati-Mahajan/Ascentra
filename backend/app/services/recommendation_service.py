import os
import json
from typing import Dict, Any, List
from app.config import settings

def normalize(s: str) -> str:
    return "".join(c.lower() for c in str(s or "") if c.isalnum() or c in "+#.")

def normalized_text(s: str) -> str:
    return "".join(c.lower() if c.isalnum() or c in "+#." else " " for c in str(s or "").replace(".js", " js"))

def skill_aliases(skill: str) -> List[str]:
    base = " ".join(normalized_text(skill).split())
    aliases = {
        "javascript": ["javascript", "js"],
        "node js": ["node js", "nodejs", "node"],
        "express js": ["express js", "expressjs", "express"],
        "rest apis": ["rest apis", "rest api", "restful api", "restful apis", "api endpoint", "api endpoints", "http api"],
        "dsa": ["dsa", "data structures", "algorithms"],
        "git": ["git", "github"],
    }
    return list(dict.fromkeys([base] + aliases.get(base, [])))

def skill_in_text(skill: str, text: Any) -> bool:
    padded = " " + " ".join(normalized_text(evidence_text(text)).split()) + " "
    return any(f" {alias} " in padded for alias in skill_aliases(skill))

def clamp_score(value: Any) -> int:
    try:
        return max(0, min(100, round(float(value or 0))))
    except Exception:
        return 0

def evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(evidence_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(evidence_text(v) for v in value)
    return str(value or "")

class RecommendationService:
    WEIGHTS = {"profile": .35, "project": .25, "assessment": .25, "resume": .10, "roadmap": .05}
    CAPS = {"resumeMention": 35, "resumeContext": 45, "projectMention": 55, "projectStrong": 75, "projectMax": 85, "roadmap": 35, "sharedAssessment": 55}
    def __init__(self):
        self.roles_file = os.path.abspath(os.path.join(settings.BASE_DIR, "..", "data", "career_core_roles.json"))
        self._load_roles()

    def _load_roles(self):
        if os.path.exists(self.roles_file):
            with open(self.roles_file, 'r', encoding='utf-8') as f:
                canonical = json.load(f)
                self.roles = [
                    {
                        "role_id": role.get("id"),
                        "title": role.get("name"),
                        "category": role.get("cat"),
                        "description": role.get("desc"),
                        "required_skills": [
                            {"skill": skill, "importance": level, "minimum_level": level}
                            for skill, level in (role.get("req") or {}).items()
                        ],
                        "preferred_skills": role.get("pref", []),
                    }
                    for role in canonical
                ]
        else:
            self.roles = []

    def get_all_roles(self) -> List[Dict[str, Any]]:
        return self.roles

    def _skill_evidence(self, skill: str, role: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        student = profile.get("student") if isinstance(profile.get("student"), dict) else profile
        skills = profile.get("technical_skills") or student.get("skills") or profile.get("skills") or {}
        resume = profile.get("resume") or {}
        parsed = resume.get("parsed") or {}
        projects = profile.get("projects") or student.get("projects") or []
        learning = profile.get("learning") or student.get("learning") or profile.get("assessment") or {}
        roadmap = student.get("roadmap") or profile.get("roadmap") or {}

        requirement = next((item.get("minimum_level") or item.get("importance") or 0 for item in role.get("required_skills", []) if item.get("skill") == skill), 0)
        profile_has = skill in skills and clamp_score(skills.get(skill)) > 0
        profile_score = clamp_score(skills.get(skill)) if skill in skills else 0

        resume_skills = parsed.get("skills") or resume.get("skills") or []
        resume_mentioned = any(normalize(s) == normalize(skill) or skill_in_text(skill, s) for s in resume_skills)
        resume_context = any(skill_in_text(skill, p) for p in (parsed.get("projects") or [])) or skill_in_text(skill, parsed.get("sections", {}).get("experience", "")) or ("project" in str(parsed.get("text") or resume.get("text") or "").lower() and skill_in_text(skill, parsed.get("text") or resume.get("text") or ""))
        resume_score = self.CAPS["resumeContext"] if resume_context else self.CAPS["resumeMention"] if resume_mentioned else 0

        matches = []
        for project in projects:
            project_text = evidence_text(project)
            if skill_in_text(skill, project_text):
                strong = skill_in_text(skill, project.get("tech") or project.get("stack") or project.get("technologies") or "") or bool(skill_in_text(skill, project.get("detail") or "") and any(word in str(project.get("detail") or "").lower() for word in ["built", "implemented", "deployed", "api", "database", "frontend", "backend", "component", "server", "model", "query"]))
                matches.append({"project": project, "strong": strong})
        project_score = 0
        if matches:
            project_score = (self.CAPS["projectStrong"] if any(x["strong"] for x in matches) else self.CAPS["projectMention"]) + min(10, (len(matches)-1)*5)
            project_score = min(self.CAPS["projectMax"], project_score)

        entries = list(learning.get("quizHistory") or []) + ([learning.get("lastQuiz")] if learning.get("lastQuiz") else [])
        hit = next((q for q in reversed([x for x in entries if x]) if any(normalize(s) == normalize(skill) or skill_in_text(skill, s) for s in (q.get("skills") or []))), None)
        assessment_score = 0
        if hit:
            per = hit.get("perSkillScores") or hit.get("skillScores") or {}
            key = next((k for k in per if normalize(k) == normalize(skill) or skill_in_text(skill, k)), None)
            assessment_score = clamp_score(per.get(key) if key else hit.get("score"))
            assessment_score = min(assessment_score, self.CAPS["sharedAssessment"])

        role_id = role.get("role_id")
        roadmap_score = self.CAPS["roadmap"] if roadmap.get(f"{role_id}-{skill}") else 0
        sources = {"profile": profile_score if profile_has else 0, "project": project_score, "assessment": assessment_score, "resume": resume_score, "roadmap": roadmap_score}
        available = [(name, score) for name, score in sources.items() if score > 0]
        weight = sum(self.WEIGHTS[name] for name, _ in available)
        weighted = sum(self.WEIGHTS[name] * score for name, score in available) / weight if weight else 0
        without_roadmap = [(name, score) for name, score in available if name != "roadmap"]
        roadmap_free_weight = sum(self.WEIGHTS[name] for name, _ in without_roadmap)
        roadmap_free = sum(self.WEIGHTS[name] * score for name, score in without_roadmap) / roadmap_free_weight if roadmap_free_weight else 0
        evidence_score = max(profile_score if profile_has else 0, round(weighted), round(roadmap_free))
        if len(available) == 1 and available[0][0] == "resume" and requirement:
            evidence_score = min(evidence_score, requirement - 1)
        status = "No evidence yet" if not available else "Ready" if evidence_score >= requirement else "Developing" if evidence_score >= requirement * .6 else "Priority gap"
        return {
            "skill": skill,
            "role_requirement": requirement,
            "profile_score": profile_score,
            "resume_score": resume_score,
            "project_score": project_score,
            "assessment_score": assessment_score,
            "roadmap_score": roadmap_score,
            "evidence_score": evidence_score,
            "gap": max(0, requirement - evidence_score),
            "status": status,
            "meaningful_evidence": [name for name, _ in available],
        }

    def calculate_role_match(self, student_profile: Dict[str, Any], target_role_id: str = None) -> Dict[str, Any]:
        target_id = target_role_id or student_profile.get("target_role")
        target_role = next((r for r in self.roles if r["role_id"] == target_id), None)
        
        if not target_role:
            target_role = self.roles[0] if self.roles else {}

        reqs = target_role.get("required_skills", [])
        if not reqs:
            return {"match_score": 0, "strong_matches": [], "gaps": []}

        weighted_match = 0
        total_weight = 0
        strong_matches = []
        developing_matches = []
        gaps = []

        for req in reqs:
            sk_name = req["skill"]
            importance = req.get("minimum_level") or req["importance"]
            total_weight += importance
            evidence = self._skill_evidence(sk_name, target_role, student_profile)
            matched_ratio = min(1.0, evidence["evidence_score"] / importance) if importance else 0

            if matched_ratio > 0:
                weighted_match += importance * matched_ratio
                if evidence["status"] == "Ready":
                    strong_matches.append(sk_name)
                else:
                    developing_matches.append(sk_name)
            else:
                gaps.append({
                    "skill": sk_name,
                    "importance": importance,
                    "gap_size": importance
                })

        score = int(round((weighted_match / total_weight) * 100)) if total_weight > 0 else 0
        gaps_sorted = sorted(gaps, key=lambda x: x["importance"], reverse=True)

        return {
            "role_id": target_role.get("role_id"),
            "role_title": target_role.get("title"),
            "match_score": score,
            "strong_matches": strong_matches,
            "developing_skills": developing_matches + [g["skill"] for g in gaps_sorted[2:]],
            "missing_skills": [g["skill"] for g in gaps_sorted[:2]],
            "explanation": f"Calculated from ASCENTRA Core Intelligence evidence semantics ({len(strong_matches)} ready skills, {len(developing_matches)} developing signals out of {len(reqs)}). Resume keywords are weak evidence and do not count as strong matches by themselves."
        }

recommendation_service = RecommendationService()
