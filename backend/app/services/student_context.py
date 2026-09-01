from typing import Dict, Any, List, Optional
import os
import json
from app.config import settings
from app.ml.predictor import placement_10k_predictor
from app.services.recommendation_service import recommendation_service

class StudentContextBuilder:
    def __init__(self):
        self._load_roles_and_skills()

    def _load_roles_and_skills(self):
        roles_path = os.path.join(settings.RAW_DATA_DIR, "all_job_post.csv")
        skills_path = os.path.join(settings.RAW_DATA_DIR, "skills_list.csv")
        
        self.known_roles = [
            "Full Stack Developer", "Backend Developer", "Frontend Developer",
            "Software Engineer", "Data Analyst", "Data Scientist",
            "Machine Learning Engineer", "DevOps / Cloud Engineer",
            "Cybersecurity Analyst", "QA / Automation Engineer"
        ]

    def build_context(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds complete, un-fabricated student context for LLM prompt grounding.
        """
        profile_data = user_profile.get("profile") if "profile" in user_profile else user_profile
        student = profile_data.get("student", {}) if isinstance(profile_data, dict) else {}
        resume_data = profile_data.get("resume", {}) if isinstance(profile_data, dict) else {}
        parsed_resume = resume_data.get("parsed", {}) if isinstance(resume_data, dict) else {}

        # 1. Student Identity & Academic Profile
        full_name = student.get("name") or profile_data.get("name") or "Unknown"
        university = student.get("university") or "Not provided"
        degree = student.get("degree") or "Not provided"
        department = student.get("department") or student.get("branch") or "Not provided"
        semester = student.get("semester")
        year = student.get("year") or (int((semester + 1) / 2) if semester else "Not provided")
        cgpa = student.get("cgpa")
        backlogs = student.get("backlogs", 0)

        # 2. Technical & Soft Skills
        raw_skills = student.get("skills", {})
        if isinstance(raw_skills, list):
            student_skills = raw_skills
        elif isinstance(raw_skills, dict):
            student_skills = [k for k, v in raw_skills.items() if v > 0]
        else:
            student_skills = []

        resume_skills = parsed_resume.get("skills", [])
        all_tech_skills = sorted(list(set(student_skills + resume_skills)))

        soft_skills = parsed_resume.get("soft", [])

        # 3. Resume Evidence
        projects = parsed_resume.get("projects", []) or student.get("projects", [])
        internships = parsed_resume.get("sections", {}).get("internships", False)
        has_resume = bool(parsed_resume.get("text") or resume_data.get("text"))

        # 4. Career Goal & Alignment
        target_role_id = student.get("target") or "fullstack"
        role_match = recommendation_service.calculate_role_match(profile_data, target_role_id)
        target_role_name = role_match.get("role_title", target_role_id)

        # 5. Placement ML Model & SHAP
        ml_result = placement_10k_predictor.predict(profile_data)
        readiness_score = ml_result.get("readiness_score")
        ml_prediction = ml_result.get("prediction")
        pos_factors = ml_result.get("positive_factors", [])
        neg_factors = ml_result.get("risk_factors", [])

        context = {
            "student_info": {
                "full_name": full_name,
                "year_of_study": year,
                "degree": degree,
                "department": department,
                "university": university,
            },
            "academic": {
                "cgpa": cgpa if cgpa is not None else "Not provided",
                "backlogs": backlogs,
            },
            "skills": {
                "technical_skills": all_tech_skills if all_tech_skills else "None provided yet",
                "soft_skills": soft_skills if soft_skills else "None provided yet",
            },
            "resume": {
                "uploaded": has_resume,
                "detected_skills": resume_skills if resume_skills else "No resume skills detected",
                "projects": projects if projects else "No projects recorded",
                "internship_experience": "Yes" if internships else "No",
            },
            "career_preference": {
                "target_role": target_role_name,
                "role_match_score": f"{role_match.get('match_score', 0)}%",
                "covered_skills": role_match.get("strong_matches", []),
                "missing_skills_for_target": role_match.get("missing_skills", []),
            },
            "placement_readiness": {
                "source": "ASCENTRA readiness model",
                "placement_readiness_score": f"{readiness_score}/100" if readiness_score is not None else "Not enough profile data",
                "readiness_band": ml_prediction if ml_prediction else "Unknown",
                "strengths": pos_factors,
                "improvement_areas": neg_factors,
            }
        }
        return context

    def build_ascentra_context(
        self,
        user_profile: Dict[str, Any],
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        live_context: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds compact, relevant ASCENTRA context for Gemini.

        Intent/context routing selects data only. It must not select or generate
        canned answers; Gemini remains the normal natural-language response layer.
        """
        base_profile = user_profile.get("profile") if "profile" in user_profile else user_profile
        base_profile = base_profile if isinstance(base_profile, dict) else {}
        student = base_profile.get("student") or {}
        resume = base_profile.get("resume") or {}
        live = live_context or {}
        text = (message or "").lower()
        intent = intent or "general_career_question"

        normalized_profile = dict(base_profile)
        normalized_profile.setdefault("target_role", student.get("target"))
        normalized_profile.setdefault("skills", student.get("skills") or {})
        normalized_profile.setdefault("technical_skills", student.get("skills") or {})
        normalized_profile.setdefault("resume", resume)
        normalized_profile.setdefault("cgpa", student.get("cgpa"))
        normalized_profile.setdefault("backlogs", student.get("backlogs", 0))

        compact: Dict[str, Any] = {
            "current_question": message,
            "detected_intent": intent,
            "selected_context_sections": [],
            "profile": {
                "name": student.get("name") or base_profile.get("name"),
                "degree": student.get("degree"),
                "department": student.get("department") or student.get("branch"),
                "semester": student.get("semester"),
                "cgpa": student.get("cgpa"),
                "backlogs": student.get("backlogs", 0),
            },
        }

        def add(section: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (dict, list)) and not value:
                return
            compact[section] = value
            compact["selected_context_sections"].append(section)

        target = live.get("targetRole") or {}
        if not target:
            target_id = student.get("target") or normalized_profile.get("target_role")
            role_match = recommendation_service.calculate_role_match(normalized_profile, target_id or "fullstack")
            target = {
                "id": role_match.get("role_id") or target_id,
                "name": role_match.get("role_title") or target_id,
                "alignment": role_match.get("match_score"),
                "covered_skills": role_match.get("strong_matches", []),
                "missing_skills": role_match.get("missing_skills", []),
            }
        add("target_role", target)

        always_profile_dependent = {
            "role_recommendation",
            "role_comparison",
            "placement_readiness",
            "skill_gap",
            "roadmap_question",
            "project_guidance",
            "resume_question",
            "company_eligibility",
            "academic_risk",
            "profile_question",
            "follow_up",
        }

        needs_roles = intent in {"role_recommendation", "role_comparison", "placement_readiness", "follow_up"} or any(
            phrase in text for phrase in ["which role", "suitable", "compare", "backend", "frontend", "full stack", "machine learning", "ml engineer"]
        )
        needs_skills = intent in always_profile_dependent or any(word in text for word in ["skill", "gap", "improve", "learn", "ready", "project", "resume"])
        needs_resume = intent in {"resume_question", "project_guidance", "placement_readiness", "follow_up"} or any(word in text for word in ["resume", "cv", "project", "evidence", "portfolio"])
        needs_companies = intent == "company_eligibility" or any(word in text for word in ["company", "companies", "eligible", "eligibility", "amazon", "google", "microsoft", "tcs", "infosys", "accenture", "wipro"])
        needs_roadmap = intent in {"roadmap_question", "skill_gap", "placement_readiness", "follow_up"} or any(word in text for word in ["roadmap", "next", "first", "priority", "work on", "improve"])
        needs_academic = intent == "academic_risk" or any(word in text for word in ["academic", "risk", "attendance", "tgpa", "mid-term", "mid term"])
        needs_weekly = intent in {"placement_readiness", "skill_gap", "roadmap_question", "follow_up"} or "weekly" in text
        needs_simulation = "what-if" in text or "what if" in text or "simulate" in text or "simulation" in text
        broad_summary = any(phrase in text for phrase in ["everything ascentra knows", "everything you know", "based on everything", "overall", "complete system"])
        if broad_summary:
            needs_roles = needs_skills = needs_resume = needs_companies = needs_roadmap = needs_academic = needs_weekly = True

        if needs_roles:
            roles = live.get("roles") or []
            if not roles:
                roles = [
                    {
                        "id": role.get("role_id"),
                        "name": role.get("title"),
                        "description": role.get("description"),
                        "match": recommendation_service.calculate_role_match(normalized_profile, role.get("role_id")),
                    }
                    for role in recommendation_service.get_all_roles()
                ]
            add("role_compatibility", roles[:10])

        if needs_skills:
            add("skill_profile", {
                "student_reported_scores": live.get("skills") or student.get("skills") or {},
                "target_gaps": live.get("gaps") or target.get("missing_skills") or [],
            })

        if needs_resume:
            parsed = (live.get("resume") or resume or {}).get("parsed") if isinstance(live.get("resume") or resume, dict) else {}
            parsed = parsed or {}
            add("resume_evidence", {
                "uploaded": bool(parsed.get("text") or resume.get("text")),
                "detected_skills": parsed.get("skills") or resume.get("skills") or [],
                "projects": (parsed.get("projects") or student.get("projects") or [])[:6],
                "sections": parsed.get("sections") or {},
                "tools_or_links": parsed.get("links") or [],
            })

        if needs_companies:
            add("company_opportunities", live.get("companies") or [])

        if needs_roadmap:
            add("roadmap", live.get("roadmap") or {"completed": student.get("roadmap") or {}, "active_actions": []})

        if needs_weekly:
            learning = live.get("learning") or base_profile.get("learning") or {}
            add("weekly_validation", {
                "latest": learning.get("lastQuiz"),
                "history_count": len(learning.get("quizHistory") or []),
                "methodology": "supporting weekly assessment evidence; not certified proficiency",
            })

        if needs_academic:
            result = live.get("academicRisk") or student.get("academicRiskResult")
            input_data = student.get("academicRiskInput")
            add("academic_health", {
                "input": input_data,
                "result": self._student_facing_academic_result(result),
            })

        if needs_simulation:
            add("what_if", live.get("lastSimulation"))

        if intent == "general_career_question" and not any(k in compact for k in ["skill_profile", "role_compatibility", "resume_evidence", "company_opportunities", "academic_health"]):
            compact["selected_context_sections"].append("general_conversation")

        return compact

    def _student_facing_academic_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        raw = result.get("result") if isinstance(result.get("result"), dict) else result

        def clean_factor(item: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "signal": str(item.get("feature") or "").replace("_", " "),
                "value": item.get("value"),
                "direction": item.get("direction"),
            }

        return {
            "risk_probability": raw.get("risk_probability"),
            "risk_level": raw.get("risk_level"),
            "decision": raw.get("decision"),
            "attention_factors": [clean_factor(x) for x in (raw.get("risk_factors") or [])[:4] if isinstance(x, dict)],
            "protective_factors": [clean_factor(x) for x in (raw.get("protective_factors") or [])[:4] if isinstance(x, dict)],
            "recommended_actions": [
                {
                    "area": item.get("risk_factor"),
                    "recommendation": item.get("recommendation"),
                }
                for item in (raw.get("interventions") or [])[:3]
                if isinstance(item, dict)
            ],
        }

student_context_builder = StudentContextBuilder()
