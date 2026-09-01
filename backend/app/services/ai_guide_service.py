from typing import Any, Dict, List, Optional

from app.ml.intent_classifier import intent_classifier
from app.ml.predictor import placement_10k_predictor
from app.services.recommendation_service import recommendation_service


class AIGuideService:
    COMPANY_HINTS = {"amazon", "google", "microsoft", "tcs", "infosys", "accenture", "wipro", "cognizant", "capgemini"}

    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(profile, dict):
            return {}
        student = profile.get("student") or {}
        normalized = dict(profile)
        if isinstance(student, dict):
            normalized.setdefault("target_role", student.get("target"))
            normalized.setdefault("skills", student.get("skills") or {})
            normalized.setdefault("technical_skills", student.get("skills") or {})
            normalized.setdefault("projects", student.get("projects") or [])
            normalized.setdefault("cgpa", student.get("cgpa"))
            normalized.setdefault("backlogs", student.get("backlogs", 0))
            normalized.setdefault("year", student.get("year"))
            normalized.setdefault("academic_risk", student.get("academicRiskResult"))
        return normalized

    def _intent(self, text: str, history: List[Dict[str, Any]]) -> str:
        q = text.lower().strip()
        last_topic = self._last_topic(history) if history else ""
        if self._looks_like_role_recommendation(q):
            return "role_recommendation"
        if self._looks_like_company(q):
            return "company_eligibility"
        if last_topic == "project" and (
            "it" in q
            or "good for" in q
            or "useful for" in q
            or any(t in q for t in ["react", "node", "python", "tensorflow", "pytorch", "nlp", "model", "api"])
        ):
            return "project_guidance"
        if self._looks_like_readiness(q, last_topic):
            return "placement_readiness"
        if (
            q in {"why", "why?", "how", "how?", "tell me more", "explain in detail", "explain this in detail", "give an example"}
            or q.startswith(("why ", "explain that", "explain this", "tell me more", "give me an example", "what about ", "what should i improve"))
        ):
            return "follow_up"
        if last_topic == "project" and (
            q.startswith(("it ", "this ", "that "))
            or "uses " in q
            or any(t in q for t in ["react", "node", "python", "tensorflow", "pytorch", "nlp", "model", "api"])
        ):
            return "project_guidance"
        if last_topic == "project" and ("ml-focused" in q or "ml focused" in q or "machine-learning" in q or "machine learning" in q):
            return "project_guidance"
        if "academic risk" in q or "risk is" in q or "attendance" in q and "academic" in q:
            return "academic_risk"
        if "company" in q or "companies" in q or "eligible" in q or "eligibility" in q:
            return "company_eligibility"
        if "backend" in q and ("full stack" in q or "fullstack" in q or "frontend" in q or "ml" in q or "machine learning" in q or "better" in q or "suit" in q):
            return "role_comparison"
        if "compare" in q or " vs " in q or " versus " in q or "which one" in q:
            return "role_comparison"
        if "project" in q or "portfolio" in q or "building" in q or "build" in q or "worth" in q or "idea" in q:
            return "project_guidance"
        if "resume" in q or "cv" in q:
            return "resume_question"
        if "learn" in q or "this week" in q or "roadmap" in q:
            return "roadmap_question"
        if "gap" in q or "missing" in q or "lack" in q or "why is" in q:
            return "skill_gap"
        if "ready" in q or "readiness" in q or "placement" in q:
            return "placement_readiness"
        if "what do you know about me" in q or "my profile" in q:
            return "profile_question"
        if history and len(q.split()) <= 4:
            return "follow_up"
        model_intent = intent_classifier.predict(text)
        return model_intent if model_intent != "general_career_question" else "general_career_question"

    def detect_intent(self, text: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        return self._intent(text, (history or [])[-12:])

    def _looks_like_readiness(self, q: str, last_topic: str) -> bool:
        if "target role" in q or "ready" in q or "readiness" in q:
            return True
        if q.startswith(("yes", "yeah", "yep")) and any(x in q for x in ["ml", "machine learning", "backend", "frontend", "full stack", "fullstack"]) and last_topic in {"readiness", "placement_readiness", "role_comparison", "role_recommendation", "previous answer"}:
            return True
        return False

    def _looks_like_role_recommendation(self, q: str) -> bool:
        return any(phrase in q for phrase in ["which role", "suitable", "best fit", "fit for me", "recommend a role", "role should i", "better role"])

    def _looks_like_company(self, q: str) -> bool:
        words = {w.strip(" ?!.:,;") for w in q.split()}
        return bool(words & self.COMPANY_HINTS)

    def _live(self, profile: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        live = context or {}
        target = live.get("targetRole") or {}
        if not target:
            match = recommendation_service.calculate_role_match(profile, profile.get("target_role") or "fullstack")
            target = {
                "id": match.get("role_id"),
                "name": match.get("role_title") or profile.get("target_role") or "your target role",
                "requirements": {},
                "alignment": match.get("match_score", 0),
            }
        return {
            "student": live.get("student") or {},
            "target": target,
            "skills": live.get("skills") or profile.get("skills") or {},
            "gaps": live.get("gaps") or [],
            "readiness": live.get("readiness"),
            "companies": live.get("companies") or [],
            "resume": live.get("resume") or (profile.get("resume") or {}),
            "roadmap": live.get("roadmap") or {},
            "learning": live.get("learning") or {},
            "academicRisk": live.get("academicRisk") or profile.get("academic_risk"),
            "roles": live.get("roles") or [],
        }

    def _role_score(self, role: Dict[str, Any], skills: Dict[str, Any]) -> int:
        req = role.get("requirements") or {}
        if not req:
            return int(role.get("alignment") or 0)
        total = sum(float(v) for v in req.values()) or 1
        earned = sum(min(1, float(skills.get(k, 0) or 0) / float(v or 1)) * float(v) for k, v in req.items())
        return round(earned / total * 100)

    def _last_topic(self, history: List[Dict[str, Any]]) -> str:
        recent = history[-12:]
        user_messages = [m for m in recent if m.get("role") in {"user", "student"}]
        for msg in list(reversed(user_messages)) + list(reversed(recent)):
            text = str(msg.get("content") or msg.get("message") or "").lower()
            if "academic risk" in text or "attendance" in text or "mid-term" in text:
                return "academic_risk"
            if "company" in text or "companies" in text or "eligible" in text or self._looks_like_company(text):
                return "company"
            if "ready" in text or "readiness" in text or "target role" in text:
                return "placement_readiness"
            if self._looks_like_role_recommendation(text):
                return "role_recommendation"
            if "backend" in text and ("ml" in text or "machine learning" in text or "better fit" in text):
                return "role_comparison"
            if "project" in text or "platform" in text or "ml-focused" in text or "react" in text and "node" in text:
                return "project"
            if "biggest gap" in text or "skill gap" in text or "tensorflow" in text or "missing skill" in text:
                return "skill_gap"
        return "previous answer"

    def _gap_names(self, live: Dict[str, Any], limit: int = 3) -> List[str]:
        return [g.get("name") or g.get("skill") for g in live["gaps"][:limit] if g.get("name") or g.get("skill")]

    def _mentioned_skill(self, text: str, live: Dict[str, Any]) -> Optional[str]:
        q = text.lower()
        names = set((live.get("skills") or {}).keys())
        names.update(g.get("name") or g.get("skill") for g in live.get("gaps") or [])
        names.update((live.get("target") or {}).get("requirements") or {})
        for skill in sorted([str(x) for x in names if x], key=len, reverse=True):
            compact = skill.lower().replace(".js", "")
            if skill.lower() in q or compact in q:
                return skill
        return None

    def _role_rankings(self, live: Dict[str, Any], skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        roles = live.get("roles") or []
        ranked = []
        for role in roles:
            req = role.get("requirements") or {}
            missing = [k for k, v in req.items() if float(skills.get(k, 0) or 0) < float(v)]
            ranked.append({
                "name": role.get("name") or role.get("id") or "Role",
                "score": self._role_score(role, skills),
                "missing": missing,
            })
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def _company_answer(self, query_text: str, live: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        q = query_text.lower().strip()
        companies = live.get("companies") or []
        named = next((c for c in companies if (c.get("name") or "").lower() in q or q in (c.get("name") or "").lower()), None)
        common_name = next((name for name in self.COMPANY_HINTS if name in q), None) or self._last_company(history)
        if named:
            return f"{named.get('name')} is in the ASCENTRA opportunity data I can see. Your current status is {named.get('eligibility', 'not assessed')}. {named.get('detail') or 'Check CGPA, backlog, department, and role alignment before treating it as reachable.'}"
        if common_name and any(w in q for w in ["improve", "work on", "prepare", "need"]):
            return f"For {common_name.title()}-style roles, I would improve the strongest transferable evidence first: DSA/problem solving, one role-relevant project, and clear backend or ML depth. In your current ASCENTRA profile, TensorFlow and Statistics are the visible ML gaps, but I cannot call them Amazon eligibility rules because Amazon is not in the stored dataset."
        if common_name:
            return f"I do not currently have {common_name.title()} in the stored ASCENTRA campus-opportunity dataset. I can still help you prepare generally for {common_name.title()}-style software or ML roles, but I cannot claim ASCENTRA-specific eligibility without stored company rules."
        if companies:
            readyish = [c for c in companies if c.get("eligibility") != "NOT ELIGIBLE"] or companies
            c = readyish[0]
            return f"From the mapped opportunities I can see, start with {c.get('name')}. Your status is {c.get('eligibility')}. {c.get('detail') or 'Use this as rule-based eligibility, not a placement prediction.'}"
        return "I do not have mapped company opportunities in the current context. Are you asking about a specific company, or which companies fit your profile?"

    def _history_techs(self, history: List[Dict[str, Any]]) -> List[str]:
        recent = " ".join(str(m.get("content") or m.get("message") or "") for m in history[-8:]).lower()
        found = []
        for name, aliases in {
            "React": ["react"],
            "Node.js": ["node", "node.js"],
            "Python": ["python"],
            "TensorFlow": ["tensorflow"],
            "PyTorch": ["pytorch"],
            "NLP": ["nlp"],
            "SQL": ["sql"],
        }.items():
            if any(alias in recent for alias in aliases):
                found.append(name)
        return found

    def _last_company(self, history: List[Dict[str, Any]]) -> Optional[str]:
        for msg in reversed(history[-12:]):
            text = str(msg.get("content") or msg.get("message") or "").lower()
            found = next((name for name in self.COMPANY_HINTS if name in text), None)
            if found:
                return found
        return None

    def process_query(
        self,
        student_profile: Dict[str, Any],
        query_text: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        student_profile = self._normalize_profile(student_profile)
        history = (history or [])[-12:]
        live = self._live(student_profile, context)
        intent = self._intent(query_text, history)
        target = live["target"]
        role_name = target.get("name") or "your target role"
        skills = live["skills"] if isinstance(live["skills"], dict) else {}
        gaps = self._gap_names(live)
        readiness = live["readiness"]

        if not context:
            role_match = recommendation_service.calculate_role_match(student_profile, student_profile.get("target_role") or "fullstack")
            readiness_result = placement_10k_predictor.predict(student_profile)
            if not gaps:
                gaps = role_match.get("missing_skills", [])
            readiness = readiness if readiness is not None else readiness_result.get("readiness_score")

        q_lower = query_text.lower()

        if intent == "follow_up":
            topic = self._last_topic(history)
            if topic == "skill_gap":
                top = gaps[0] if gaps else "the top missing skill"
                current = skills.get(top, 0) if gaps else 0
                need = (target.get("requirements") or {}).get(top)
                need_text = f" against a target level of {need}%" if need is not None else ""
                answer = f"Because {top} has the largest visible distance between your saved evidence ({current}%){need_text} and what {role_name} expects. It also has high portfolio value: one small feature that proves it can improve both your skill signal and your resume evidence."
            elif topic == "project":
                answer = f"The important test is whether the project proves {role_name} work, not just that it exists. Show the stack, real user flow, saved data, and one deployable feature. If it uses React, Node, or Python, make the backend/API and any data or ML logic visible in the README so it becomes portfolio evidence."
            elif topic == "role_comparison":
                answer = "The better fit is the one where your current evidence is stronger and the remaining gaps are realistic to close soon. I would compare role alignment, your strongest saved skills, and the top two gaps before choosing."
            elif topic == "role_recommendation":
                ranked = self._role_rankings(live, skills)
                if ranked:
                    best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
                    second_text = f" {second['name']} follows at {second['score']}%." if second else ""
                    gap_text = f" The main gaps for {best['name']} are {', '.join(best['missing'][:2])}." if best["missing"] else f" {best['name']} has no major visible gaps in the current role data."
                    answer = f"Because {best['name']} has the strongest compatibility score from your saved skills: {best['score']}%.{second_text}{gap_text}"
                else:
                    answer = "I need the role compatibility list from Role Explorer to explain that properly."
            elif topic == "placement_readiness":
                top = gaps[0] if gaps else None
                gap_text = f" The first gap to work on is {top}." if top else " I would next turn your current skills into stronger project evidence."
                answer = f"For {role_name}, readiness is about whether your saved evidence matches the role requirements. Your current readiness is {readiness if readiness is not None else 'not assessed'}/100.{gap_text}"
            elif topic == "company":
                answer = self._company_answer(query_text, live, history)
            elif topic == "academic_risk":
                risk = live.get("academicRisk") or {}
                result = risk.get("result") if isinstance(risk, dict) and risk.get("result") else risk
                factors = result.get("risk_factors") if isinstance(result, dict) else []
                names = [str(f.get("feature", "")).replace("_", " ") for f in (factors or [])[:2] if isinstance(f, dict)]
                focus = " and ".join(names) if names else "your attention factors"
                answer = f"Because {focus} are the signals currently pushing the academic-risk result upward. Keep the response narrow: stabilize attendance, prepare for the next assessment checkpoint, and use the strongest protective factor as support rather than trying to fix everything at once."
            else:
                answer = f"I can continue from the previous point, but I need one bit of direction: are you asking about readiness, a role, a company, a skill, or your project?"

        elif intent == "project_guidance":
            known = [name for name, value in skills.items() if value and value >= 45]
            techs = [skill for skill in skills if skill.lower() in q_lower]
            if not techs:
                techs = [s for s in ["React", "Node.js", "SQL", "REST APIs", "Python", "Docker", "JavaScript", "TensorFlow", "PyTorch", "NLP"] if s.lower().replace(".js", "").replace("node.js", "node") in q_lower]
            if not techs and self._last_topic(history) == "project":
                techs = self._history_techs(history)
            if "ml-focused" in q_lower or "ml focused" in q_lower or "machine learning" in q_lower:
                answer = "Make it ML-focused by adding one real intelligence loop: collect practice data, recommend questions by weak topic, score answer quality with NLP, or predict readiness from quiz history. Keep the UI, but make Python own the model/data layer and show evaluation metrics so it counts as ML Engineer evidence."
            elif len(q_lower.split()) <= 6 and techs and self._last_topic(history) == "project":
                ml_hint = " For an ML Engineer target, add a data-driven piece: model-based question recommendations, answer-quality scoring, topic clustering, analytics, or an NLP feedback module." if "ml" in role_name.lower() or "machine" in role_name.lower() else ""
                answer = f"That stack can work. React gives the product surface, Node can handle auth/APIs, and Python is the strongest place to add intelligence or analytics.{ml_hint} The next step is to make one technically deep feature visible enough to discuss in interviews."
            elif not any(w in q_lower for w in ["useful", "worth", "good", "suggest", "idea", "build", "building", "project"]):
                answer = f"Tell me what the project does and the main technologies you are using, and I will evaluate it against {role_name}."
            if "suggest" in q_lower and known:
                gap_text = f" and deliberately add {', '.join(gaps[:2])}" if gaps else ""
                answer = f"I would build a focused {role_name} portfolio project around {', '.join(known[:3])}{gap_text}. Keep the scope small: one real user flow, saved data, clear README notes, and a deployed version so it becomes usable resume evidence."
            elif "answer" not in locals():
                direct = "Yes, it can be useful" if any(w in q_lower for w in ["useful", "worth", "good"]) else "That can be a strong project"
                stack = f" The technologies you mentioned ({', '.join(techs)}) fit the direction." if techs else " I do not have the project stack yet, so I would judge it more accurately if you share the technologies."
                gap_text = f" Use it to close {', '.join(gaps[:2])}." if gaps else " Since your visible gaps are limited, make it deeper with testing, deployment, and documentation."
                evidence_text = " Add it to your profile or resume once it is real so ASCENTRA can count it as evidence."
                answer = f"{direct} for {role_name} if it demonstrates real implementation, not just preparation content.{stack}{gap_text}{evidence_text}"

        elif intent == "placement_readiness":
            top = gaps[0] if gaps else None
            strong = [name for name, value in skills.items() if value and value >= 55][:3]
            ready_text = f"Your current readiness is {readiness}/100." if readiness is not None else "I do not have a readiness score in the current context."
            strength_text = f" Your stronger visible evidence is {', '.join(strong)}." if strong else " I do not see strong saved skill evidence yet."
            gap_text = f" The first thing I would improve is {top}, because it is the largest visible gap for {role_name}." if top else " The next move is to add stronger project or resume evidence."
            if q_lower.startswith(("yes", "yeah", "yep")):
                answer = f"For {role_name} specifically, you are partly ready but still building evidence. {ready_text}{strength_text}{gap_text}"
            else:
                answer = f"You are partially ready for {role_name}, but not fully there yet. {ready_text}{strength_text}{gap_text}"

        elif intent in {"skill_gap", "roadmap_question"}:
            mentioned = self._mentioned_skill(query_text, live)
            if mentioned:
                req = (target.get("requirements") or {}).get(mentioned)
                current = skills.get(mentioned, 0)
                target_text = f" The target level I can see is {req}%." if req is not None else ""
                gap_text = f" Your saved evidence is {current}%, so the gap is {max(0, int(req) - int(current))} points." if req is not None else f" Your saved evidence is {current}%."
                answer = f"{mentioned} matters because it supports the practical work expected in {role_name}.{target_text}{gap_text} A good next step is to show it through a concrete project feature, not just list it as a skill."
            elif gaps:
                top = gaps[0]
                current = skills.get(top, 0)
                answer = f"I would focus on {top} next. Your saved evidence is {current}% for that skill, and it is one of the clearest gaps for {role_name}. Work on one small project feature or weekly practice task that proves it."
            else:
                answer = f"You do not have a major visible skill gap for {role_name} right now. The next best move is to turn your strongest skills into project or resume evidence."

        elif intent == "role_recommendation":
            ranked = self._role_rankings(live, skills)
            if ranked:
                best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
                second_text = f" followed by {second['name']} at {second['score']}%" if second else ""
                target_text = f" Your selected target is {role_name}; compare that with the ranking instead of assuming it is the best fit."
                gap_text = f" For {best['name']}, the visible gaps are {', '.join(best['missing'][:2])}." if best["missing"] else f" {best['name']} has no major visible gaps in the current role data."
                answer = f"Based on your current profile, {best['name']} is your strongest match at {best['score']}%{second_text}.{target_text}{gap_text}"
            else:
                answer = f"I need Role Explorer compatibility data to compare roles accurately. From the current target alone, I can discuss {role_name}, but I should not pretend that is a full role recommendation."

        elif intent == "role_comparison":
            roles = live.get("roles") or []
            q = query_text.lower()
            wanted = [r for r in roles if any(token in (r.get("name") or "").lower() for token in ["backend", "full-stack", "full stack", "frontend", "ml", "machine learning"]) and any(word in q for word in (r.get("name") or "").lower().replace("/", " ").split())]
            if not wanted:
                wanted = [r for r in roles if any(token in (r.get("name") or "").lower() for token in ["backend", "ml", "machine learning"]) and ("backend" in q or "ml" in q or "machine learning" in q)]
            if not wanted:
                wanted = [target]
            scored = [(r.get("name"), self._role_score(r, skills), [k for k, v in (r.get("requirements") or {}).items() if float(skills.get(k, 0) or 0) < float(v)]) for r in wanted]
            best = sorted(scored, key=lambda x: x[1], reverse=True)[0]
            details = "; ".join(f"{name}: {score}% alignment, gaps: {', '.join(missing[:2]) or 'none visible'}" for name, score, missing in scored[:3])
            answer = f"{best[0]} looks like the stronger fit from the evidence I have right now. {details}. Use this as a fit signal, not a final career decision."

        elif intent == "company_eligibility":
            answer = self._company_answer(query_text, live, history)

        elif intent == "academic_risk":
            risk = live.get("academicRisk") or {}
            result = risk.get("result") if isinstance(risk, dict) and risk.get("result") else risk
            if not result:
                answer = "I do not have a saved academic-risk result yet. Run the Academic Health check, then I can help interpret the attention areas and action plan."
            else:
                level = result.get("risk_level") or "available"
                factors = result.get("risk_factors") or []
                strengths = result.get("protective_factors") or []
                focus = factors[0].get("feature", "the top attention area").replace("_", " ") if factors else "the clearest attention area"
                support = strengths[0].get("feature", "your stronger indicators").replace("_", " ") if strengths else "your stronger indicators"
                answer = f"Your current academic risk level is {level}. I would focus first on {focus}, while continuing what is helping with {support}. Keep this separate from placement readiness; it is an early-warning planning signal."

        elif intent == "resume_question":
            resume = live["resume"] or {}
            parsed = resume.get("parsed") if isinstance(resume, dict) else {}
            parsed = parsed or {}
            if parsed.get("text") or parsed.get("skills"):
                answer = f"Your resume currently shows {len(parsed.get('skills') or [])} detected technical skills and {len(parsed.get('projects') or [])} project signals. Compare those against {role_name}; the strongest update is to add genuine project evidence for any remaining gaps."
            else:
                answer = "I do not have a validated resume yet. Upload one so I can compare its skills and projects against your target role."

        elif intent == "profile_question":
            student = live["student"] or {}
            known_skills = [k for k, v in skills.items() if v]
            answer = f"I know your target role is {role_name}. I can see {len(known_skills)} saved skill scores, readiness {readiness if readiness is not None else 'not assessed'}, department {student.get('department') or 'not provided'}, and CGPA {student.get('cgpa') if student.get('cgpa') is not None else 'not provided'}."

        else:
            if any(word in q_lower for word in ["prioritize", "priority", "this month", "given everything", "everything you know"]):
                top = gaps[0] if gaps else None
                roadmap = live.get("roadmap") or {}
                road_items = roadmap.get("items") if isinstance(roadmap, dict) else []
                companies = live.get("companies") or []
                risk = live.get("academicRisk") or {}
                result = risk.get("result") if isinstance(risk, dict) and risk.get("result") else risk
                risk_level = result.get("risk_level") if isinstance(result, dict) else None
                sim = live.get("lastSimulation") or {}
                first = f"Prioritize {top} this month." if top else f"Prioritize turning your strongest {role_name} skills into visible project and resume evidence this month."
                readiness_text = f" Your readiness is {readiness}/100" if readiness is not None else ""
                road_text = ""
                if road_items:
                    name = road_items[0].get("name") if isinstance(road_items[0], dict) else str(road_items[0])
                    road_text = f" Start with the roadmap item: {name}."
                company_text = f" This also matters for {len(companies)} relevant stored campus opportunities." if companies else ""
                academic_text = f" Keep a parallel academic focus because your saved academic-risk level is {risk_level}." if risk_level else ""
                sim_text = ""
                if isinstance(sim, dict) and sim.get("delta") is not None:
                    sim_text = f" Your latest What-if simulation showed a {sim.get('delta'):+} readiness change, so use that as a planning signal, not a placement probability."
                answer = f"{first}{readiness_text} for {role_name}, and the clearest leverage is the largest gap plus evidence quality.{road_text}{company_text}{academic_text}{sim_text}"
            elif role_name and role_name != "your target role":
                answer = f"I can answer that against your current {role_name} profile. Tell me whether you want readiness, role fit, company eligibility, skills, or project feedback, and I will use the saved ASCENTRA data."
            else:
                answer = "I need a little more context to answer well. Are you asking about choosing a role, checking readiness, a company, a skill gap, or a project?"

        return {
            "intent": intent,
            "answer": answer,
            "target_role": target.get("id") or student_profile.get("target_role"),
            "match_score": target.get("alignment"),
        }


ai_guide_service = AIGuideService()
