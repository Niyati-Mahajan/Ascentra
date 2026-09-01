import os
import json
import uuid
from typing import Dict, Any, List, Optional
from app.config import settings

class AssessmentService:
    def __init__(self):
        self.storage_file = os.path.join(settings.STORAGE_DIR, "assessments_history.json")
        self._ensure_file()
        
        self.question_bank = [
            {"id": "q1", "text": "Which HTTP method is normally used to create a new resource?", "options": ["GET", "POST", "DELETE", "HEAD"], "correct": 1, "skill": "REST APIs"},
            {"id": "q2", "text": "What does a React component return?", "options": ["A database row", "UI representation", "A server port", "A CSS file"], "correct": 1, "skill": "React"},
            {"id": "q3", "text": "What is the average lookup time complexity for a hash map?", "options": ["O(1)", "O(n)", "O(log n)", "O(n²)"], "correct": 0, "skill": "DSA"},
            {"id": "q4", "text": "Which clause in SQL is used to filter records after aggregation?", "options": ["WHERE", "ORDER BY", "HAVING", "GROUP BY"], "correct": 2, "skill": "SQL"},
            {"id": "q5", "text": "Which Docker command creates and starts a container?", "options": ["docker build", "docker run", "docker pull", "docker exec"], "correct": 1, "skill": "Docker"}
        ]

    def _ensure_file(self):
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _read_data(self) -> Dict[str, Any]:
        with open(self.storage_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_data(self, data: Dict[str, Any]):
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_weekly_test(self, user_id: str) -> List[Dict[str, Any]]:
        db = self._read_data()
        user_history = db.get(user_id, {}).get("attempted_questions", [])
        
        available = [q for q in self.question_bank if q["id"] not in user_history]
        if len(available) < 3:
            available = self.question_bank # Reset cycle if completed
            
        selected = available[:3]
        return [{"id": q["id"], "text": q["text"], "options": q["options"], "skill": q["skill"]} for q in selected]

    def submit_weekly_test(self, user_id: str, answers: Dict[str, int]) -> Dict[str, Any]:
        db = self._read_data()
        if user_id not in db:
            db[user_id] = {"attempted_questions": [], "history": []}

        correct_count = 0
        total = len(answers)
        
        for qid, choice in answers.items():
            db[user_id]["attempted_questions"].append(qid)
            q = next((q for q in self.question_bank if q["id"] == qid), None)
            if q and q["correct"] == int(choice):
                correct_count += 1

        score = int(round((correct_count / max(1, total)) * 100))
        record = {"score": score, "total": total, "date": str(pd.Timestamp.now()) if 'pd' in globals() else "2026-08-25"}
        db[user_id]["history"].append(record)
        self._save_data(db)
        
        return {"score": score, "correct": correct_count, "total": total}

assessment_service = AssessmentService()
