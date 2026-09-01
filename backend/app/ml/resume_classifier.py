import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from app.config import settings

class ResumeClassifier:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "resume_role_classifier.joblib")
        self.vec_path = os.path.join(settings.MODELS_DIR, "resume_tfidf_vectorizer.joblib")
        self.clf = None
        self.vectorizer = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            self.clf = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vec_path)

    def predict_roles(self, resume_text: str, top_n: int = 3) -> List[Dict[str, Any]]:
        if not resume_text or not self.clf or not self.vectorizer:
            return []
            
        X_vec = self.vectorizer.transform([resume_text])
        probs = self.clf.predict_proba(X_vec)[0]
        classes = self.clf.classes_
        
        top_indices = probs.argsort()[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                "role": classes[idx],
                "probability": round(float(probs[idx]), 4)
            })
        return results

resume_classifier = ResumeClassifier()
