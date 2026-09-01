import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from app.config import settings

INTENT_DATA = [
    ("what skills do I need for full stack developer?", "skill_requirements"),
    ("what are the requirements for backend developer", "skill_requirements"),
    ("tell me about software engineer role", "role_information"),
    ("what does a data analyst do?", "role_information"),
    ("show my skill gaps", "skill_gap"),
    ("what am I missing for fullstack?", "skill_gap"),
    ("which career should I choose?", "career_direction"),
    ("what role suits me best?", "career_direction"),
    ("how is my resume?", "resume_question"),
    ("did my resume parse correctly?", "resume_question"),
    ("am I ready for placement?", "placement_readiness"),
    ("what is my readiness score?", "placement_readiness"),
    ("am I eligible for Northstar Systems?", "eligibility_question"),
    ("check my company eligibility", "eligibility_question"),
    ("what should I learn next on my roadmap?", "roadmap_question"),
    ("show my learning roadmap", "roadmap_question"),
    ("suggest a project for fullstack", "project_guidance"),
    ("what projects should I build?", "project_guidance"),
    ("how do I get an internship?", "internship_guidance"),
    ("tell me about companies coming to campus", "company_question"),
    ("what package does Quanta Labs offer?", "company_question"),
    ("how should I prepare for interview?", "interview_preparation"),
    ("compare backend and fullstack developer", "role_comparison"),
    ("whats the difference between data analyst and data scientist", "role_comparison"),
    ("what is my profile status?", "profile_question")
]

def train_intent_model():
    texts, labels = zip(*INTENT_DATA)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(texts)
    
    clf = LogisticRegression(C=1.0)
    clf.fit(X_vec, labels)
    
    model_path = os.path.join(settings.MODELS_DIR, "intent_classifier.joblib")
    vec_path = os.path.join(settings.MODELS_DIR, "intent_vectorizer.joblib")
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print("Intent classifier trained and saved successfully!")

class IntentClassifier:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "intent_classifier.joblib")
        self.vec_path = os.path.join(settings.MODELS_DIR, "intent_vectorizer.joblib")
        self.clf = None
        self.vectorizer = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vec_path):
            self.clf = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vec_path)

    def predict(self, text: str) -> str:
        if not text or not self.clf or not self.vectorizer:
            return "general_career_question"
            
        X_vec = self.vectorizer.transform([text])
        return str(self.clf.predict(X_vec)[0])

intent_classifier = IntentClassifier()

if __name__ == '__main__':
    train_intent_model()
