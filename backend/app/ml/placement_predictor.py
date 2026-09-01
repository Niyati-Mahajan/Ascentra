import os
import joblib
import json
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, Optional
from app.config import settings

class PlacementPredictor:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "placement_xgboost.joblib")
        self.lr_model_path = os.path.join(settings.MODELS_DIR, "placement_logistic_regression.joblib")
        self.features_path = os.path.join(settings.MODELS_DIR, "placement_features.joblib")
        
        self.pipeline = None
        self.feature_cols = []
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        elif os.path.exists(self.lr_model_path):
            self.pipeline = joblib.load(self.lr_model_path)
            
        if os.path.exists(self.features_path):
            self.feature_cols = joblib.load(self.features_path)

    def predict(self, student_profile: Dict[str, Any]) -> Dict[str, Any]:
        # Validate minimum inputs
        cgpa = student_profile.get("cgpa")
        branch = student_profile.get("branch") or student_profile.get("department") or "CSE"
        
        if cgpa is None or float(cgpa) <= 0:
            return {
                "status": "Not enough information",
                "readiness_score": None,
                "confidence": None,
                "missing_fields": ["cgpa"],
                "message": "Please enter your CGPA in your profile to generate placement readiness."
            }

        # Map student profile to feature vector
        skills = student_profile.get("technical_skills") or student_profile.get("skills") or {}
        projects = student_profile.get("projects") or []
        
        avg_tech = sum(skills.values()) / len(skills) if skills else 40.0

        sample = {
            'cgpa': float(cgpa),
            'branch': str(branch).upper(),
            'college_tier': 2,
            'internships_count': int(student_profile.get("internships", 0)),
            'projects_count': len(projects),
            'certifications_count': len(student_profile.get("certifications", [])),
            'coding_skill_score': float(skills.get("DSA", skills.get("Python", avg_tech))),
            'aptitude_score': 70.0,
            'communication_skill_score': float(skills.get("Communication", 70.0)),
            'logical_reasoning_score': 70.0,
            'hackathons_participated': 0,
            'github_repos': len(projects) * 2,
            'linkedin_connections': 150,
            'mock_interview_score': 68.0,
            'attendance_percentage': 85.0,
            'backlogs': int(student_profile.get("backlogs", 0)),
            'extracurricular_score': 60.0,
            'leadership_score': 55.0,
            'volunteer_experience': 0,
            'study_hours_per_day': 4.0
        }

        df_input = pd.DataFrame([sample])
        
        if self.pipeline is None:
            return {
                "status": "Model not trained",
                "readiness_score": 75,
                "message": "Baseline calculated; ML binary not found."
            }

        prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        readiness_score = int(round(prob * 100))
        prediction_label = "High Placement Likelihood" if prob >= 0.6 else "Needs Preparedness Building"

        # Calculate feature contributions for explainability (SHAP heuristic / linear weights)
        positive_factors = []
        negative_factors = []

        if sample['cgpa'] >= 7.5:
            positive_factors.append("Strong CGPA")
        elif sample['cgpa'] < 6.5:
            negative_factors.append("Low CGPA threshold")

        if sample['projects_count'] >= 2:
            positive_factors.append("Hands-on project evidence")
        else:
            negative_factors.append("Limited project evidence")

        if sample['internships_count'] >= 1:
            positive_factors.append("Internship experience")

        if sample['backlogs'] > 0:
            negative_factors.append(f"Active backlogs ({sample['backlogs']})")

        if sample['coding_skill_score'] >= 70:
            positive_factors.append("Strong technical skill score")

        return {
            "status": "Evaluated",
            "prediction": prediction_label,
            "probability": round(prob, 4),
            "readiness_score": readiness_score,
            "confidence": "Medium-High",
            "positive_factors": positive_factors if positive_factors else ["Good academic foundation"],
            "negative_factors": negative_factors if negative_factors else ["Maintain current momentum"],
            "data_provenance": "Prototype model trained on educational dataset; replace with university data."
        }

placement_predictor = PlacementPredictor()
