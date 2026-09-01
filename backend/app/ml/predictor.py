import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from app.config import settings
from app.ml.feature_mapper import map_profile_to_10k_features

class Placement10kPredictor:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, "models", "selected", "selected_placement_model.joblib")
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)

    def predict(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        df_feat, missing_fields = map_profile_to_10k_features(profile)
        
        if missing_fields and "cgpa" in missing_fields:
            return {
                "status": "insufficient_data",
                "message": "Complete more of your profile (such as CGPA) to generate a placement-readiness estimate.",
                "missing_features": missing_fields,
                "readiness_score": None
            }

        if self.pipeline is None:
            return {
                "status": "model_not_trained",
                "message": "Placement ML model is not trained yet.",
                "readiness_score": None
            }

        prob = float(self.pipeline.predict_proba(df_feat)[0, 1])
        readiness_score = int(round(prob * 100))
        
        pos_contributors = []
        neg_contributors = []
        
        cgpa = float(df_feat['CGPA'].iloc[0])
        internships = int(df_feat['Internships'].iloc[0])
        projects = int(df_feat['Projects'].iloc[0])
        aptitude = float(df_feat['AptitudeTestScore'].iloc[0])
        
        if cgpa >= 7.5:
            pos_contributors.append("Your academic CGPA is contributing positively to the model's placement-readiness prediction.")
        elif cgpa < 6.5:
            neg_contributors.append("Your academic CGPA is currently contributing less positively to your prediction.")
            
        if internships >= 1:
            pos_contributors.append("Your internship experience is one of the stronger positive signals in your current profile.")
        else:
            neg_contributors.append("Adding internship experience will boost your placement prediction signal.")
            
        if projects >= 2:
            pos_contributors.append("Your project experience is currently contributing positively to the model's prediction.")
            
        if aptitude < 60:
            neg_contributors.append("Your aptitude test performance is currently contributing less positively to your prediction.")

        return {
            "status": "evaluated",
            "readiness": readiness_score,
            "readiness_score": readiness_score,
            "probability": round(prob, 4),
            "confidence": "high" if prob >= 0.7 else "medium",
            "positive_factors": pos_contributors if pos_contributors else ["Academic profile foundation"],
            "negative_factors": neg_contributors if neg_contributors else ["Maintain current skill momentum"],
            "model_explanation": "Predicted using the official 10K Student Placement Machine Learning Pipeline.",
            "data_provenance": "Evaluated by trained 10K Placement Model."
        }

placement_10k_predictor = Placement10kPredictor()
