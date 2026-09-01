import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def map_profile_to_10k_features(profile: Dict[str, Any]) -> Tuple[pd.DataFrame, list]:
    missing_fields = []
    
    student = profile.get("student") if isinstance(profile.get("student"), dict) else profile
    
    cgpa = profile.get("cgpa") or student.get("cgpa")
    if cgpa is None or float(cgpa) <= 0:
        missing_fields.append("cgpa")
        
    skills = profile.get("technical_skills") or student.get("skills") or {}
    projects = profile.get("projects") or student.get("projects") or []
    internships = profile.get("internships") or student.get("internships") or 0
    certifications = profile.get("certifications") or student.get("certifications") or []
    
    cgpa_val = float(cgpa) if cgpa is not None else 7.0
    internships_val = int(internships)
    projects_val = len(projects) if isinstance(projects, list) else int(projects)
    certifications_val = len(certifications) if isinstance(certifications, list) else int(certifications)
    
    avg_skill = sum(skills.values()) / len(skills) if skills else 70.0
    aptitude_val = float(skills.get("Aptitude", skills.get("DSA", avg_skill)))
    soft_val = float(skills.get("Communication", 72.0))
    
    extracurricular_val = 1 if profile.get("extracurricular") or student.get("extracurricular") else 0
    training_val = 1 if profile.get("placement_training") or student.get("placement_training") else 1
    
    ssc_val = float(profile.get("ssc_marks") or student.get("ssc_marks") or 78.0)
    hsc_val = float(profile.get("hsc_marks") or student.get("hsc_marks") or 75.0)
    stream_val = str(profile.get("branch") or student.get("department") or "Computer Science")
    
    feat_dict = {
        'CGPA': cgpa_val,
        'Internships': internships_val,
        'Projects': projects_val,
        'Certifications': certifications_val,
        'AptitudeTestScore': aptitude_val,
        'SoftSkillsScore': soft_val,
        'ExtracurricularActivities': extracurricular_val,
        'PlacementTraining': training_val,
        'SSC_Marks': ssc_val,
        'HSC_Marks': hsc_val,
        'Stream': stream_val,
        'AcademicConsistencyIndex': np.round((ssc_val * 0.25 + hsc_val * 0.25 + (cgpa_val * 10) * 0.5), 2),
        'PracticalExperienceScore': (internships_val * 3.0 + projects_val * 2.0 + certifications_val * 1.5),
        'SkillsBalanceScore': np.round((aptitude_val + soft_val) / 2.0, 2)
    }
    
    df_feat = pd.DataFrame([feat_dict])
    return df_feat, missing_fields
