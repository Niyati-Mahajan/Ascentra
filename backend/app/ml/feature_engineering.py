import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    # Defensible Engineered Feature 1: Academic Consistency Index
    # Combines SSC, HSC, and CGPA (scaled to 100)
    df_feat['AcademicConsistencyIndex'] = np.round(
        (df_feat['SSC_Marks'] * 0.25 + df_feat['HSC_Marks'] * 0.25 + (df_feat['CGPA'] * 10) * 0.5), 2
    )
    
    # Defensible Engineered Feature 2: Practical Experience Score
    # Combines Internships, Projects, and Certifications
    df_feat['PracticalExperienceScore'] = (
        df_feat['Internships'] * 3.0 + df_feat['Projects'] * 2.0 + df_feat['Certifications'] * 1.5
    )
    
    # Defensible Engineered Feature 3: Aptitude & Soft Skill Balance Score
    df_feat['SkillsBalanceScore'] = np.round(
        (df_feat['AptitudeTestScore'] + df_feat['SoftSkillsScore']) / 2.0, 2
    )
    
    return df_feat
