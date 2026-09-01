import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_before = df.isnull().sum().to_dict()
    dup_count = int(df.duplicated().sum())
    
    # 1. Remove duplicate rows
    df_clean = df.drop_duplicates().copy()
    rows_after_dup = len(df_clean)
    
    # 2. Impute numerical features with median
    num_cols = ['CGPA', 'AptitudeTestScore', 'SoftSkillsScore', 'SSC_Marks', 'HSC_Marks']
    for col in num_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # 3. Impute categorical features with mode
    cat_cols = ['Stream', 'Gender']
    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    # 4. Clean invalid ranges if present
    if 'CGPA' in df_clean.columns:
        df_clean['CGPA'] = df_clean['CGPA'].clip(0.0, 10.0)
    if 'AptitudeTestScore' in df_clean.columns:
        df_clean['AptitudeTestScore'] = df_clean['AptitudeTestScore'].clip(0.0, 100.0)
        
    report = {
        "missing_before": missing_before,
        "missing_after": df_clean.isnull().sum().to_dict(),
        "duplicates_removed": dup_count,
        "initial_rows": len(df),
        "final_rows": len(df_clean)
    }
    return df_clean, report
