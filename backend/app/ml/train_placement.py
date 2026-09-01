import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from app.config import settings

def train_placement_models():
    raw_path = os.path.join(settings.RAW_DATA_DIR, "student_placement_prediction_dataset_2026.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Placement dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path)

    # DATA LEAKAGE / DEMOGRAPHIC EXCLUSIONS:
    # Excluded 'student_id': Arbitrary identifier.
    # Excluded 'salary_package_lpa': Target leakage (occurs after placement).
    # Excluded 'gender': Responsible ML / fairness (demographic characteristic).
    # Excluded 'sleep_hours': Weak causal link to academic/placement performance.
    
    feature_cols = [
        'cgpa', 'branch', 'college_tier', 'internships_count', 'projects_count',
        'certifications_count', 'coding_skill_score', 'aptitude_score',
        'communication_skill_score', 'logical_reasoning_score',
        'hackathons_participated', 'github_repos', 'linkedin_connections',
        'mock_interview_score', 'attendance_percentage', 'backlogs',
        'extracurricular_score', 'leadership_score', 'volunteer_experience',
        'study_hours_per_day'
    ]
    
    target_col = 'placement_status'
    
    # Binary encode target
    y = (df[target_col] == 'Placed').astype(int)
    X = df[feature_cols]

    numeric_features = [
        'cgpa', 'college_tier', 'internships_count', 'projects_count',
        'certifications_count', 'coding_skill_score', 'aptitude_score',
        'communication_skill_score', 'logical_reasoning_score',
        'hackathons_participated', 'github_repos', 'linkedin_connections',
        'mock_interview_score', 'attendance_percentage', 'backlogs',
        'extracurricular_score', 'leadership_score', 'volunteer_experience',
        'study_hours_per_day'
    ]
    categorical_features = ['branch']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. LOGISTIC REGRESSION BASELINE
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average='binary')
    lr_auc = roc_auc_score(y_test, lr_probs)

    # 2. XGBOOST MODEL
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'))
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_p, xgb_r, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_preds, average='binary')
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print("=== PLACEMENT MODEL EVALUATION ===")
    print(f"Logistic Regression -> Acc: {lr_acc:.4f}, Prec: {lr_p:.4f}, Rec: {lr_r:.4f}, F1: {lr_f1:.4f}, ROC-AUC: {lr_auc:.4f}")
    print(f"XGBoost Classifier  -> Acc: {xgb_acc:.4f}, Prec: {xgb_p:.4f}, Rec: {xgb_r:.4f}, F1: {xgb_f1:.4f}, ROC-AUC: {xgb_auc:.4f}")

    # Save models
    joblib.dump(lr_pipeline, os.path.join(settings.MODELS_DIR, "placement_logistic_regression.joblib"))
    joblib.dump(xgb_pipeline, os.path.join(settings.MODELS_DIR, "placement_xgboost.joblib"))
    
    # Save preprocessor and feature metadata for SHAP
    joblib.dump(feature_cols, os.path.join(settings.MODELS_DIR, "placement_features.joblib"))

    report = {
        "dataset": "student_placement_prediction_dataset_2026.csv (Prototype Training Data)",
        "sample_count": len(df),
        "features_used": feature_cols,
        "features_excluded": ["student_id", "salary_package_lpa", "gender", "sleep_hours"],
        "exclusion_reasons": {
            "student_id": "Arbitrary non-predictive identifier",
            "salary_package_lpa": "Target leakage (downstream of placement)",
            "gender": "Responsible ML / fairness principle (protected demographic)",
            "sleep_hours": "Uncorrelated lifestyle metric"
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": {
            "logistic_regression": {
                "accuracy": round(lr_acc, 4),
                "precision": round(lr_p, 4),
                "recall": round(lr_r, 4),
                "f1": round(lr_f1, 4),
                "roc_auc": round(lr_auc, 4)
            },
            "xgboost": {
                "accuracy": round(xgb_acc, 4),
                "precision": round(xgb_p, 4),
                "recall": round(xgb_r, 4),
                "f1": round(xgb_f1, 4),
                "roc_auc": round(xgb_auc, 4)
            }
        },
        "selected_model": "xgboost" if xgb_auc >= lr_auc else "logistic_regression"
    }

    with open(settings.REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == '__main__':
    train_placement_models()
