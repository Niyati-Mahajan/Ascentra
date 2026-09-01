import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure backend root directory is always on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from app.config import settings
from app.ml.inspect_dataset import audit_and_inspect_dataset
from app.ml.preprocessing import clean_data
from app.ml.eda import generate_eda
from app.ml.feature_engineering import engineer_features
from app.ml.shap_analysis import run_shap_analysis

def train_models():
    print("=== STARTING OFFICIAL 10K PLACEMENT ML PIPELINE ===")
    
    raw_path = os.path.join(settings.RAW_DATA_DIR, "placement_10k.csv")
    reports_dir = os.path.join(settings.BASE_DIR, "reports")
    models_dir = os.path.join(settings.BASE_DIR, "models", "selected")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Audit & Inspection
    inspection = audit_and_inspect_dataset(raw_path)

    # 2. Cleaning
    df_raw = pd.read_csv(raw_path)
    df_clean, clean_report = clean_data(df_raw)

    # 3. EDA
    eda_dir = generate_eda(df_clean)

    # 4. Feature Engineering
    df_feat = engineer_features(df_clean)

    # 5. Data Leakage & Feature Selection
    target_col = "PlacementStatus"
    excluded_cols = ["StudentID", "Gender"]
    
    y = (df_feat[target_col] == "Placed").astype(int)
    X = df_feat.drop(columns=[target_col] + [c for c in excluded_cols if c in df_feat.columns])

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    eng_features = ["AcademicConsistencyIndex", "PracticalExperienceScore", "SkillsBalanceScore"]

    # Save Metadata JSON
    metadata = {
        "target": target_col,
        "excluded_features": excluded_cols,
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "engineered_features": eng_features
    }
    metadata_path = os.path.join(settings.PROCESSED_DATA_DIR, "feature_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # 6. Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 7. Preprocessing ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # 8. Train 3 Models
    models = {
        "logistic_regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=500))
        ]),
        "random_forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
        ]),
        "xgboost": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'))
        ])
    }

    eval_results = {}
    plt.figure(figsize=(8, 6))

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_prob)

        eval_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4)
        }

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.4f})")

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Placed', 'Placed'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f"confusion_matrix_{name}.png"))
        plt.close()

    # Save ROC Plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "roc_curve.png"))
    plt.close()

    # Model Comparison CSV
    comp_df = pd.DataFrame.from_dict(eval_results, orient='index')
    comp_df.index.name = 'model'
    comp_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"))

    # Select Best Model based on ROC-AUC / F1
    best_model_name = max(eval_results, key=lambda k: (eval_results[k]['roc_auc'], eval_results[k]['f1']))
    best_pipeline = models[best_model_name]

    # Save Model Artifacts
    joblib.dump(best_pipeline, os.path.join(models_dir, "selected_placement_model.joblib"))
    
    selection_data = {
        "selected_model": best_model_name,
        "selection_reason": f"Highest combination of ROC-AUC ({eval_results[best_model_name]['roc_auc']}) and F1 score ({eval_results[best_model_name]['f1']}) on untouched 20% test set.",
        "selection_metric": "roc_auc + f1",
        "metrics": eval_results
    }
    with open(os.path.join(models_dir, "model_selection.json"), 'w', encoding='utf-8') as f:
        json.dump(selection_data, f, indent=2)

    print("=== MODEL TRAINING COMPLETE ===")
    print(f"Selected Winner: {best_model_name.upper()}")

    # 9. SHAP Analysis
    run_shap_analysis()

    # 10. Generate Markdown Report
    report_md_path = os.path.join(reports_dir, "training_report.md")
    metrics = selection_data["metrics"]
    winner = selection_data["selected_model"]
    
    report_content = f"""# OFFICIAL 10K PLACEMENT GUIDANCE ML PIPELINE REPORT

## 1. Dataset Summary & Inspection
- **Raw Dataset**: `placement_10k.csv` (10,015 total rows including 15 duplicates)
- **Final Clean Dataset**: 10,000 unique student records
- **Target Variable**: `PlacementStatus` ('Placed': 9,358, 'Not Placed': 642)

## 2. Preprocessing & Feature Engineering
- **Imputation**: Median imputation for numerical features (`CGPA`, `AptitudeTestScore`); Mode imputation for categorical (`Stream`).
- **Feature Engineering**:
  - `AcademicConsistencyIndex`: Weighted average of SSC (25%), HSC (25%), and scaled CGPA (50%).
  - `PracticalExperienceScore`: Combined weighted count of Internships (x3.0), Projects (x2.0), and Certifications (x1.5).
  - `SkillsBalanceScore`: Mean of AptitudeTestScore and SoftSkillsScore.
- **Data Leakage & Fairness Exclusions**: Excluded `StudentID` (identifier) and `Gender` (fairness principle).

## 3. Model Training & Comparative Evaluation
Model evaluations on untouched 20% stratified test set:

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `{metrics['logistic_regression']['accuracy']}` | `{metrics['logistic_regression']['precision']}` | `{metrics['logistic_regression']['recall']}` | `{metrics['logistic_regression']['f1']}` | `{metrics['logistic_regression']['roc_auc']}` |
| **Random Forest** | `{metrics['random_forest']['accuracy']}` | `{metrics['random_forest']['precision']}` | `{metrics['random_forest']['recall']}` | `{metrics['random_forest']['f1']}` | `{metrics['random_forest']['roc_auc']}` |
| **XGBoost Classifier** | `{metrics['xgboost']['accuracy']}` | `{metrics['xgboost']['precision']}` | `{metrics['xgboost']['recall']}` | `{metrics['xgboost']['f1']}` | `{metrics['xgboost']['roc_auc']}` |

## 4. Selected Model & Justification
- **Winner**: `{winner.upper()}`
- **Reasoning**: {selection_data['selection_reason']}
- **Saved Model**: `models/selected/selected_placement_model.joblib`
"""

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated successfully at {report_md_path}")
    return selection_data

if __name__ == '__main__':
    train_models()
