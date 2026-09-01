import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.config import settings

def run_shap_analysis():
    models_dir = os.path.join(settings.BASE_DIR, "models", "selected")
    reports_dir = os.path.join(settings.BASE_DIR, "reports")
    
    model_path = os.path.join(models_dir, "selected_placement_model.joblib")
    meta_path = os.path.join(settings.PROCESSED_DATA_DIR, "feature_metadata.json")
    raw_path = os.path.join(settings.RAW_DATA_DIR, "placement_10k.csv")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Selected model not found. Run train_models.py first.")
        
    pipeline = joblib.load(model_path)
    df_raw = pd.read_csv(raw_path)
    
    # Preprocess a sample of 200 background rows for SHAP
    from app.ml.feature_engineering import engineer_features
    df_feat = engineer_features(df_raw.head(300))
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    excluded = meta["excluded_features"] + [meta["target"]]
    X_sample = df_feat.drop(columns=[c for c in excluded if c in df_feat.columns])
    
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    X_trans = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    
    # Linear model (Logistic Regression) uses LinearExplainer or KernelExplainer
    if hasattr(classifier, 'coef_'):
        explainer = shap.LinearExplainer(classifier, X_trans)
        shap_values = explainer.shap_values(X_trans)
    else:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_trans)

    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "shap_summary.png"))
    plt.close()

    # 2. Global Feature Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "shap_feature_importance.png"))
    plt.close()

    print("=== SHAP ANALYSIS COMPLETE ===")
    print("Saved shap_summary.png and shap_feature_importance.png")

if __name__ == '__main__':
    run_shap_analysis()
