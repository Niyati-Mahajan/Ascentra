import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from app.config import settings

def train_resume_role_model():
    raw_path = os.path.join(settings.RAW_DATA_DIR, "training_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Training dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    X = df["Resume Text"].fillna("")
    y = df["Job Role"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print("=== RESUME-ROLE MODEL EVALUATION ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save models
    model_path = os.path.join(settings.MODELS_DIR, "resume_role_classifier.joblib")
    vectorizer_path = os.path.join(settings.MODELS_DIR, "resume_tfidf_vectorizer.joblib")
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics = {
        "dataset": "training_data.csv (Dataset D)",
        "sample_count": len(df),
        "target": "Job Role",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4)
    }
    return metrics

if __name__ == '__main__':
    train_resume_role_model()
