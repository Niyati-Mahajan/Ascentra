"""Train and optimize ASCENTRA Module A on the refined academic-risk dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:
    from .config import (
        ARTIFACT_DIR,
        CATEGORICAL_FEATURES,
        DATA_PATH,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        RANDOM_STATE,
        REPORT_DIR,
        RISK_THRESHOLDS,
        TEST_SIZE,
    )
    from .explain import global_shap_importance, local_shap_table, xgboost_feature_importance
    from .preprocessing import build_preprocessor, split_features_target, validate_refined_dataset
except ImportError:  # Allows `python src/train.py`.
    from config import (
        ARTIFACT_DIR,
        CATEGORICAL_FEATURES,
        DATA_PATH,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        RANDOM_STATE,
        REPORT_DIR,
        RISK_THRESHOLDS,
        TEST_SIZE,
    )
    from explain import global_shap_importance, local_shap_table, xgboost_feature_importance
    from preprocessing import build_preprocessor, split_features_target, validate_refined_dataset


BASELINE_METRICS = {
    "accuracy": 0.6373,
    "precision": 0.5850,
    "recall": 0.4799,
    "f1": 0.5272,
    "roc_auc": 0.6761,
}


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    return len(counts) > 1 and counts.min() >= 2


def _make_xgb(scale_pos_weight: float = 1.0, **overrides) -> XGBClassifier:
    params = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def _cv_summary(estimator, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
    )
    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "precision_mean": float(scores["test_precision"].mean()),
        "recall_mean": float(scores["test_recall"].mean()),
        "f1_mean": float(scores["test_f1"].mean()),
        "f1_std": float(scores["test_f1"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
    }


def _build_model_comparison(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost Baseline": _make_xgb(scale_pos_weight=scale_pos_weight),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    comparison = []
    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        comparison.append({"model": name, **_cv_summary(pipeline, X_train, y_train, cv)})
    return sorted(
        comparison,
        key=lambda row: (row["f1_mean"], row["recall_mean"], row["roc_auc_mean"]),
        reverse=True,
    )


def _compare_midterm_redundancy(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    variants = {
        "A_with_mid_term": list(MODEL_FEATURES),
        "B_without_mid_term": [feature for feature in MODEL_FEATURES if feature != "mid_term_score"],
    }
    results = []
    for name, features in variants.items():
        numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
        categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric, categorical)),
                ("model", _make_xgb(scale_pos_weight=scale_pos_weight)),
            ]
        )
        results.append({"variant": name, **_cv_summary(pipeline, X_train[features], y_train, cv)})
    return results


def _tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> tuple[dict, dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", _make_xgb(scale_pos_weight=scale_pos_weight)),
        ]
    )
    param_distributions = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.75, 0.9, 1.0],
        "model__colsample_bytree": [0.75, 0.9, 1.0],
        "model__gamma": [0, 0.1, 0.3],
        "model__reg_alpha": [0, 0.01, 0.1],
        "model__reg_lambda": [0.8, 1.0, 1.5, 2.0],
        "model__scale_pos_weight": [1.0, scale_pos_weight],
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=18,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
        if key.startswith("model__")
    }
    return best_params, {"best_f1_mean": float(search.best_score_), "best_params": best_params}


def _select_threshold_from_oof(model_params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[float, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", _make_xgb(**model_params)),
        ]
    )
    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    rows = []
    for threshold in np.arange(0.30, 0.701, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_train, predictions, zero_division=0),
                "recall": recall_score(y_train, predictions, zero_division=0),
                "f1": f1_score(y_train, predictions, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    viable = table[table["recall"] >= 0.60]
    selected = (
        viable.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        if not viable.empty
        else table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    )
    return float(selected["threshold"]), table


def _evaluate_at_threshold(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }
    return metrics, predictions


def _target_and_leakage_report(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, target_column: str) -> dict:
    numeric = X.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    high_pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if value >= 0.85:
                high_pairs.append({"feature_1": left, "feature_2": right, "absolute_correlation": float(value)})
    target_corr = {
        column: float(numeric[column].corr(y))
        for column in numeric.columns
        if numeric[column].notna().sum() > 1
    }
    return {
        "target_generation": (
            "The refined-data notebook samples academic_risk from a synthetic probabilistic risk score using "
            "attendance, best_2_ca_average, mid_term_score, previous_semester_tgpa, academic_trend, and noise."
        ),
        "target_column": target_column,
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "class_percentage": (y.value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
        "is_synthetic_target": True,
        "known_synthetic_limitation": (
            "The target is generated from several predictor variables. This is acceptable for the current synthetic "
            "dataset exercise but limits real-world claims until validated against institutional outcomes."
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "highly_correlated_feature_pairs": sorted(
            high_pairs,
            key=lambda row: row["absolute_correlation"],
            reverse=True,
        ),
        "target_correlations": dict(sorted(target_corr.items(), key=lambda item: abs(item[1]), reverse=True)),
        "target_used_as_feature": target_column in X.columns,
        "student_id_used_as_feature": "student_id" in X.columns,
    }


def _save_evaluation_plots(report_dir: Path, y_test, y_pred, y_proba) -> None:
    confusion_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    confusion_display.ax_.set_title("ASCENTRA Confusion Matrix")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
    roc_display.ax_.set_title("ASCENTRA ROC Curve")
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=160)
    plt.close()


def _write_markdown_report(report_dir: Path, report: dict) -> None:
    metrics = report["metrics"]
    baseline = report["baseline_metrics"]
    changes = report["metric_change_percentage_points"]
    validation = report["validation"]
    metadata = report["metadata"]
    shap_top = report["shap_importance_top"][:10]
    profile_lines: list[str] = []
    profiles_path = report_dir / "representative_profile_results.json"
    if profiles_path.exists():
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        for item in profiles:
            prediction = item["prediction"]
            profile_lines.append(
                f"- {prediction['student_id']}: {prediction['risk_probability']:.4f}, "
                f"{prediction['risk_level']}, {prediction['decision']}"
            )

    lines = [
        "# ASCENTRA Module A Technical Report",
        "",
        "## Dataset And Target",
        f"- Rows: {validation['shape'][0]}",
        f"- Columns: {validation['shape'][1]}",
        f"- Target column: `{validation['target_column']}`",
        f"- Class distribution: {report['target_analysis']['class_distribution']}",
        f"- Class percentage: {report['target_analysis']['class_percentage']}",
        f"- Target generation: {report['target_analysis']['target_generation']}",
        f"- Synthetic-target limitation: {report['target_analysis']['known_synthetic_limitation']}",
        "",
        "## Leakage And Redundancy",
        "- `student_id` and target are excluded from features.",
        "- Preprocessing, model search, and threshold tuning use training data only.",
        "- Semester 1 previous TGPA remains missing until pipeline imputation; it is not replaced by 0.",
        f"- High-correlation feature pairs >= 0.85: {report['target_analysis']['highly_correlated_feature_pairs'][:8]}",
        f"- Mid-term comparison: {report['midterm_redundancy']}",
        "",
        "## Feature Engineering",
        f"- Final model features: {', '.join(f'`{feature}`' for feature in metadata['model_features'])}",
        f"- Engineered features: {', '.join(f'`{feature}`' for feature in metadata['engineered_features'])}",
        "",
        "## Cross-Validation Model Comparison",
        "| Model | Accuracy | Accuracy SD | Precision | Recall | F1 | F1 SD | ROC-AUC | ROC-AUC SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["model_comparison"]:
        lines.append(
            f"| {row['model']} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | "
            f"{row['precision_mean']:.4f} | {row['recall_mean']:.4f} | {row['f1_mean']:.4f} | "
            f"{row['f1_std']:.4f} | {row['roc_auc_mean']:.4f} | {row['roc_auc_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Final Model",
            "- Algorithm: tuned XGBoost binary classifier",
            f"- Classification threshold: {metadata['classification_threshold']:.2f}",
            f"- Hyperparameters: `{metadata['model_params']}`",
            "",
            "## Final Test Evaluation",
            "| Metric | Baseline | Improved | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lines.append(f"| {key} | {baseline[key] * 100:.2f}% | {metrics[key] * 100:.2f}% | {changes[key]:+.2f} pp |")
    lines.extend(
        [
            f"- Confusion matrix [[TN, FP], [FN, TP]]: {report['confusion_matrix']}",
            "",
            "## SHAP",
            "- Positive SHAP pushes toward the risk class; negative SHAP pushes away from risk.",
            "| Feature | Mean Absolute SHAP |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {row['feature']} | {row['mean_absolute_shap']:.6f} |" for row in shap_top)
    lines.extend(
        [
            "",
            "## Representative Profiles",
            *(profile_lines or ["- Run `python -m src.representative_profiles` to generate profile results."]),
            "",
            "## API",
            "- `POST /api/academic-risk` is preserved.",
            "- Response includes student ID, probability, risk level, SHAP-ranked risk/protective factors, and linked interventions.",
        ]
    )
    (report_dir / "module_a_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_ascentra_model(
    data_path: Path = DATA_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict:
    df = pd.read_csv(data_path)
    validation = validate_refined_dataset(df)
    X, y, target_column = split_features_target(df)
    y = y.astype(int)

    if not _can_stratify(y):
        raise ValueError(
            "Stratified split is unsafe because the target has fewer than two classes "
            "or at least one class has fewer than two samples."
        )

    target_analysis = _target_and_leakage_report(df, X, y, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts.get(0, 1) / class_counts.get(1, 1))

    midterm_redundancy = _compare_midterm_redundancy(X_train, y_train, scale_pos_weight)
    model_comparison = _build_model_comparison(X_train, y_train, scale_pos_weight)
    best_xgb_params, tuning_report = _tune_xgboost(X_train, y_train, scale_pos_weight)
    tuned_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", _make_xgb(**best_xgb_params)),
        ]
    )
    tuned_xgb_cv = _cv_summary(
        tuned_pipeline,
        X_train,
        y_train,
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    model_comparison.append({"model": "XGBoost Tuned", **tuned_xgb_cv})
    model_comparison = sorted(
        model_comparison,
        key=lambda row: (row["f1_mean"], row["recall_mean"], row["roc_auc_mean"]),
        reverse=True,
    )

    selected_threshold, threshold_table = _select_threshold_from_oof(best_xgb_params, X_train, y_train)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model = _make_xgb(**best_xgb_params)
    model.fit(X_train_processed, y_train)

    y_proba = model.predict_proba(X_test_processed)[:, 1]
    metrics, y_pred = _evaluate_at_threshold(y_test, y_proba, selected_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    feature_importance = xgboost_feature_importance(model, preprocessor)
    shap_importance = global_shap_importance(model, preprocessor, X_test_processed)
    local_shap = local_shap_table(model, preprocessor, X_test_processed, row_index=0)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "ascentra_xgboost.pkl")
    joblib.dump(preprocessor, artifact_dir / "ascentra_preprocessor.pkl")

    optimized_thresholds = {
        "low_max": RISK_THRESHOLDS["low_max"],
        "medium_max": RISK_THRESHOLDS["medium_max"],
        "classification_threshold": float(selected_threshold),
    }
    metadata = {
        "dataset": str(data_path),
        "target_column": target_column,
        "model_features": X.columns.tolist(),
        "engineered_features": [
            "ca_average",
            "ca_improvement",
            "ca_consistency",
            "attendance_best2_interaction",
            "attendance_midterm_interaction",
            "tgpa_best2_interaction",
            "previous_tgpa_available",
        ],
        "excluded_columns": [
            column for column in df.columns if column not in X.columns and column != target_column
        ],
        "semester_handling": "semester is treated as a numeric ordinal feature because academic progression is ordered.",
        "missing_value_handling": "previous_semester_tgpa remains NaN for Semester 1 and is imputed inside the training-fitted preprocessing pipeline with a missingness indicator.",
        "risk_thresholds": optimized_thresholds,
        "classification_threshold": float(selected_threshold),
        "threshold_selection": "Selected from 5-fold out-of-fold training predictions, preferring F1 with recall >= 0.60 for early warning.",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_params": model.get_params(),
        "best_xgb_search": tuning_report,
    }
    (artifact_dir / "ascentra_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    (artifact_dir / "ascentra_risk_thresholds.json").write_text(
        json.dumps(optimized_thresholds, indent=2),
        encoding="utf-8",
    )

    feature_importance.to_csv(report_dir / "xgboost_feature_importance.csv", index=False)
    shap_importance.to_csv(report_dir / "shap_global_importance.csv", index=False)
    local_shap.to_csv(report_dir / "shap_local_example.csv", index=False)
    pd.DataFrame(model_comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(midterm_redundancy).to_csv(report_dir / "midterm_redundancy_cv.csv", index=False)
    threshold_table.to_csv(report_dir / "threshold_optimization.csv", index=False)

    report = {
        "validation": validation.__dict__,
        "target_analysis": target_analysis,
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "stratified": True,
            "target_distribution_train": y_train.value_counts().to_dict(),
            "target_distribution_test": y_test.value_counts().to_dict(),
        },
        "baseline_metrics": BASELINE_METRICS,
        "metrics": metrics,
        "metric_change_percentage_points": {
            key: float((metrics[key] - BASELINE_METRICS[key]) * 100) for key in BASELINE_METRICS
        },
        "classification_threshold": float(selected_threshold),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "model_comparison": model_comparison,
        "midterm_redundancy": midterm_redundancy,
        "threshold_optimization_top": threshold_table.sort_values(
            ["f1", "recall", "precision"], ascending=False
        ).head(10).to_dict(orient="records"),
        "feature_importance_top": feature_importance.head(20).to_dict(orient="records"),
        "shap_importance_top": shap_importance.head(20).to_dict(orient="records"),
        "local_shap_example": local_shap.head(12).to_dict(orient="records"),
        "metadata": metadata,
        "data_leakage_check": {
            "student_id_used_as_feature": "student_id" in X.columns,
            "target_used_as_feature": target_column in X.columns,
            "preprocessor_fit_scope": "training data only",
            "cv_and_tuning_scope": "training data only",
            "threshold_tuning_scope": "training out-of-fold predictions only",
            "previous_tgpa_semester_1_policy": "NaN retained until pipeline imputation; not replaced with 0",
            "removed_features_used": [
                column
                for column in (
                    "quiz_average",
                    "late_submission_count",
                    "lms_resource_access",
                    "assignment_average",
                    "lms_login_frequency",
                )
                if column in X.columns
            ],
        },
    }
    (report_dir / "module_a_training_report.json").write_text(
        json.dumps(report, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    _save_evaluation_plots(report_dir, y_test, y_pred, y_proba)
    _write_markdown_report(report_dir, report)

    print("=" * 60)
    print("ASCENTRA OPTIMIZED XGBOOST RESULTS")
    print("=" * 34)
    print(f"Threshold : {selected_threshold:.2f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("=" * 34)
    print("Confusion Matrix:", conf_matrix)
    print("\nModel comparison:")
    print(pd.DataFrame(model_comparison).to_string(index=False))
    print("\nTop SHAP Features:")
    print(shap_importance.head(10).to_string(index=False))
    return report


if __name__ == "__main__":
    train_ascentra_model()
