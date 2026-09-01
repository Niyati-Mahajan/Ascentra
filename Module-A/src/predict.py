"""Prediction interface for ASCENTRA Module A."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from agent.risk_agent import (
        build_academic_risk_agent_response,
        decision_from_level,
        risk_level_from_probability,
    )
    from .config import ARTIFACT_DIR, BASE_INPUT_FEATURES, MODEL_FEATURES, RISK_THRESHOLDS
    from .explain import shap_values_for_processed
    from .gemini_advisor import generate_gemini_advice
    from .preprocessing import add_academic_features, get_transformed_feature_names
except ImportError:  # Allows `python src/predict.py`.
    from agent.risk_agent import (
        build_academic_risk_agent_response,
        decision_from_level,
        risk_level_from_probability,
    )
    from config import ARTIFACT_DIR, BASE_INPUT_FEATURES, MODEL_FEATURES, RISK_THRESHOLDS
    from explain import shap_values_for_processed
    from gemini_advisor import generate_gemini_advice
    from preprocessing import add_academic_features, get_transformed_feature_names


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> tuple[Any, Any, dict[str, Any]]:
    model = joblib.load(artifact_dir / "ascentra_xgboost.pkl")
    preprocessor = joblib.load(artifact_dir / "ascentra_preprocessor.pkl")
    metadata_path = artifact_dir / "ascentra_feature_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, preprocessor, metadata


def _normalize_student_profile(student_profile: dict[str, Any] | pd.Series) -> tuple[str | None, pd.DataFrame]:
    row = student_profile.to_dict() if isinstance(student_profile, pd.Series) else dict(student_profile)
    student_id = row.get("student_id")
    missing = [feature for feature in BASE_INPUT_FEATURES if feature not in row]
    if missing:
        raise ValueError(f"Student profile is missing required fields: {missing}")
    X_student = pd.DataFrame([{feature: row.get(feature) for feature in BASE_INPUT_FEATURES}])
    return student_id, X_student


def _transformed_values(preprocessor, X_processed) -> dict[str, Any]:
    feature_names = get_transformed_feature_names(preprocessor)
    row = np.asarray(X_processed)[0]
    return {feature: _clean_value(value) for feature, value in zip(feature_names, row)}


def _original_value(feature: str, profile: pd.DataFrame, transformed_values: dict[str, Any]) -> Any:
    if feature in profile.columns:
        return _clean_value(profile.iloc[0][feature])
    return transformed_values.get(feature)


def _validate_student_profile(profile: pd.DataFrame) -> None:
    row = profile.iloc[0]
    ranges = {
        "attendance_percentage": (0, 100),
        "ca1_score": (0, 30),
        "ca2_score": (0, 30),
        "ca3_score": (0, 30),
        "best_2_ca_average": (0, 30),
        "mid_term_score": (0, 30),
    }
    for feature, (low, high) in ranges.items():
        value = float(row[feature])
        if value < low or value > high:
            raise ValueError(f"{feature} must be between {low} and {high}.")

    ca_scores = sorted([float(row["ca1_score"]), float(row["ca2_score"]), float(row["ca3_score"])])
    expected_best_2 = round(sum(ca_scores[-2:]) / 2, 1)
    if not math.isclose(float(row["best_2_ca_average"]), expected_best_2, abs_tol=0.05):
        raise ValueError(
            "best_2_ca_average must equal the rounded average of the best two CA scores "
            f"({expected_best_2})."
        )

    if int(row["semester"]) == 1:
        previous_tgpa = row["previous_semester_tgpa"]
        if previous_tgpa is not None and not pd.isna(previous_tgpa):
            raise ValueError("Semester 1 students must use null/NaN for previous_semester_tgpa.")
    elif pd.isna(row["previous_semester_tgpa"]):
        raise ValueError("Semester 2+ students must include previous_semester_tgpa.")


def _aggregate_explanation_rows(rows: list[dict[str, Any]], profile: pd.DataFrame) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    trend_shap = 0.0
    for row in rows:
        if row["feature"].startswith("academic_trend="):
            trend_shap += row["shap_value"]
        else:
            aggregated.append(row)

    aggregated.append(
        {
            "feature": "academic_trend",
            "value": _clean_value(profile.iloc[0]["academic_trend"]),
            "shap_value": float(trend_shap),
            "direction": "increases_risk" if trend_shap > 0 else "reduces_risk",
        }
    )
    return aggregated


def predict_student_risk(
    student_profile: dict[str, Any] | pd.Series,
    model=None,
    preprocessor=None,
    metadata: dict[str, Any] | None = None,
    use_gemini: bool = False,
) -> dict[str, Any]:
    if model is None or preprocessor is None:
        model, preprocessor, loaded_metadata = load_artifacts()
        metadata = metadata or loaded_metadata

    student_id, X_student = _normalize_student_profile(student_profile)
    _validate_student_profile(X_student)
    X_model = add_academic_features(X_student).loc[:, list(MODEL_FEATURES)]
    X_processed = preprocessor.transform(X_model)
    risk_probability = float(model.predict_proba(X_processed)[0][1])
    thresholds = (metadata or {}).get("risk_thresholds", RISK_THRESHOLDS)
    risk_level = risk_level_from_probability(risk_probability, thresholds)
    decision = decision_from_level(risk_level)

    _, shap_values = shap_values_for_processed(model, X_processed)
    feature_names = get_transformed_feature_names(preprocessor)
    processed_value_map = _transformed_values(preprocessor, X_processed)
    local_rows = []
    for feature, shap_value in zip(feature_names, shap_values[0]):
        local_rows.append(
            {
                "feature": feature,
            "value": _original_value(feature, X_model, processed_value_map),
                "shap_value": float(shap_value),
                "direction": "increases_risk" if shap_value > 0 else "reduces_risk",
            }
        )

    aggregated_rows = _aggregate_explanation_rows(local_rows, X_model)

    risk_factors = sorted(
        [row for row in aggregated_rows if row["shap_value"] > 0],
        key=lambda row: abs(row["shap_value"]),
        reverse=True,
    )[:5]
    protective_factors = sorted(
        [row for row in aggregated_rows if row["shap_value"] < 0],
        key=lambda row: abs(row["shap_value"]),
        reverse=True,
    )[:5]

    prediction = {
        "student_id": student_id,
        "risk_probability": round(risk_probability, 4),
        "risk_level": risk_level,
        "decision": decision,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }
    agent_response = build_academic_risk_agent_response(prediction, risk_factors)
    prediction["interventions"] = agent_response["priority_interventions"]
    prediction["agent_response"] = agent_response
    if use_gemini:
        prediction["gemini_advisor"] = generate_gemini_advice(prediction)
    return prediction


def academic_risk_prediction(
    student_profile: dict[str, Any] | pd.Series,
    use_gemini: bool = False,
) -> dict[str, Any]:
    """Frontend/API-ready prediction function returning JSON-serializable data."""
    return predict_student_risk(student_profile, use_gemini=use_gemini)


if __name__ == "__main__":
    example = {
        "student_id": "STU_EXAMPLE",
        "semester": 2,
        "attendance_percentage": 54.0,
        "ca1_score": 9.0,
        "ca2_score": 12.0,
        "ca3_score": 13.0,
        "best_2_ca_average": 12.5,
        "mid_term_score": 11.0,
        "previous_semester_tgpa": 5.8,
        "academic_trend": "Declining",
    }
    print(json.dumps(academic_risk_prediction(example), indent=2))
