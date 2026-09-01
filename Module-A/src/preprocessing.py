"""Data validation and preprocessing for ASCENTRA Module A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BASE_INPUT_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_TRENDS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    REMOVED_FEATURES,
    TARGET_CANDIDATES,
)


@dataclass(frozen=True)
class ValidationResult:
    target_column: str
    shape: tuple[int, int]
    missing_values: dict[str, int]
    duplicate_rows: int
    target_distribution: dict[Any, int]
    semester_distribution: dict[Any, int]
    categorical_values: dict[str, list[Any]]
    numerical_ranges: dict[str, dict[str, float | None]]
    warnings: list[str]


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the existing target/risk outcome column without renaming it."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not detect target column. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
    )


def validate_refined_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate the refined dataset and raise on rules that must not be violated."""
    target_column = detect_target_column(df)
    warnings: list[str] = []

    missing_features = [feature for feature in BASE_INPUT_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required refined model features: {missing_features}")

    old_features_present = [feature for feature in REMOVED_FEATURES if feature in df.columns]
    if old_features_present:
        raise ValueError(f"Removed/obsolete features are still present: {old_features_present}")

    if df[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains NaN values.")

    range_checks = {
        "attendance_percentage": (0, 100),
        "ca1_score": (0, 30),
        "ca2_score": (0, 30),
        "ca3_score": (0, 30),
        "mid_term_score": (0, 30),
        "best_2_ca_average": (0, 30),
    }
    for column, (low, high) in range_checks.items():
        out_of_range = ~df[column].between(low, high, inclusive="both")
        if out_of_range.any():
            raise ValueError(
                f"{column} must be between {low} and {high}; "
                f"found {int(out_of_range.sum())} invalid rows."
            )

    recomputed_best_2 = np.sort(
        df[["ca1_score", "ca2_score", "ca3_score"]].to_numpy(dtype=float),
        axis=1,
    )[:, -2:].mean(axis=1)
    if not np.allclose(df["best_2_ca_average"].to_numpy(dtype=float), np.round(recomputed_best_2, 1)):
        raise ValueError("best_2_ca_average does not match the rounded average of the best two CA scores.")

    semester_1 = df["semester"].eq(1)
    if not df.loc[semester_1, "previous_semester_tgpa"].isna().all():
        raise ValueError("Semester 1 rows must have previous_semester_tgpa as NaN.")
    if df.loc[~semester_1, "previous_semester_tgpa"].isna().any():
        raise ValueError("Semester 2+ rows must have previous_semester_tgpa values.")

    unexpected_trends = sorted(set(df["academic_trend"].dropna()) - set(EXPECTED_TRENDS))
    if unexpected_trends:
        raise ValueError(f"academic_trend contains unexpected values: {unexpected_trends}")

    numerical_ranges: dict[str, dict[str, float | None]] = {}
    for column in BASE_NUMERIC_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")
        numerical_ranges[column] = {
            "min": None if values.dropna().empty else float(values.min()),
            "max": None if values.dropna().empty else float(values.max()),
            "mean": None if values.dropna().empty else float(values.mean()),
        }

    if df.duplicated().any():
        warnings.append(f"Dataset contains {int(df.duplicated().sum())} duplicate rows.")

    return ValidationResult(
        target_column=target_column,
        shape=df.shape,
        missing_values={column: int(count) for column, count in df.isna().sum().items()},
        duplicate_rows=int(df.duplicated().sum()),
        target_distribution={key: int(value) for key, value in df[target_column].value_counts(dropna=False).items()},
        semester_distribution={key: int(value) for key, value in df["semester"].value_counts(dropna=False).sort_index().items()},
        categorical_values={
            column: sorted(df[column].dropna().unique().tolist()) for column in CATEGORICAL_FEATURES
        },
        numerical_ranges=numerical_ranges,
        warnings=warnings,
    )


def add_academic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add small, academically meaningful derived features without changing raw data."""
    frame = X.copy()
    ca_columns = ["ca1_score", "ca2_score", "ca3_score"]
    frame["ca_average"] = frame[ca_columns].mean(axis=1).round(3)
    frame["ca_improvement"] = frame["ca3_score"] - frame["ca1_score"]
    frame["ca_consistency"] = frame[ca_columns].std(axis=1, ddof=0).round(3)
    frame["attendance_best2_interaction"] = (
        frame["attendance_percentage"] / 100.0
    ) * frame["best_2_ca_average"]
    frame["attendance_midterm_interaction"] = (
        frame["attendance_percentage"] / 100.0
    ) * frame["mid_term_score"]
    frame["tgpa_best2_interaction"] = (
        frame["previous_semester_tgpa"] / 10.0
    ) * frame["best_2_ca_average"]
    frame["previous_tgpa_available"] = frame["previous_semester_tgpa"].notna().astype(int)
    return frame


def build_preprocessor(
    numeric_features: tuple[str, ...] | list[str] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] | list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build a training-only fitted preprocessing pipeline.

    Semester is intentionally treated as a numeric ordinal variable because academic
    progression is ordered from Semester 1 through Semester 8.
    """
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(
    df: pd.DataFrame,
    model_features: tuple[str, ...] | list[str] = MODEL_FEATURES,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return model features and target while excluding IDs and unused columns."""
    target_column = detect_target_column(df)
    base = df.loc[:, list(BASE_INPUT_FEATURES)].copy()
    X = add_academic_features(base).loc[:, list(model_features)]
    y = df[target_column].copy()
    if len(X) != len(y):
        raise ValueError("X and y row counts do not match.")
    if y.isna().any():
        raise ValueError(f"Target column {target_column!r} contains NaN values.")
    return X, y, target_column


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name in preprocessor.get_feature_names_out():
        readable = name.replace("academic_trend_", "academic_trend=")
        readable = readable.replace("missingindicator_previous_semester_tgpa", "previous_semester_tgpa_missing")
        names.append(readable)
    return names
