"""Feature importance and SHAP utilities for ASCENTRA Module A."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

try:
    from .preprocessing import get_transformed_feature_names
except ImportError:  # Allows direct script-style imports from src/.
    from preprocessing import get_transformed_feature_names


def xgboost_feature_importance(model, preprocessor) -> pd.DataFrame:
    feature_names = get_transformed_feature_names(preprocessor)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_values_for_processed(model, X_processed):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_processed)
    if isinstance(values, list):
        values = values[1]
    return explainer, np.asarray(values)


def global_shap_importance(model, preprocessor, X_processed) -> pd.DataFrame:
    _, shap_values = shap_values_for_processed(model, X_processed)
    feature_names = get_transformed_feature_names(preprocessor)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_shap_table(model, preprocessor, X_processed, row_index: int = 0) -> pd.DataFrame:
    _, shap_values = shap_values_for_processed(model, X_processed)
    feature_names = get_transformed_feature_names(preprocessor)
    row_values = shap_values[row_index]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": row_values,
                "direction": np.where(row_values > 0, "increases_risk", "reduces_risk"),
            }
        )
        .assign(abs_shap=lambda frame: frame["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )
