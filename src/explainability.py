from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, POSITIVE_LABEL, TRAIN_PATH

try:  # optional dependency
    import shap
except Exception:  # pragma: no cover
    shap = None


def _default_background() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data required for SHAP baseline was not found: {TRAIN_PATH}")
    training_features = pd.read_csv(TRAIN_PATH, usecols=FEATURE_COLUMNS)
    return training_features.sample(min(100, len(training_features)), random_state=42)


def _get_shap_values(
    model,
    X_df: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
    explained_class=None,
):
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X_df)
    background = _default_background() if background_df is None else background_df
    missing = [column for column in FEATURE_COLUMNS if column not in background.columns]
    if missing:
        raise ValueError(f"SHAP background is missing required features: {missing}")
    transformed_background = preprocessor.transform(background[FEATURE_COLUMNS])
    feature_names = list(preprocessor.get_feature_names_out())
    classifier = model.named_steps["classifier"]
    explainer = shap.Explainer(
        classifier,
        transformed_background,
        feature_names=feature_names,
    )
    shap_values = explainer(transformed)

    classes = list(getattr(classifier, "classes_", []))
    if explained_class is None:
        explained_class = 1 if 1 in classes else POSITIVE_LABEL
    if explained_class not in classes:
        raise ValueError(f"Explanation class {explained_class!r} is not in fitted classes {classes!r}.")
    class_index = classes.index(explained_class)
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, class_index]
    elif values.ndim != 2:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    return shap_values, values, feature_names, transformed, explained_class


def explain_model(
    model,
    X_df: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
):
    if shap is None:
        return {
            "status": "unavailable",
            "message": "SHAP is not installed or the model is not explainable.",
            "global_top_features": [],
        }

    if X_df.empty:
        raise ValueError("At least one row is required for SHAP explanation.")
    _, values, feature_names, _, explained_class = _get_shap_values(
        model, X_df, background_df
    )
    importance = np.abs(values).mean(axis=0)

    ranked = sorted(
        zip(feature_names, importance.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "status": "available",
        "feature_names": [name for name, _ in ranked],
        "importance": [float(score) for _, score in ranked],
        "global_top_features": [
            {"feature": name, "importance": float(score)} for name, score in ranked
        ],
        "explained_rows": len(X_df),
        "explained_class": explained_class,
    }


def explain_single_prediction(
    model,
    X_df: pd.DataFrame,
    row_index: int = 0,
    background_df: pd.DataFrame | None = None,
):
    if shap is None:
        return {
            "status": "unavailable",
            "message": "SHAP is not installed or the model is not explainable.",
            "top_features": [],
        }
    if X_df.empty or row_index < 0 or row_index >= len(X_df):
        raise IndexError("row_index must select an existing input row.")

    selected = X_df.iloc[[row_index]]
    predicted_class = model.predict(selected)[0]
    _, values, feature_names, transformed, explained_class = _get_shap_values(
        model, selected, background_df, explained_class=predicted_class
    )
    row_values = values[0]
    row_data = np.asarray(transformed)[0]
    ordered = sorted(
        zip(feature_names, row_data.tolist(), row_values.tolist()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:5]
    return {
        "top_features": [
            {
                "feature": str(feature),
                "value": float(value),
                "contribution": float(contribution),
            }
            for feature, value, contribution in ordered
        ],
        "status": "available",
        "explained_class": explained_class,
    }
