from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from .schema import FEATURE_COLUMNS, MODEL_PATH, NEGATIVE_LABEL, POSITIVE_LABEL
from .validation import validate_input_row


def load_model(model_path: str = str(MODEL_PATH)):
    return joblib.load(model_path)


def predict_for_row(model, row: dict[str, Any]) -> dict[str, Any]:
    validation_result = validate_input_row(row)
    if not validation_result["valid"]:
        raise ValueError(validation_result["error"])

    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    classifier = model.named_steps.get("classifier")
    classes = list(getattr(classifier, "classes_", []))
    if not classes:
        raise ValueError("Saved classifier does not expose its fitted classes.")
    positive_class = 1 if 1 in classes else POSITIVE_LABEL
    if positive_class not in classes:
        raise ValueError(f"Positive class {POSITIVE_LABEL!r} is not present in {classes!r}.")

    probabilities = model.predict_proba(input_df)[0]
    probability_by_class = {
        label: float(probabilities[index])
        for index, label in enumerate(classes)
    }
    placed_probability = probability_by_class[positive_class]
    predicted_class = model.predict(input_df)[0]
    label_by_class = {
        0: NEGATIVE_LABEL,
        1: POSITIVE_LABEL,
        NEGATIVE_LABEL: NEGATIVE_LABEL,
        POSITIVE_LABEL: POSITIVE_LABEL,
    }
    predicted_label = label_by_class.get(predicted_class)
    if predicted_label is None:
        raise ValueError(f"Unexpected predicted class: {predicted_class!r}")

    return {
        "prediction": predicted_label,
        "probability": placed_probability,
        "probability_label": POSITIVE_LABEL,
        "predicted_class": predicted_class,
        "predicted_class_probability": probability_by_class[predicted_class],
        "probability_by_class": {
            NEGATIVE_LABEL: probability_by_class[
                0 if 0 in probability_by_class else NEGATIVE_LABEL
            ],
            POSITIVE_LABEL: probability_by_class[positive_class],
        },
    }
