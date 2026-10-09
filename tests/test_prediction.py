import pandas as pd
from types import SimpleNamespace
import pytest

from src.predict import load_model, predict_for_row
from src.schema import FEATURE_COLUMNS


def test_saved_model_returns_valid_prediction():
    model = load_model("placement_model.pkl")
    assert set(model.feature_names_in_) == set(FEATURE_COLUMNS)
    assert "Student_ID" not in model.feature_names_in_
    sample = {
        "Age": 22,
        "Gender": "Female",
        "Degree": "B.Tech",
        "Branch": "IT",
        "CGPA": 8.2,
        "Internships": 2,
        "Projects": 4,
        "Coding_Skills": 7,
        "Communication_Skills": 7,
        "Aptitude_Test_Score": 80,
        "Soft_Skills_Rating": 7,
        "Certifications": 2,
        "Backlogs": 0,
    }
    result = predict_for_row(model, sample)
    sample_frame = pd.DataFrame([sample])
    predicted_probabilities = model.predict_proba(sample_frame)[0]
    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    assert result["prediction"] in {"Placed", "Not Placed"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["probability"] == predicted_probabilities[positive_index]
    assert result["probability_label"] == "Placed"
    assert result["prediction"] == (
        "Placed" if model.predict(sample_frame)[0] == 1 else "Not Placed"
    )
    assert result["predicted_class_probability"] == result["probability_by_class"][result["prediction"]]


def test_invalid_student_input_is_rejected():
    model = load_model("placement_model.pkl")
    sample = {
        "Age": 22,
        "Gender": "Female",
        "Degree": "B.Tech",
        "Branch": "IT",
        "CGPA": 12.0,
        "Internships": 2,
        "Projects": 4,
        "Coding_Skills": 7,
        "Communication_Skills": 7,
        "Aptitude_Test_Score": 80,
        "Soft_Skills_Rating": 7,
        "Certifications": 2,
        "Backlogs": 0,
    }

    with pytest.raises(ValueError, match="CGPA must be between"):
        predict_for_row(model, sample)


def test_probability_uses_fitted_class_order_not_assumed_column_order():
    model = SimpleNamespace(
        named_steps={"classifier": SimpleNamespace(classes_=[1, 0])},
        predict_proba=lambda frame: [[0.8, 0.2]],
        predict=lambda frame: [0],
    )
    sample = {
        "Age": 22,
        "Gender": "Female",
        "Degree": "B.Tech",
        "Branch": "IT",
        "CGPA": 8.2,
        "Internships": 2,
        "Projects": 4,
        "Coding_Skills": 7,
        "Communication_Skills": 7,
        "Aptitude_Test_Score": 80,
        "Soft_Skills_Rating": 7,
        "Certifications": 2,
        "Backlogs": 0,
    }

    result = predict_for_row(model, sample)

    assert result["probability"] == 0.8
    assert result["probability_by_class"] == {"Not Placed": 0.2, "Placed": 0.8}
    assert result["prediction"] == "Not Placed"
    assert result["predicted_class_probability"] == 0.2
