import pandas as pd

from src.schema import FEATURE_COLUMNS
from src.validation import validate_dataframe, validate_input_row


def valid_input():
    return {
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


def test_train_dataset_passes_validation():
    df = pd.read_csv("train.csv")
    result = validate_dataframe(df)
    assert result["valid"] is True
    assert result["shape"][0] > 0


def test_dataset_missing_target_is_rejected():
    df = pd.read_csv("train.csv").drop(columns=["Placement_Status"])
    result = validate_dataframe(df)
    assert result["valid"] is False
    assert any("Missing required columns" in issue for issue in result["issues"])


def test_feature_schema_excludes_student_identifier():
    assert "Student_ID" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_valid_student_input_passes_validation():
    assert validate_input_row(valid_input())["valid"] is True


def test_non_finite_or_out_of_range_input_is_rejected():
    row = valid_input()
    row["CGPA"] = float("nan")
    assert validate_input_row(row)["valid"] is False

    row = valid_input()
    row["CGPA"] = 12
    assert validate_input_row(row)["valid"] is False
