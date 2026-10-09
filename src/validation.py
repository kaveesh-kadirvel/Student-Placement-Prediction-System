from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if df is None or df.empty:
        return {"valid": False, "issues": ["Dataset is empty."], "shape": (0, 0)}

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].isna().any():
        issues.append("Target column contains missing values.")

    if df.duplicated().any():
        issues.append(f"Dataset contains {int(df.duplicated().sum())} duplicate row(s).")

    if df.isna().sum().sum() > 0:
        na_columns = df.columns[df.isna().any()].tolist()
        issues.append(f"Missing values detected in: {na_columns}")

    return {
        "valid": not issues,
        "issues": issues,
        "shape": df.shape,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def check_split_integrity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    overlap = set(train_df["Student_ID"]).intersection(set(test_df["Student_ID"]))
    return {
        "shared_student_ids": sorted(list(overlap))[:10],
        "overlap_count": len(overlap),
        "train_unique_ids": train_df["Student_ID"].nunique(),
        "test_unique_ids": test_df["Student_ID"].nunique(),
    }


def encode_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        encoded = series.map({"Not Placed": 0, "Placed": 1})
        if encoded.isna().any():
            raise ValueError(f"Unrecognized target labels found: {sorted(series.unique().tolist())}")
        return encoded.astype(int)
    return series.astype(int)


def validate_input_row(values: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in FEATURE_COLUMNS if name not in values or values[name] in (None, "")]
    if missing:
        return {"valid": False, "error": f"Missing required fields: {missing}"}

    invalid: list[str] = []
    for name in NUMERIC_COLUMNS:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            invalid.append(f"{name} must be a finite number")
            continue

        lower, upper = {
            "Age": (17, 35),
            "CGPA": (0, 10),
            "Internships": (0, 10),
            "Projects": (0, 10),
            "Coding_Skills": (1, 10),
            "Communication_Skills": (1, 10),
            "Aptitude_Test_Score": (0, 100),
            "Soft_Skills_Rating": (1, 10),
            "Certifications": (0, 10),
            "Backlogs": (0, 10),
        }[name]
        if not lower <= value <= upper:
            invalid.append(f"{name} must be between {lower} and {upper}")
        elif name != "CGPA" and not float(value).is_integer():
            invalid.append(f"{name} must be an integer")

    for name in CATEGORICAL_COLUMNS:
        if not isinstance(values[name], str) or not values[name].strip():
            invalid.append(f"{name} must be a non-empty string")

    if invalid:
        return {"valid": False, "error": "; ".join(invalid)}
    return {"valid": True}
