from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import MODEL_PATH, REPORTS_DIR, TEST_PATH, TRAIN_PATH
from src.validation import check_split_integrity, validate_dataframe


def build_audit_report() -> str:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_validation = validate_dataframe(train_df)
    test_validation = validate_dataframe(test_df)
    split_report = check_split_integrity(train_df, test_df)

    train_distribution = train_df["Placement_Status"].value_counts().to_dict()
    test_distribution = test_df["Placement_Status"].value_counts().to_dict()
    unique_ids = train_df["Student_ID"].nunique(), test_df["Student_ID"].nunique()
    record_columns = [column for column in train_df.columns if column != "Student_ID"]
    feature_columns = [
        column
        for column in record_columns
        if column != "Placement_Status"
    ]
    exact_record_overlap = len(
        train_df[record_columns]
        .drop_duplicates()
        .merge(test_df[record_columns].drop_duplicates(), how="inner")
    )
    feature_vector_overlap = len(
        train_df[feature_columns]
        .drop_duplicates()
        .merge(test_df[feature_columns].drop_duplicates(), how="inner")
    )
    if MODEL_PATH.exists():
        saved_model = joblib.load(MODEL_PATH)
        estimator_name = type(saved_model.named_steps["classifier"]).__name__
        artifact_features = list(saved_model.feature_names_in_)
    else:
        estimator_name = "missing"
        artifact_features = []

    strongest_features = []
    for column in [
        "CGPA",
        "Internships",
        "Projects",
        "Coding_Skills",
        "Communication_Skills",
        "Aptitude_Test_Score",
        "Backlogs",
    ]:
        placed_mean = train_df.loc[train_df["Placement_Status"] == "Placed", column].mean()
        not_placed_mean = train_df.loc[train_df["Placement_Status"] == "Not Placed", column].mean()
        strongest_features.append(f"- {column}: placed mean {placed_mean:.2f} vs not-placed mean {not_placed_mean:.2f}")

    report = f"""# Dataset Audit

## Scope
This audit reviews the current repository files, the train/test CSVs, and the saved model artifacts before trusting the project for deployment.

## 1. Dataset structure
- Training rows: {len(train_df)}
- Test rows: {len(test_df)}
- Columns: {list(train_df.columns)}
- Missing values: {train_df.isna().sum().to_dict()}
- Duplicate rows: {int(train_df.duplicated().sum())}
- Unique Student_ID values: train={unique_ids[0]}, test={unique_ids[1]}
- Train target distribution: {train_distribution}
- Test target distribution: {test_distribution}

## 2. Split integrity and overlap
- Overlap in Student_ID between train and test: {split_report['overlap_count']}
- Exact record overlap excluding Student_ID: {exact_record_overlap}
- Shared feature-vector count: {feature_vector_overlap}
- Shared ID preview: {split_report['shared_student_ids']}
- Interpretation: the repository preserves a distinct held-out test set; there is no direct record overlap between the files.

## 3. Data quality checks
- Training validation: {train_validation}
- Test validation: {test_validation}
- No missing values were found in the current CSVs, and there are no duplicate identifiers. That said, the dataset can still be synthetic or rule-generated even when it is perfectly clean.

## 4. Likely target-generation behavior
The project notebook is explicit that the target variable is likely generated from deterministic rules and that the Random Forest model reaches near-perfect validation metrics. This is not evidence of a real-world placement process; it instead suggests label construction from a formula based on academic and skill features.

The strongest separation in the training data is visible in the following features:
{chr(10).join(strongest_features)}

Observations:
- CGPA is substantially higher among placed students.
- Backlogs are lower among placed students.
- Coding, communication, aptitude, project, and internship counts all trend upward for placed students.
- The model can separate the classes almost perfectly because the label appears to be strongly tied to these attributes.

## 5. Leakage and feature-risk assessment
- Student_ID is a unique identifier and not a legitimate predictive feature; it should be removed before modeling.
- The dataset does not show obvious leaked target columns, but the label-generation pattern is highly structured and may encode a scoring formula indirectly in the observed feature values.
- Because the target was likely generated from deterministic rules, the resulting accuracy numbers are not trustworthy as real-world deployment metrics.

## 6. Saved model artifact review
The current artifact is a scikit-learn pipeline with a `{estimator_name}` classifier and {len(artifact_features)} input features: {artifact_features}. It uses the current scikit-learn {sklearn.__version__} runtime. Rebuild the artifact after dependency changes because joblib pickle compatibility is version-sensitive. Its performance remains limited by the unverified/synthetic target provenance.

## 7. Project readiness assessment
The repository now contains a modular training, preprocessing, prediction, evaluation, validation, explainability, Streamlit, audit, and test implementation. The legacy notebook is isolated to `train.csv` and saves separate demo artifacts. The canonical workflow preserves the original test split and emits current JSON/CSV evaluation outputs.

## 8. Recommended actions
1. Obtain and document a verified real-world dataset before making placement prediction claims.
2. Keep the original train/test CSVs untouched and reserve the test set for the single final evaluation.
3. Consider cross-validation and hyperparameter search on training-only folds only after data provenance is resolved.
4. Treat current scores as pipeline-development results, not as evidence of real-world placement prediction.
5. Use the app only as a demonstration rather than as a real hiring or selection decision tool.
"""
    return report


def main() -> None:
    report_text = build_audit_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "dataset_audit.md"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Audit report written to {output_path}")


if __name__ == "__main__":
    main()
