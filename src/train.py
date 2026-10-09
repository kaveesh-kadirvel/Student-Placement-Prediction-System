from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .evaluate import compute_metrics
from .preprocessing import build_preprocessor
from .schema import BENCHMARK_PATH, MODEL_PATH, POSITIVE_LABEL, ROOT_DIR, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from .validation import check_split_integrity, encode_target, validate_dataframe


def build_model_catalog():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
        "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ),
    }


def train_and_compare(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    audit = validate_dataframe(train_df)
    if not audit["valid"]:
        raise ValueError(f"Training data invalid: {audit['issues']}")

    test_audit = validate_dataframe(test_df)
    if not test_audit["valid"]:
        raise ValueError(f"Test data invalid: {test_audit['issues']}")

    split_report = check_split_integrity(train_df, test_df)
    if split_report["overlap_count"]:
        raise ValueError(f"Train/test overlap exists in Student_ID values: {split_report['shared_student_ids']}")

    X_train_df = train_df.drop(columns=["Student_ID", TARGET_COLUMN])
    y_train_raw = train_df[TARGET_COLUMN]
    y_train = encode_target(y_train_raw)

    X_test_df = test_df.drop(columns=["Student_ID", TARGET_COLUMN])
    y_test_raw = test_df[TARGET_COLUMN]
    y_test = encode_target(y_test_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    comparison = []
    for model_name, estimator in build_model_catalog().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_tr, y_tr)

        val_pred = pipeline.predict(X_val)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_pred, val_proba)
        comparison.append(
            {
                "model": model_name,
                "validation_metrics": metrics,
                "pipeline": pipeline,
            }
        )

    best_entry = max(
        comparison,
        key=lambda entry: (
            entry["validation_metrics"]["f1"],
            entry["validation_metrics"].get("roc_auc", 0.0),
            entry["validation_metrics"]["accuracy"],
        ),
    )
    best_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clone(best_entry["pipeline"].named_steps["classifier"])),
        ]
    )
    best_model.fit(X_train_df, y_train)

    test_pred = best_model.predict(X_test_df)
    test_proba = best_model.predict_proba(X_test_df)[:, 1]
    final_test_metrics = compute_metrics(y_test, test_pred, test_proba)

    model_path = MODEL_PATH
    joblib.dump(best_model, model_path)

    benchmark = train_df[train_df[TARGET_COLUMN] == POSITIVE_LABEL].loc[:, [
        "CGPA",
        "Internships",
        "Projects",
        "Coding_Skills",
        "Communication_Skills",
        "Aptitude_Test_Score",
        "Soft_Skills_Rating",
        "Certifications",
        "Backlogs",
    ]].mean()
    joblib.dump(benchmark, BENCHMARK_PATH)

    report = {
        "interpretation": {
            "data_provenance": "Unverified; repository notebook and audit indicate synthetic or rule-generated labels.",
            "result_scope": "Development-pipeline results on the current repository dataset only; not evidence of real-world placement validity.",
            "positive_class": POSITIVE_LABEL,
            "selection_metric": "Validation F1, with validation ROC-AUC and accuracy as tie-breakers.",
            "test_set_policy": "The original test.csv is evaluated only after selection/refitting, but rerunning this script recomputes test metrics; do not use these results for tuning or selection.",
            "preprocessing_policy": "Each preprocessing pipeline is fitted only on the training partition of train.csv.",
        },
        "dataset": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_target_distribution": train_df[TARGET_COLUMN].value_counts().to_dict(),
            "test_target_distribution": test_df[TARGET_COLUMN].value_counts().to_dict(),
            "split_integrity": split_report,
        },
        "comparison": [
            {
                "model": entry["model"],
                "validation_metrics": entry["validation_metrics"],
            }
            for entry in comparison
        ],
        "selected_model": {
            "name": best_entry["model"],
            "test_metrics": final_test_metrics,
        },
    }

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_dir / "model_metrics.csv").write_text(
        "model,accuracy,precision,recall,f1,roc_auc\n" + "\n".join(
            f"{entry['model']},{entry['validation_metrics']['accuracy']},{entry['validation_metrics']['precision']},{entry['validation_metrics']['recall']},{entry['validation_metrics']['f1']},{entry['validation_metrics'].get('roc_auc', 0.0)}"
            for entry in comparison
        ) + "\n",
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    train_and_compare()
