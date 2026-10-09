import json

import pytest

from src.evaluate import compute_metrics
from src.schema import REPORTS_DIR


def _metrics_from_confusion_matrix(matrix):
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    total = true_negative + false_positive + false_negative + true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    return {
        "accuracy": (true_negative + true_positive) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / f1_denominator if f1_denominator else 0.0,
    }


def test_compute_metrics_uses_same_predictions_for_scores_and_confusion_matrix():
    actual = [0, 0, 1, 1, 1]
    predicted = [0, 1, 1, 0, 1]
    positive_scores = [0.1, 0.8, 0.7, 0.4, 0.9]

    metrics = compute_metrics(actual, predicted, positive_scores)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(2 / 3)


def test_saved_report_metrics_match_their_confusion_matrices():
    report_path = REPORTS_DIR / "model_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for item in report["comparison"]:
        metrics = item["validation_metrics"]
        expected = _metrics_from_confusion_matrix(metrics["confusion_matrix"])
        for name, value in expected.items():
            assert metrics[name] == pytest.approx(value), item["model"]

    test_metrics = report["selected_model"]["test_metrics"]
    expected = _metrics_from_confusion_matrix(test_metrics["confusion_matrix"])
    for name, value in expected.items():
        assert test_metrics[name] == pytest.approx(value)
