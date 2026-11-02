import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics

def test_compute_metrics():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1]
    y_score = [0.1, 0.2, 0.9, 0.4, 0.3, 0.8]

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert "accuracy" in metrics
    assert "fake_precision" in metrics
    assert "fake_recall" in metrics
    assert "fake_f1" in metrics
    assert "auc_roc" in metrics
    assert "TP" in metrics
    assert "TN" in metrics
    assert "FP" in metrics
    assert "FN" in metrics

    # Kiểm tra tính đúng đắn thủ công cơ bản
    # y_true: [0,0,1,1,0,1] (3 thực tế 0, 3 thực tế 1)
    # y_pred: [0,0,1,0,0,1] (TP=2, FN=1, TN=3, FP=0)
    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["TN"] == 3
    assert metrics["FP"] == 0
    assert metrics["accuracy"] == 5 / 6

def test_compute_metrics_class_support_and_confusion_matrix():
    y_true = [0] * 6 + [1] * 4
    y_pred = [0] * 5 + [1] + [1] * 3 + [0]
    y_score = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1]

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert metrics["real_support"] == 6
    assert metrics["fake_support"] == 4
    assert metrics["fake_precision"] == pytest.approx(3 / 4)
    assert metrics["fake_recall"] == pytest.approx(3 / 4)
    assert metrics["fake_f1"] == pytest.approx(0.75)
    assert metrics["TN"] == 5
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TP"] == 3
