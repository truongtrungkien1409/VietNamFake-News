import csv
from pathlib import Path

import pytest


def test_evaluation_metrics_has_real_dataset_support():
    metrics_path = Path("results/tables/evaluation_metrics.csv")
    if not metrics_path.exists():
        pytest.skip("evaluation_metrics.csv does not exist")

    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics = next(csv.DictReader(metrics_file))

    assert int(metrics["fake_support"]) + int(metrics["real_support"]) >= 100
