import os
import tempfile
import pandas as pd
import pytest
from src.config import get_paths
from src.preprocessing.split_data import split_dataset

def test_config_and_isolation(tmp_path):
    real_splits = "data/splits"
    real_results = "results"
    
    real_splits_existed = os.path.exists(real_splits)
    real_results_existed = os.path.exists(real_results)

    temp_results = tmp_path / "results_temp"
    temp_splits = tmp_path / "splits_temp"
    temp_processed = tmp_path / "processed_temp"

    os.environ["RESULTS_BASE"] = str(temp_results)
    os.environ["SPLITS_BASE"] = str(temp_splits)
    os.environ["PROCESSED_BASE"] = str(temp_processed)

    paths = get_paths()
    assert paths.results_base == str(temp_results)
    assert paths.splits == str(temp_splits)
    assert paths.processed == str(temp_processed)

    dummy_csv = tmp_path / "dummy_raw.csv"
    texts = [f"Bản tin giả lập số {i} với nội dung kiểm tra hệ thống phân chia dữ liệu." for i in range(50)]
    labels = [i % 2 for i in range(50)]
    pd.DataFrame({"text": texts, "label": labels}).to_csv(dummy_csv, index=False, encoding="utf-8")

    split_dataset(str(dummy_csv))

    assert os.path.exists(os.path.join(temp_splits, "train.csv"))
    assert os.path.exists(os.path.join(temp_processed, "clean.csv"))

    if not real_splits_existed:
        assert not os.path.exists(real_splits)
    if not real_results_existed:
        assert not os.path.exists(real_results)

    os.environ.pop("RESULTS_BASE", None)
    os.environ.pop("SPLITS_BASE", None)
    os.environ.pop("PROCESSED_BASE", None)
