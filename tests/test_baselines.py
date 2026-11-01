import os
import pandas as pd
import pytest
from src.models.baselines.naive_bayes import train_and_evaluate_nb
from src.models.baselines.svm_tfidf import train_and_evaluate_svm_tfidf
from src.models.baselines.lstm import train_and_evaluate_lstm
from src.models.baselines.cnn import train_and_evaluate_cnn

def test_baselines_synthetic(tmp_path):
    # Tạo file csv giả lập nhỏ cho train, val, test
    df = pd.DataFrame({
        "original_text": ["Tin tức giả mạo một hai ba", "Tin tức chính thống uy tín bốn năm", "Bài báo kiểm tra sáu bảy", "Thông tin bịa đặt tám chín mười"],
        "text": ["tin tức giả mạo một hai ba", "tin tức chính thống uy tín bốn năm", "bài báo kiểm tra sáu bảy", "thông tin bịa đặt tám chín mười"],
        "label": [1, 0, 0, 1]
    })

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    df.to_csv(train_csv, index=False)
    df.to_csv(val_csv, index=False)
    df.to_csv(test_csv, index=False)

    out_nb = tmp_path / "nb.csv"
    pred_nb = tmp_path / "pred_nb.csv"
    cm_nb = tmp_path / "cm_nb.png"

    train_and_evaluate_nb(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        output_csv=str(out_nb),
        pred_csv=str(pred_nb),
        cm_path=str(cm_nb),
        max_samples=4
    )
    assert os.path.exists(out_nb)

    # Test LSTM 1 epoch trên dữ liệu giả lập
    out_lstm = tmp_path / "lstm.csv"
    pred_lstm = tmp_path / "pred_lstm.csv"
    cm_lstm = tmp_path / "cm_lstm.png"
    model_lstm = tmp_path / "lstm.pt"

    train_and_evaluate_lstm(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        output_csv=str(out_lstm),
        pred_csv=str(pred_lstm),
        cm_path=str(cm_lstm),
        model_path=str(model_lstm),
        epochs=1,
        max_samples=4
    )
    assert os.path.exists(out_lstm)
