import os
import numpy as np
import pytest
from src.models.train_svm import train_and_tune_svm

def test_train_svm_synthetic(tmp_path):
    # Sinh dữ liệu giả lập nhỏ bằng numpy (không lưu làm dữ liệu dự án)
    n_train = 40
    n_val = 20
    n_features = 768

    np.random.seed(42)
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.choice([0, 1], size=n_train)
    
    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.choice([0, 1], size=n_val)

    train_feat = tmp_path / "train_feat.npy"
    train_lab = tmp_path / "train_lab.npy"
    val_feat = tmp_path / "val_feat.npy"
    val_lab = tmp_path / "val_lab.npy"

    np.save(train_feat, X_train)
    np.save(train_lab, y_train)
    np.save(val_feat, X_val)
    np.save(val_lab, y_val)

    model_out = tmp_path / "saved_models" / "svm_model.pkl"
    scaler_out = tmp_path / "saved_models" / "scaler.pkl"
    tuning_table = tmp_path / "tables" / "svm_tuning.csv"
    final_config = tmp_path / "tables" / "svm_final_config.json"
    heatmap_out = tmp_path / "figures" / "svm_tuning_heatmap.png"

    # Gọi hàm train_and_tune_svm với đường dẫn giả lập
    train_and_tune_svm(
        train_feat_path=str(train_feat),
        train_label_path=str(train_lab),
        val_feat_path=str(val_feat),
        val_label_path=str(val_lab),
        model_output_path=str(model_out),
        scaler_output_path=str(scaler_out),
        tuning_table_path=str(tuning_table),
        final_config_path=str(final_config),
        heatmap_path=str(heatmap_out),
        use_best=False,
        class_weight_arg=None
    )

    # Kiểm tra các file đầu ra được tạo ra thành công
    assert os.path.exists(model_out)
    assert os.path.exists(scaler_out)
    assert os.path.exists(tuning_table)
    assert os.path.exists(final_config)
    assert os.path.exists(heatmap_out)
