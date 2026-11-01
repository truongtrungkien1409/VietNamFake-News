import pytest
import numpy as np
from app.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check_no_model(monkeypatch):
    import app.app as flask_app
    monkeypatch.setattr(flask_app, "svm_model", None)
    monkeypatch.setattr(flask_app, "scaler", None)
    
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False

def test_health_check_with_model(monkeypatch):
    import app.app as flask_app
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])
    class MockScaler:
        def transform(self, X):
            return X
    class MockBert:
        pass
    class MockTokenizer:
        pass

    monkeypatch.setattr(flask_app, "svm_model", MockModel())
    monkeypatch.setattr(flask_app, "scaler", MockScaler())
    monkeypatch.setattr(flask_app, "bert_model", MockBert())
    monkeypatch.setattr(flask_app, "tokenizer", MockTokenizer())

    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["model_loaded"] is True

def test_predict_missing_model(monkeypatch):
    import app.app as flask_app
    monkeypatch.setattr(flask_app, "svm_model", None)
    monkeypatch.setattr(flask_app, "scaler", None)
    monkeypatch.setattr(flask_app, "bert_model", None)
    monkeypatch.setattr(flask_app, "tokenizer", None)

    client = app.test_client()
    res = client.post("/predict", json={"text": "Đây là một đoạn văn bản kiểm tra đủ trên 20 ký tự."})
    assert res.status_code == 503
    data = res.get_json()
    assert "error" in data

def test_predict_missing_text(monkeypatch):
    import app.app as flask_app
    class MockModel: pass
    monkeypatch.setattr(flask_app, "svm_model", MockModel())
    monkeypatch.setattr(flask_app, "scaler", MockModel())
    monkeypatch.setattr(flask_app, "bert_model", MockModel())
    monkeypatch.setattr(flask_app, "tokenizer", MockModel())

    client = app.test_client()
    res = client.post("/predict", json={})
    assert res.status_code == 400

def test_predict_too_short(monkeypatch):
    import app.app as flask_app
    class MockModel: pass
    monkeypatch.setattr(flask_app, "svm_model", MockModel())
    monkeypatch.setattr(flask_app, "scaler", MockModel())
    monkeypatch.setattr(flask_app, "bert_model", MockModel())
    monkeypatch.setattr(flask_app, "tokenizer", MockModel())

    client = app.test_client()
    res = client.post("/predict", json={"text": "Ngắn"})
    assert res.status_code == 400

def test_predict_valid(monkeypatch):
    import app.app as flask_app
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])
    class MockScaler:
        def transform(self, X):
            return X
    class MockOutputs:
        last_hidden_state = np.zeros((1, 10, 768))
    class MockBert:
        def __call__(self, *args, **kwargs):
            return MockOutputs()
    class MockTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": None}

    monkeypatch.setattr(flask_app, "svm_model", MockModel())
    monkeypatch.setattr(flask_app, "scaler", MockScaler())
    monkeypatch.setattr(flask_app, "bert_model", MockBert())
    monkeypatch.setattr(flask_app, "tokenizer", MockTokenizer())

    client = app.test_client()
    res = client.post("/predict", json={"text": "Đây là nội dung bài báo cáo kiểm tra tin giả hợp lệ với hơn hai mươi ký tự."})
    assert res.status_code == 200
    data = res.get_json()
    assert data["is_fake"] is True
    assert data["fake_probability"] == 0.9

def test_predict_batch_valid(monkeypatch):
    import app.app as flask_app
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])
    class MockScaler:
        def transform(self, X):
            return X
    class MockOutputs:
        last_hidden_state = np.zeros((2, 10, 768))
    class MockBert:
        def __call__(self, *args, **kwargs):
            return MockOutputs()
    class MockTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": None}

    monkeypatch.setattr(flask_app, "svm_model", MockModel())
    monkeypatch.setattr(flask_app, "scaler", MockScaler())
    monkeypatch.setattr(flask_app, "bert_model", MockBert())
    monkeypatch.setattr(flask_app, "tokenizer", MockTokenizer())

    client = app.test_client()
    res = client.post("/predict_batch", json={
        "texts": [
            "Đoạn văn bản số một dùng để kiểm tra tính năng dự đoán theo lô batch.",
            "Đoạn văn bản số hai dùng để kiểm tra tính năng dự đoán theo lô batch."
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
