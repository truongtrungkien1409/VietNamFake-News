import os
import time
import logging
import joblib
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from src.config import get_paths
from src.preprocessing.preprocess import TextPreprocessor

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [r"^chrome-extension://.*$", r"^http://localhost(:\d+)?$", r"^http://127\.0\.0\.1(:\d+)?$"]}})

paths = get_paths()
MODEL_DIR = paths.saved_models
PHOBERT_NAME = "vinai/phobert-base-v2"

# Khởi tạo thiết bị và mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Khởi động Flask API trên thiết bị: {device}")

tokenizer = None
bert_model = None
svm_model = None
scaler = None
preprocessor = TextPreprocessor(use_word_segmentation=True)

try:
    logger.info("🤖 Đang tải PhoBERT Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
    bert_model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
    bert_model.eval()
except Exception as e:
    logger.warning(f"⚠️ Không thể tải mô hình PhoBERT từ Hub: {e}")

svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

if os.path.exists(svm_path):
    svm_model = joblib.load(svm_path)
    logger.info(f"✅ Đã tải SVM model từ {svm_path}")
else:
    logger.warning(f"⚠️ Không tìm thấy file SVM model tại {svm_path}")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    logger.info(f"✅ Đã tải Scaler từ {scaler_path}")
else:
    logger.warning(f"⚠️ Không tìm thấy file Scaler tại {scaler_path}")

MODEL_VERSION = "1.0.0-phobert-svm"

@app.route("/health", methods=["GET"])
def health_check():
    model_loaded = (svm_model is not None and scaler is not None and bert_model is not None)
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "model_version": MODEL_VERSION
    })

@app.route("/predict", methods=["POST"])
def predict():
    if svm_model is None or scaler is None or bert_model is None or tokenizer is None:
        return jsonify({
            "error": "Mô hình chưa được huấn luyện hoặc thiếu tệp trọng số (svm_model.pkl/scaler.pkl). Hãy huấn luyện mô hình trước."
        }), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Thiếu trường 'text' trong request JSON"}), 400

    raw_text = str(data["text"])
    text_len = len(raw_text.strip())
    
    if text_len < 20 or text_len > 20000:
        return jsonify({"error": "Nội dung không hợp lệ: độ dài phải từ 20 đến 20.000 ký tự."}), 400

    start_time = time.time()
    try:
        # 1. Tiền xử lý 5 bước
        processed = preprocessor.clean_text(raw_text)

        # 2. Trích xuất đặc trưng [CLS] 768 chiều từ PhoBERT
        encoded = tokenizer(
            processed,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        elif isinstance(encoded, dict):
            encoded = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}

        with torch.no_grad():
            outputs = bert_model(**encoded)
            hidden_state = outputs.last_hidden_state
            if hasattr(hidden_state, "cpu"):
                cls_embedding = hidden_state[:, 0, :].cpu().numpy()
            else:
                cls_embedding = hidden_state[:, 0, :]

        # 3. Chuẩn hóa bằng StandardScaler
        feat_scaled = scaler.transform(cls_embedding)

        # 4. Dự đoán bằng SVM
        probs = svm_model.predict_proba(feat_scaled)[0]
        fake_prob = float(probs[1]) # Xác suất lớp 1 (tin giả)
        is_fake = bool(fake_prob > 0.5)
        confidence = float(fake_prob if is_fake else (1.0 - fake_prob))

        label_str = "Tin giả (Fake News)" if is_fake else "Tin thật (Real News)"
        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        # Ghi log yêu cầu (không ghi nội dung đầy đủ, chỉ độ dài và kết quả)
        logger.info(f"API /predict | Length: {text_len} chars | IsFake: {is_fake} | Prob: {fake_prob:.4f} | Time: {processing_time_ms}ms")

        return jsonify({
            "prediction": label_str,
            "is_fake": is_fake,
            "fake_probability": round(fake_prob, 4),
            "confidence_score": round(confidence, 4),
            "processing_time_ms": processing_time_ms,
            "model_version": MODEL_VERSION
        })

    except Exception as e:
        logger.error(f"Lỗi suy luận /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if svm_model is None or scaler is None or bert_model is None or tokenizer is None:
        return jsonify({
            "error": "Mô hình chưa được huấn luyện hoặc thiếu tệp trọng số. Hãy huấn luyện mô hình trước."
        }), 503

    data = request.get_json()
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Thiếu danh sách 'texts' trong request JSON"}), 400

    texts = data["texts"]
    if len(texts) == 0 or len(texts) > 32:
        return jsonify({"error": "Batch phải chứa từ 1 đến tối đa 32 văn bản."}), 400

    start_time = time.time()
    results = []

    try:
        processed_texts = [preprocessor.clean_text(str(t)) for t in texts]

        encoded = tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        elif isinstance(encoded, dict):
            encoded = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}

        with torch.no_grad():
            outputs = bert_model(**encoded)
            hidden_state = outputs.last_hidden_state
            if hasattr(hidden_state, "cpu"):
                cls_embeddings = hidden_state[:, 0, :].cpu().numpy()
            else:
                cls_embeddings = hidden_state[:, 0, :]

        feat_scaled = scaler.transform(cls_embeddings)
        probs_array = svm_model.predict_proba(feat_scaled)

        for i, raw_text in enumerate(texts):
            fake_prob = float(probs_array[i][1])
            is_fake = bool(fake_prob > 0.5)
            confidence = float(fake_prob if is_fake else (1.0 - fake_prob))
            label_str = "Tin giả (Fake News)" if is_fake else "Tin thật (Real News)"

            results.append({
                "prediction": label_str,
                "is_fake": is_fake,
                "fake_probability": round(fake_prob, 4),
                "confidence_score": round(confidence, 4)
            })

        total_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"API /predict_batch | Count: {len(texts)} | Time: {total_time_ms}ms")

        return jsonify({
            "results": results,
            "batch_size": len(texts),
            "processing_time_ms": total_time_ms,
            "model_version": MODEL_VERSION
        })

    except Exception as e:
        logger.error(f"Lỗi suy luận /predict_batch: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
