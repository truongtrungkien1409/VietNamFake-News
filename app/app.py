import os
import joblib
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from src.preprocessing.preprocess import TextPreprocessor

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "results", "saved_models")
PHOBERT_NAME = "vinai/phobert-base-v2"

# Khởi tạo thiết bị và mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Khởi động Flask API trên thiết bị: {device}")

print("🤖 Đang tải PhoBERT Tokenizer & Model...")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
bert_model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
bert_model.eval()

svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

svm_model = joblib.load(svm_path) if os.path.exists(svm_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
preprocessor = TextPreprocessor(use_word_segmentation=True)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": svm_model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Thiếu trường 'text' trong request JSON"}), 400

    raw_text = data["text"]
    if len(raw_text.strip()) < 20:
        return jsonify({"error": "Nội dung quá ngắn, cần ít nhất 20 ký tự"}), 400

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
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**encoded)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # 3. Chuẩn hóa bằng StandardScaler
        if scaler is not None:
            feat_scaled = scaler.transform(cls_embedding)
        else:
            feat_scaled = cls_embedding

        # 4. Dự đoán bằng SVM
        if svm_model is not None:
            probs = svm_model.predict_proba(feat_scaled)[0]
            fake_prob = float(probs[1]) # Xác suất lớp 1 (tin giả)
            is_fake = bool(fake_prob > 0.5)
        else:
            fake_prob = 0.5
            is_fake = False

        label_str = "Tin giả (Fake News)" if is_fake else "Tin thật (Real News)"

        return jsonify({
            "text_analyzed": raw_text[:100] + "...",
            "is_fake": is_fake,
            "prediction": label_str,
            "confidence_score": round(fake_prob if is_fake else (1.0 - fake_prob), 4),
            "fake_probability": round(fake_prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
