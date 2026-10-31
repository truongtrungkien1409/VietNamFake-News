import joblib
import torch
import numpy as np
import os
import logging
import io
import sys
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from core.preprocess import clean_and_preprocess_text

def _safe_console_stream():
    """Create a UTF-8 safe console stream to avoid cp1258 UnicodeEncodeError on Windows."""
    stream = sys.stdout
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
        return stream
    except Exception:
        pass

    if hasattr(stream, "buffer"):
        return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")

    return stream


# Cấu hình logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fake_news_system.log', encoding='utf-8'),
            logging.StreamHandler(_safe_console_stream())
        ]
    )

class FakeNewsSystem:
    def __init__(self, phobert_path, svm_model_path, scaler_path=None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing FakeNewsSystem (PhoBERT-base-v2 [CLS] + StandardScaler + SVM)...")
        
        try:
            # 1. Khởi tạo PhoBERT (Feature Extractor)
            self.logger.info("Loading PhoBERT base model & tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(phobert_path)
                self.bert_model = AutoModel.from_pretrained(phobert_path)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                self.bert_model = AutoModel.from_pretrained("vinai/phobert-base")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device).eval()
            self.logger.info(f"PhoBERT loaded on device: {self.device}")

            # 2. Khởi tạo mô hình SVM phân loại và StandardScaler
            self.logger.info("Loading SVM model and StandardScaler...")
            if os.path.exists(svm_model_path):
                self.svm_model = joblib.load(svm_model_path)
                self.logger.info("SVM model loaded successfully")
            else:
                self.svm_model = None
                self.logger.warning(f"SVM model not found at {svm_model_path}.")

            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.logger.info("StandardScaler loaded successfully")
            else:
                self.scaler = None
                self.logger.warning(f"StandardScaler not found at {scaler_path}.")

            self.logger.info("System initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}", exc_info=True)
            raise

    def predict(self, text, source_url=None):
        try:
            self.logger.info(f"Starting prediction for text: {str(text)[:50]}...")
            start_time = datetime.now()
            
            # --- BƯỚC 1: TIỀN XỬ LÝ 5 BƯỚC CHUYÊN BIỆT ---
            processed_text = clean_and_preprocess_text(text)
            self.logger.debug(f"Preprocessed text: {processed_text[:100]}")

            # --- BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG 768 CHIỀU TỪ [CLS] TOKEN ---
            inputs = self.tokenizer(
                processed_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                cls_embedding = last_hidden_state[:, 0, :].cpu().numpy() # Kích thước: (1, 768)

            # --- BƯỚC 3: CHUẨN HÓA DỮ LIỆU BẰNG STANDARDSCALER ---
            if self.scaler is not None:
                feature_scaled = self.scaler.transform(cls_embedding)
            else:
                feature_scaled = cls_embedding

            # --- BƯỚC 4: DỰ ĐOÁN BẰNG SVM ---
            if self.svm_model is not None:
                probs = self.svm_model.predict_proba(feature_scaled)[0]
                svm_prob_fake = float(probs[1]) # Xác suất là tin giả (class 1)
            else:
                svm_prob_fake = 0.5
                self.logger.warning("SVM model is not loaded. Returning default probability 0.5.")

            final_risk = svm_prob_fake

            # --- BƯỚC 5: XÂY DỰNG GIẢI THÍCH (REASONING PANEL) ---
            reasons = []

            if svm_prob_fake > 0.7:
                reasons.append(f"🤖 Mô hình PhoBERT-SVM đánh giá nội dung này mang đặc trưng rõ rệt của tin giả (Độ tin cậy: {svm_prob_fake*100:.1f}%)")
            elif svm_prob_fake < 0.3:
                reasons.append(f"🤖 Mô hình PhoBERT-SVM xác định nội dung phù hợp với tin tức chính thống (Độ tin cậy: {(1-svm_prob_fake)*100:.1f}%)")
            else:
                reasons.append("⚠️ Nội dung nằm trong vùng ranh giới phân tách (Grey area), cần kiểm chứng thêm từ các cơ quan báo chí.")

            # --- BƯỚC 6: TRẢ KẾT QUẢ ---
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Prediction completed: risk={final_risk:.4f}, time={elapsed_time:.2f}s")
            
            return {
                "final_risk_score": float(final_risk),
                "is_fake": bool(final_risk > 0.5),
                "reasons": list(reasons),
                "components": {
                    "svm_phobert_risk": float(svm_prob_fake),
                    "confidence_percentage": round(final_risk * 100, 2),
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": round(elapsed_time, 3)
                }
            }
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise
