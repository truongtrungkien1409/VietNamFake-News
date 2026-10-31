from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import io
import sys
from functools import lru_cache
from typing import Optional
from app_logic import FakeNewsSystem


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read env var and fallback to simple .env parser if not set in OS env."""
    value = os.getenv(key)
    if value:
        return value

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return default

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        return default

    return default

def _safe_console_stream():
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
            logging.FileHandler('api.log', encoding='utf-8'),
            logging.StreamHandler(_safe_console_stream())
        ]
    )
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOBERT_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "phobert_fake_news")
SVM_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "svm_model.pkl")
SCALER_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "scaler.pkl")

# Khởi tạo hệ thống
detector = FakeNewsSystem(PHOBERT_MODEL, SVM_MODEL, SCALER_MODEL)
logger.info("FakeNewsSystem initialized successfully")

class NewsRequest(BaseModel):
    text: str
    source_url: Optional[str] = None

# Caching decorator - cache 100 kết quả gần nhất
@lru_cache(maxsize=100)
def _cached_predict(text: str, source_url: str = ""):
    """Internal function để cache kết quả"""
    return detector.predict(text, source_url=source_url or None)

@app.get("/")
def read_root():
    return {"status": "API is running", "version": "1.0"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict")
def predict_news(request: NewsRequest):
    """
    Phân tích tin tức - Phát hiện tin giả
    Input: text (nội dung tin cần kiểm tra, tối thiểu 20 ký tự)
    Output: JSON với risk_score, is_fake, reasons, components
    """
    if not request.text or len(request.text) < 20:
        logger.warning(f"Invalid input: text too short ({len(request.text)} chars)")
        raise HTTPException(status_code=400, detail="Nội dung phải có ít nhất 20 ký tự để phân tích")
    
    try:
        logger.info(f"Analyzing text: {request.text[:50]}...")
        source_url = request.source_url or ""
        result = _cached_predict(request.text, source_url)
        logger.info(f"Analysis completed: risk={result['final_risk_score']:.4f}, is_fake={result['is_fake']}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)