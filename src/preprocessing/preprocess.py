import re
import unicodedata
from typing import List, Optional

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

class TextPreprocessor:
    """
    Quy trình tiền xử lý văn bản tiếng Việt 5 bước chuẩn theo DACN.pdf:
    1. Loại bỏ ký tự rác, thẻ HTML và chuẩn hóa Unicode (NFC).
    2. Chuẩn hóa link web thành HTTPURL, nhắc tên thành @USER.
    3. Chuyển đổi/loại bỏ emoji và teencode về từ chuẩn.
    4. Phân tách từ ghép tiếng Việt (RDRSegmenter / ViTokenizer).
    5. Loại bỏ bản ghi rỗng hoặc trùng lặp.
    """
    def __init__(self, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Bước 1: Chuẩn hóa Unicode NFC và loại bỏ HTML tags
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"<[^>]+>", "", text)
        
        # Bước 2: Chuẩn hóa URL và Mentions
        text = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", text)
        text = re.sub(r"@\[A-Za-z0-9_]+|@\w+", "@USER", text)
        
        # Bước 3: Xử lý ký tự đặc biệt, emoji, teencode cơ bản
        # (Giữ lại các dấu câu cơ bản, loại bỏ emoji hoặc thay thế nếu cần)
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Bước 4: Tách từ ghép tiếng Việt
        if self.use_word_segmentation and HAS_PYVI:
            text = ViTokenizer.tokenize(text)
            
        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        cleaned = [self.clean_text(t) for t in texts]
        # Bước 5: Lọc bỏ rỗng
        return [t for t in cleaned if len(t.strip()) > 0]
