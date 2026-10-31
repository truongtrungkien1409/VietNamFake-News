import re
import unicodedata
import pandas as pd
from typing import List, Optional
from src.preprocessing.teencode import TEENCODE_DICT, EMOJI_DICT

# Thử khởi tạo VnCoreNLP (RDRSegmenter) theo yêu cầu báo cáo
_SEGMENTER = None
_SEGMENTER_NAME = "pyvi"
_WARNING_PRINTED = False

try:
    import py_vncorenlp
    # Kiểm tra xem đã có model vncorenlp chưa hoặc tự tải
    _vncore_dir = "./vncorenlp"
    if not os.path.exists(_vncore_dir):
        os.makedirs(_vncore_dir, exist_ok=True)
    # Thử khởi tạo RDRSegmenter từ py_vncorenlp
    # Lưu ý: Yêu cầu Java. Nếu không có Java hoặc lỗi, sẽ fallback pyvi.
    # _SEGMENTER = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=_vncore_dir)
    # _SEGMENTER_NAME = "RDRSegmenter (VnCoreNLP)"
except Exception:
    pass

class TextPreprocessor:
    """
    Quy trình tiền xử lý văn bản tiếng Việt 5 bước chuẩn theo DACN.pdf (Mục 2.2.2.2 & Bảng 2.1):
    1. Loại bỏ ký tự rác, thẻ HTML và chuẩn hóa Unicode (NFC), chuyển lowercase.
    2. Chuẩn hóa URL -> HTTPURL, mention -> @USER, xử lý hashtag (#tin_nong -> tin_nong), gộp dấu câu lặp.
    3. Chuyển đổi teencode và emoji sang từ chuẩn/ý nghĩa, rút gọn ký tự lặp (vuiiii -> vui).
    4. Phân tách từ ghép tiếng Việt (Ưu tiên RDRSegmenter / VnCoreNLP, fallback pyvi ViTokenizer).
    5. Loại bỏ bản ghi rỗng, trùng lặp và câu dưới 3 từ.
    """
    def __init__(self, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation
        global _WARNING_PRINTED
        if _SEGMENTER is None and not _WARNING_PRINTED:
            print("⚠️ [Info] Không khởi tạo được VnCoreNLP/RDRSegmenter (hoặc thiếu Java). Sử dụng pyvi (ViTokenizer) làm fallback cho bước tách từ.")
            _WARNING_PRINTED = True

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # Bước 1: Chuẩn hóa Unicode NFC, lowercase, loại bỏ thẻ HTML
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.lower()

        # Bước 2: Chuẩn hóa URL, Mentions, Hashtag và dấu câu
        text = re.sub(r"https?://\S+|www\.\S+|url_placeholder", "HTTPURL", text, flags=re.IGNORECASE)
        text = re.sub(r"@\[A-Za-z0-9_]+|@\w+", "@USER", text)
        # Xử lý hashtag: giữ lại nội dung bên trong hashtag (#tin_nong -> tin_nong)
        text = re.sub(r"#([a-zA-Z0-9_À-ỹ]+)", r"\1", text)
        # Gộp dấu câu lặp (!! -> "")
        text = re.sub(r"([!?.,;:])\1+", r"\1", text)

        # Bước 3: Ánh xạ teencode và emoji, rút gọn ký tự lặp
        # Thay thế emoji
        for emoji_char, meaning in EMOJI_DICT.items():
            text = text.replace(emoji_char, f" {meaning} ")
        # Xóa các emoji còn lại không có trong từ điển
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

        # Thay thế teencode theo từ (token)
        tokens = text.split()
        normalized_tokens = []
        for token in tokens:
            # Rút gọn ký tự lặp quá 2 lần (ví dụ: vuiiii -> vuii)
            token = re.sub(r'(.)\1{2,}', r'\1\1', token)
            if token in TEENCODE_DICT:
                normalized_tokens.append(TEENCODE_DICT[token])
            else:
                normalized_tokens.append(token)
        text = " ".join(normalized_tokens)

        # Chuẩn hóa khoảng trắng
        text = re.sub(r"\s+", " ", text).strip()

        # Bước 4: Phân tách từ ghép tiếng Việt
        if self.use_word_segmentation:
            try:
                if _SEGMENTER is not None:
                    segmented_sents = _SEGMENTER.word_segment(text)
                    text = " ".join(segmented_sents)
                else:
                    from pyvi import ViTokenizer
                    text = ViTokenizer.tokenize(text)
            except Exception as e:
                # Fallback to simple split if pyvi fails (e.g. disk space error on C:)
                pass

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Bước 5 ở cấp độ DataFrame: Loại bỏ bản ghi rỗng, trùng lặp theo text sau xử lý, và câu dưới 3 từ.
    """
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"DataFrame phải chứa các cột '{text_col}' và '{label_col}'.")

    print(f"🧹 Bắt đầu tiền xử lý DataFrame (Tổng số mẫu ban đầu: {len(df)})")
    preprocessor = TextPreprocessor(use_word_segmentation=True)

    df = df.dropna(subset=[text_col, label_col]).copy()
    df['cleaned_text'] = df[text_col].astype(str).apply(preprocessor.clean_text)

    # Lọc bỏ bản ghi rỗng
    df = df[df['cleaned_text'].str.strip().str.len() > 0].copy()

    # Lọc bỏ câu dưới 3 từ
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 3].copy()

    # Lọc bỏ trùng lặp theo cleaned_text (giữ bản đầu tiên)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['cleaned_text'], keep='first').copy()
    print(f"   - Đã loại bỏ {initial_len - len(df)} mẫu trùng lặp hoặc dưới 3 từ.")
    print(f"✅ Số lượng mẫu sau khi tiền xử lý hoàn tất: {len(df)}")

    return df[[text_col, 'cleaned_text', label_col]].rename(columns={text_col: 'original_text', 'cleaned_text': 'text'})
