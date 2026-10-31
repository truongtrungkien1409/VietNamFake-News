import os
import re
import string
import unicodedata
import pandas as pd
from typing import List, Optional
from src.preprocessing.teencode import TEENCODE_DICT, EMOJI_DICT

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

class TextPreprocessor:
    """
    Quy trình tiền xử lý văn bản tiếng Việt 5 bước chuẩn theo DACN.pdf (Mục 2.2.2.2 & Bảng 2.1):
    1. Loại bỏ ký tự rác, thẻ HTML và chuẩn hóa Unicode (NFC), chuyển lowercase.
    2. Chuẩn hóa URL -> HTTPURL, mention -> @USER, xử lý hashtag (#tin_nong -> tin_nong), xóa dấu câu lặp.
    3. Chuyển đổi teencode và emoji (xử lý cả variation selectors và các emoji khác) sang từ chuẩn, rút gọn ký tự lặp trên chữ cái.
    4. Phân tách từ ghép tiếng Việt bằng pyvi (ViTokenizer).
    5. Loại bỏ bản ghi rỗng, trùng lặp và câu dưới 3 từ.
    """
    def __init__(self, use_word_segmentation: bool = True):
        self.use_word_segmentation = use_word_segmentation
        self.segmentation_error_count = 0
        if use_word_segmentation and not HAS_PYVI:
            raise ImportError("Không tìm thấy thư viện pyvi. Yêu cầu cài đặt pyvi để thực hiện tách từ tiếng Việt.")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # Bước 1: Chuẩn hóa Unicode NFC, lowercase, loại bỏ thẻ HTML
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.lower()

        # Bước 2: Chuẩn hóa URL, Mentions, Hashtag và xóa dấu câu lặp (!!, ??...)
        text = re.sub(r"https?://\S+|www\.\S+|url_placeholder", "HTTPURL", text, flags=re.IGNORECASE)
        text = re.sub(r"@\[A-Za-z0-9_]+|@\w+", "@USER", text)
        # Xử lý hashtag: giữ lại nội dung bên trong hashtag (#tin_nong -> tin_nong)
        text = re.sub(r"#([a-zA-Z0-9_À-ỹ]+)", r"\1", text)
        # Xóa chuỗi dấu câu lặp (!!, ??, ...) hoặc gộp theo Bảng 2.1
        text = re.sub(r"([!?.,;:])\1+", r"\1", text)

        # Bước 3: Xử lý Emoji (bao gồm variation selectors U+FE0F và mọi emoji BMP/non-BMP) và Teencode
        for emoji_char, meaning in EMOJI_DICT.items():
            text = text.replace(emoji_char, f" {meaning} ")
        # Xóa variation selectors và các emoji còn lại
        text = re.sub(r"[\U00010000-\U0010ffff\uFE0F]", "", text)
        # Xóa các ký tự emoji Unicode bổ sung
        text = re.sub(r"[^\w\s,.;:!?/@-_]", "", text)

        # Rút gọn ký tự lặp CHỈ trên chữ cái, không đụng chữ số (vuiiii -> vui)
        tokens = text.split()
        normalized_tokens = []
        for token in tokens:
            token = re.sub(r"([^\W\d_])\1{2,}", r"\1", token)
            if token in TEENCODE_DICT:
                normalized_tokens.append(TEENCODE_DICT[token])
            else:
                normalized_tokens.append(token)
        text = " ".join(normalized_tokens)

        # Chuẩn hóa khoảng trắng
        text = re.sub(r"\s+", " ", text).strip()

        # Bước 4: Phân tách từ ghép tiếng Việt bằng pyvi
        if self.use_word_segmentation:
            try:
                text = ViTokenizer.tokenize(text)
            except Exception:
                self.segmentation_error_count += 1

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]

def _get_normalization_key(text: str) -> str:
    """Chuẩn hóa thêm để loại bỏ dấu câu và khoảng trắng thừa khi kiểm tra trùng lặp."""
    # Bỏ toàn bộ dấu câu
    no_punct = text.translate(str.maketrans('', '', string.punctuation))
    # Gộp khoảng trắng
    return re.sub(r"\s+", " ", no_punct).strip()

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Bước 5 ở cấp độ DataFrame: Loại bỏ bản ghi rỗng, trùng lặp, câu dưới 3 từ,
    và loại bỏ hoàn toàn các văn bản xuất hiện với hơn một nhãn khác nhau (nhãn xung đột).
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

    # Tạo khóa chuẩn hóa (bỏ toàn bộ dấu câu và khoảng trắng thừa)
    df['norm_key'] = df['cleaned_text'].apply(_get_normalization_key)

    # Loại bỏ các văn bản xuất hiện với hơn một nhãn khác nhau (nhãn xung đột)
    label_counts_per_key = df.groupby('norm_key')[label_col].nunique()
    conflicting_keys = label_counts_per_key[label_counts_per_key > 1].index
    n_conflicting = df[df['norm_key'].isin(conflicting_keys)].shape[0]
    
    if n_conflicting > 0:
        print(f"   - Đã loại bỏ {n_conflicting} mẫu do có nhãn xung đột (cùng nội dung nhưng khác nhãn).")
    df = df[~df['norm_key'].isin(conflicting_keys)].copy()

    # Bỏ trùng lặp theo norm_key (giữ bản đầu tiên)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['norm_key'], keep='first').copy()
    print(f"   - Đã loại bỏ {initial_len - len(df)} mẫu trùng lặp.")
    print(f"✅ Số lượng mẫu sau khi tiền xử lý hoàn tất: {len(df)}")

    return df[[text_col, 'cleaned_text', label_col]].rename(columns={text_col: 'original_text', 'cleaned_text': 'text'})
