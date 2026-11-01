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
    Quy trình tiền xử lý văn bản tiếng Việt linh hoạt hỗ trợ các cờ cấu hình độc lập cho Nghiên cứu Ablation (Giả thuyết H2):
    - lowercase: chuyển chữ thường
    - clean_url_html: chuẩn hóa Unicode NFC, xóa HTML tags, chuẩn hóa URL và mentions (@USER)
    - hashtag_punct: xử lý hashtag và xóa/gộp dấu câu lặp
    - teencode_emoji: ánh xạ teencode và loại bỏ/ánh xạ emoji
    - repeat_chars: rút gọn ký tự lặp trên chữ cái
    - word_segmentation: phân tách từ ghép tiếng Việt (pyvi)
    """
    def __init__(
        self,
        lowercase: bool = True,
        clean_url_html: bool = True,
        hashtag_punct: bool = True,
        teencode_emoji: bool = True,
        repeat_chars: bool = True,
        word_segmentation: bool = True,
        use_word_segmentation: Optional[bool] = None
    ):
        if use_word_segmentation is not None:
            word_segmentation = use_word_segmentation
        self.lowercase = lowercase
        self.clean_url_html = clean_url_html
        self.hashtag_punct = hashtag_punct
        self.teencode_emoji = teencode_emoji
        self.repeat_chars = repeat_chars
        self.word_segmentation = word_segmentation
        self.segmentation_error_count = 0

        if word_segmentation and not HAS_PYVI:
            raise ImportError("Không tìm thấy thư viện pyvi. Yêu cầu cài đặt pyvi để thực hiện tách từ tiếng Việt.")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if not text.strip():
            return ""

        # Luôn chuẩn hóa NFC cơ bản
        text = unicodedata.normalize("NFC", text)

        # 1. clean_url_html: Xóa HTML tags
        if self.clean_url_html:
            text = re.sub(r"<[^>]+>", "", text)

        # 2. lowercase
        if self.lowercase:
            text = text.lower()

        # Chuẩn hóa URL và Mentions sau lowercase để giữ chữ hoa HTTPURL và @USER
        if self.clean_url_html:
            text = re.sub(r"https?://\S+|www\.\S+|url_placeholder", "HTTPURL", text, flags=re.IGNORECASE)
            text = re.sub(r"@\[A-Za-z0-9_]+|@\w+|@user", "@USER", text, flags=re.IGNORECASE)

        # 3. hashtag_punct: Xử lý hashtag và dấu câu lặp
        if self.hashtag_punct:
            text = re.sub(r"#([a-zA-Z0-9_À-ỹ]+)", r"\1", text)
            text = re.sub(r"([!?.,;:])\1+", r"\1", text)

        # 4. teencode_emoji: Ánh xạ teencode và emoji
        if self.teencode_emoji:
            for emoji_char, meaning in EMOJI_DICT.items():
                text = text.replace(emoji_char, f" {meaning} ")
            emoji_pattern = re.compile(
                r"[\U00010000-\U0010ffff\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F\U0000200D]",
                flags=re.UNICODE
            )
            text = emoji_pattern.sub("", text)

        # 5. repeat_chars: Rút gọn ký tự lặp trên chữ cái
        if self.repeat_chars:
            tokens = text.split()
            normalized_tokens = []
            for token in tokens:
                token = re.sub(r"([^\W\d_])\1{2,}", r"\1", token)
                if self.teencode_emoji and token in TEENCODE_DICT:
                    normalized_tokens.append(TEENCODE_DICT[token])
                else:
                    normalized_tokens.append(token)
            text = " ".join(normalized_tokens)
        elif self.teencode_emoji:
            tokens = text.split()
            normalized_tokens = [TEENCODE_DICT.get(t, t) for t in tokens]
            text = " ".join(normalized_tokens)

        text = re.sub(r"\s+", " ", text).strip()

        # 6. word_segmentation: Phân tách từ ghép (pyvi)
        if self.word_segmentation:
            try:
                text = ViTokenizer.tokenize(text)
            except Exception:
                self.segmentation_error_count += 1

        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.clean_text(t) for t in texts]

def _get_normalization_key(text: str) -> str:
    no_punct = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"DataFrame phải chứa các cột '{text_col}' và '{label_col}'.")

    print(f"🧹 Bắt đầu tiền xử lý DataFrame (Tổng số mẫu ban đầu: {len(df)})")
    preprocessor = TextPreprocessor(
        lowercase=True,
        clean_url_html=True,
        hashtag_punct=True,
        teencode_emoji=True,
        repeat_chars=True,
        word_segmentation=True
    )

    df = df.dropna(subset=[text_col, label_col]).copy()
    df['cleaned_text'] = df[text_col].astype(str).apply(preprocessor.clean_text)

    if preprocessor.segmentation_error_count > 0:
        print(f"⚠️ [Cảnh báo] Số câu gặp lỗi khi tách từ bằng pyvi: {preprocessor.segmentation_error_count}")
    else:
        print(f"✅ Hoàn tất tách từ không có lỗi (Số lỗi: {preprocessor.segmentation_error_count})")

    df = df[df['cleaned_text'].str.strip().str.len() > 0].copy()
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 3].copy()

    df['norm_key'] = df['cleaned_text'].apply(_get_normalization_key)
    label_counts_per_key = df.groupby('norm_key')[label_col].nunique()
    conflicting_keys = label_counts_per_key[label_counts_per_key > 1].index
    n_conflicting = df[df['norm_key'].isin(conflicting_keys)].shape[0]
    
    if n_conflicting > 0:
        print(f"   - Đã loại bỏ {n_conflicting} mẫu do có nhãn xung đột (cùng nội dung nhưng khác nhãn).")
    df = df[~df['norm_key'].isin(conflicting_keys)].copy()

    initial_len = len(df)
    df = df.drop_duplicates(subset=['norm_key'], keep='first').copy()
    print(f"   - Đã loại bỏ {initial_len - len(df)} mẫu trùng lặp.")
    print(f"✅ Số lượng mẫu sau khi tiền xử lý hoàn tất: {len(df)}")

    return df[[text_col, 'cleaned_text', label_col]].rename(columns={text_col: 'original_text', 'cleaned_text': 'text'})
