import re
import unicodedata
from pyvi import ViTokenizer

# Từ điển ánh xạ teencode và từ viết tắt phổ biến
TEENCODE_DICT = {
    "ko": "không",
    "k": "không",
    "kg": "không",
    "kô": "không",
    "khg": "không",
    "đc": "được",
    "dc": "được",
    "được": "được",
    "j": "gì",
    "gi": "gì",
    "ntn": "như thế này",
    "như_thế_này": "như thế này",
    "mk": "mình",
    "mình": "mình",
    "mik": "mình",
    "vn": "việt nam",
    "hcm": "hồ chí minh",
    "tpHCM": "thành phố hồ chí minh",
    "ib": "nhắn tin",
    "online": "trực tuyến",
    "rip": "thành kính phân ưu",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "thanh_ks": "cảm ơn",
    "ae": "anh em",
    "b": "bạn",
    "ẩu": "ẩu",
    "bt": "bình thường",
    "t": "tao",
    "tao": "tao",
    "m": "mày",
    "mày": "mày"
}

# Từ điển ánh xạ Emoji/Icon sang từ ngữ diễn đạt cảm xúc
EMOJI_DICT = {
    "😡": "tức_giận",
    "😠": "tức_giận",
    "🤬": "tức_giận",
    "😢": "khóc",
    "😭": "khóc",
    "😂": "cười",
    "😁": "cười",
    "👍": "thích",
    "👎": "không_thích",
    "❤️": "yêu_thích",
    "❤": "yêu_thích",
    "🔥": "nóng",
    "⚠️": "cảnh_báo",
    "🚨": "khẩn_cấp",
    "❓": "nghi_vấn",
    "❗": "nhấn_mạnh"
}

def clean_and_preprocess_text(text: str) -> str:
    """
    Quy trình tiền xử lý 5 bước chuyên biệt cho văn bản tiếng Việt trên mạng xã hội (theo mục 2.2.2.2 DACN.pdf):
    1. Chuẩn hóa mã hóa Unicode NFC và làm sạch ký tự rác (HTML tags, lowercase).
    2. Chuẩn hóa đại diện thực thể mạng xã hội (URL -> HTTPURL, User -> HTTPUSER, bỏ dấu # ở Hashtag).
    3. Xử lý Teencode, từ viết tắt và biểu tượng cảm xúc (Emoji/Icon).
    4. Phân tách từ ghép tiếng Việt (Word Segmentation) bằng công cụ RDRSegmenter (qua pyvi / ViTokenizer).
    5. Loại bỏ trùng lặp / khoảng trắng thừa và chuẩn hóa định dạng.
    """
    if not isinstance(text, str):
        text = str(text)

    # Bước 1: Chuẩn hóa Unicode NFC và chữ thường (lowercase), loại bỏ thẻ HTML/XML
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    
    # Loại bỏ các thẻ HTML tags cơ bản như <br>, <p>, &amp;
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&nbsp;', ' ')

    # Bước 2: Chuẩn hóa URL, User (@mention), và Hashtag (#)
    # Thay thế URL thành HTTPURL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL', text)
    
    # Thay thế @username thành HTTPUSER
    text = re.sub(r'@\w+', 'HTTPUSER', text)
    
    # Xử lý Hashtag: giữ lại phần từ ngữ bên trong, loại bỏ dấu #
    text = re.sub(r'#(\w+)', r'\1', text)

    # Bước 3: Xử lý Emoji/Icon và Teencode
    for emoji, replacement in EMOJI_DICT.items():
        text = text.replace(emoji, f" {replacement} ")

    # Tách từ để xử lý teencode theo từ đơn/cụm từ
    words = text.split()
    processed_words = []
    for w in words:
        # Làm sạch ký tự đặc biệt dính kèm quanh từ
        cleaned_w = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', w)
        if cleaned_w in TEENCODE_DICT:
            w = w.replace(cleaned_w, TEENCODE_DICT[cleaned_w])
        processed_words.append(w)
    text = " ".join(processed_words)

    # Bước 4: Phân tách từ ghép tiếng Việt (Word Segmentation) bằng ViTokenizer (RDRSegmenter equivalent)
    segmented_text = ViTokenizer.tokenize(text)

    # Bước 5: Làm sạch khoảng trắng thừa
    segmented_text = re.sub(r'\s+', ' ', segmented_text).strip()

    return segmented_text
