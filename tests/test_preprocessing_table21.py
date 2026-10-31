import pytest
import pandas as pd
from src.preprocessing.preprocess import TextPreprocessor, _get_normalization_key, preprocess_dataframe

def test_table21_sample():
    # (a) Câu mẫu Bảng 2.1 (use_word_segmentation=False)
    sample_text = "SỐC!! ko thể tin đc link URL_PLACEHOLDER #tin_nong dịch bệnh bùng phát 😡"
    preprocessor = TextPreprocessor(use_word_segmentation=False)
    cleaned = preprocessor.clean_text(sample_text)
    
    assert "không" in cleaned
    assert "được" in cleaned
    assert "HTTPURL" in cleaned
    assert "tin_nong" in cleaned
    assert "tức giận" in cleaned

def test_number_unchanged():
    # (b) Số không bị đổi bởi quy tắc rút gọn ký tự lặp
    text = "2000 người, 10000 đồng"
    preprocessor = TextPreprocessor(use_word_segmentation=False)
    cleaned = preprocessor.clean_text(text)
    assert "2000" in cleaned
    assert "10000" in cleaned

def test_symbols_and_formatting_retained():
    # Test ký hiệu %, $, -, (), số và chữ được giữ nguyên
    text = "tăng 100%, giá $5, Covid-19 (nguy hiểm)"
    preprocessor = TextPreprocessor(use_word_segmentation=False)
    cleaned = preprocessor.clean_text(text)
    assert "%" in cleaned
    assert "$" in cleaned
    assert "-" in cleaned
    assert "(" in cleaned
    assert ")" in cleaned
    assert "100" in cleaned
    assert "5" in cleaned
    assert "covid-19" in cleaned

def test_punctuation_duplicate_normalization():
    # (c) Hai câu chỉ khác dấu câu bị coi là trùng qua _get_normalization_key
    sent1 = "Tin tức nóng hổi hôm nay!"
    sent2 = "Tin tức nóng hổi hôm nay?"
    
    preprocessor = TextPreprocessor(use_word_segmentation=False)
    c1 = preprocessor.clean_text(sent1)
    c2 = preprocessor.clean_text(sent2)
    
    key1 = _get_normalization_key(c1)
    key2 = _get_normalization_key(c2)
    assert key1 == key2

def test_conflicting_labels_dropped():
    # (d) Văn bản trùng khác nhãn bị loại hoàn toàn
    data = {
        "text": [
            "Bài viết kiểm tra tin tức số một trên mạng xã hội.",
            "Bài viết kiểm tra tin tức số một trên mạng xã hội!",  # trùng normalized key với dòng 1
            "Bài viết kiểm tra tin tức số một trên mạng xã hội.",  # trùng normalized key với dòng 1
            "Một nội dung hoàn toàn khác biệt để kiểm tra hệ thống phân loại."
        ],
        "label": [0, 1, 0, 1]  # dòng 1 và dòng 2 nội dung giống nhau nhưng nhãn khác nhau (0 và 1) -> xung đột!
    }
    df = pd.DataFrame(data)
    result_df = preprocess_dataframe(df, text_col="text", label_col="label")
    
    texts = result_df['original_text'].tolist()
    assert "Bài viết kiểm tra tin tức số một trên mạng xã hội." not in texts
    assert "Một nội dung hoàn toàn khác biệt để kiểm tra hệ thống phân loại." in texts
