from src.preprocessing.preprocess import TextPreprocessor

def test_preprocessing_table21():
    sample_text = "SỐC!! ko thể tin đc link URL_PLACEHOLDER #tin_nong dịch bệnh bùng phát 😡"
    print(f"Original Text: {sample_text}\n")

    preprocessor = TextPreprocessor(use_word_segmentation=True)
    
    # Let's inspect step by step
    # Step 1: Unicode NFC + HTML tags removal + lowercase
    import unicodedata, re
    t1 = unicodedata.normalize("NFC", sample_text)
    t1 = re.sub(r"<[^>]+>", "", t1).lower()
    print(f"Step 1 (NFC + lowercase + HTML removal): {t1}")

    # Step 2: URL -> HTTPURL, mention -> @USER, hashtag -> word, punctuation collapse
    t2 = re.sub(r"https?://\S+|www\.\S+|url_placeholder", "HTTPURL", t1, flags=re.IGNORECASE)
    t2 = re.sub(r"@\[A-Za-z0-9_]+|@\w+", "@USER", t2)
    t2 = re.sub(r"#([a-zA-Z0-9_À-ỹ]+)", r"\1", t2)
    t2 = re.sub(r"([!?.,;:])\1+", r"\1", t2)
    print(f"Step 2 (URL, mention, hashtag, punct): {t2}")

    # Step 3: Teencode & Emoji mapping, repetition collapse
    from src.preprocessing.teencode import TEENCODE_DICT, EMOJI_DICT
    t3 = t2
    for emoji_char, meaning in EMOJI_DICT.items():
        t3 = t3.replace(emoji_char, f" {meaning} ")
    t3 = re.sub(r"[\U00010000-\U0010ffff]", "", t3)
    
    tokens = t3.split()
    norm_tokens = []
    for token in tokens:
        if token in TEENCODE_DICT:
            norm_tokens.append(TEENCODE_DICT[token])
        else:
            norm_tokens.append(token)
    t3 = " ".join(norm_tokens)
    t3 = re.sub(r"\s+", " ", t3).strip()
    print(f"Step 3 (Teencode + Emoji mapping): {t3}")

    # Step 4 & Full pipeline
    final_result = preprocessor.clean_text(sample_text)
    print(f"Step 4 & Final Pipeline (Word Segmentation): {final_result}")

if __name__ == "__main__":
    test_preprocessing_table21()
