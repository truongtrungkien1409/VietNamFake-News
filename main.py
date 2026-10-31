import os
import re
import sys
from app_logic import FakeNewsSystem
from core.scraper import NewsScraper

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Đảm bảo đường dẫn khớp với cấu thư mục "ML core" của bạn
PHOBERT_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "phobert_fake_news")
SVM_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "svm_model.pkl")
SCALER_MODEL = os.path.join(BASE_DIR, "ML core", "ML", "models", "scaler.pkl")

def is_url(string):
    """Kiểm tra định dạng URL"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return re.match(url_pattern, string) is not None

def print_menu():
    print("\n" + "="*40)
    print(" HỆ THỐNG XÁC THỰC TIN TỨC TIẾNG VIỆT")
    print("="*40)
    print("1. Kiểm tra qua đường link (URL)")
    print("2. Kiểm tra bằng cách dán văn bản (Text)")
    print("3. Thoát")
    print("-"*40)

def main():
    # Khởi tạo hệ thống
    try:
        detector = FakeNewsSystem(PHOBERT_MODEL, SVM_MODEL, SCALER_MODEL)
        scraper = NewsScraper()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        return

    while True:
        print_menu()
        choice = input("Vui lòng chọn (1-3): ").strip()

        final_text = ""

        if choice == '1':
            url = input("🔗 Dán link bài báo vào đây: ").strip()
            if is_url(url):
                print(f"🌐 Đang cào nội dung từ URL...")
                scraped_data = scraper.get_content(url)
                if scraped_data["success"]:
                    final_text = scraped_data["text"]
                    print(f"✅ Đã lấy được bài báo: {scraped_data['title']}")
                else:
                    print(f"❌ Lỗi Scraper: {scraped_data['error']}")
                    if "Facebook" in url or "fb.com" in url:
                        print("💡 Mẹo: Facebook chặn truy cập tự động. Hãy dùng Option 2 và dán nội dung bài viết.")
                    continue
            else:
                print("❌ Định dạng link không hợp lệ. Vui lòng thử lại.")
                continue

        elif choice == '2':
            print("📝 Nhập/Dán nội dung tin tức vào bên dưới (Nhấn Enter 2 lần để kết thúc):")
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    break
            final_text = " ".join(lines)
            
            if len(final_text.strip()) < 20:
                print("⚠️ Nội dung quá ngắn để phân tích chính xác.")
                continue

        elif choice == '3':
            print("Cảm ơn bạn đã sử dụng hệ thống. Tạm biệt!")
            sys.exit()
        else:
            print("❌ Lựa chọn không hợp lệ.")
            continue

        # Thực hiện kiểm tra nếu có dữ liệu
        if final_text:
            print(f"\n--- ĐANG KIỂM TRA TIN TỨC ---")
            result = detector.predict(final_text)
            
            # Hiển thị kết quả
            print(f"Nội dung: {final_text[:100]}...")
            status = "⚠️ TIN GIẢ" if result['is_fake'] else "✅ TIN THẬT"
            print(f"Kết luận: {status}")
            print(f"Mức độ rủi ro: {result['final_risk_score'] * 100}%")
            print(f"Chi tiết: {result['components']}")
            
            input("\nNhấn Enter để quay lại menu...")

if __name__ == "__main__":
    main()