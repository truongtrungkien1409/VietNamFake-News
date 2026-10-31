from newspaper import Article, Config
from datetime import datetime

class NewsScraper:
    def __init__(self):
        # Cấu hình user-agent để tránh bị một số báo chặn (giả lập trình duyệt)
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'

    def get_content(self, url):
        """Tải và trích xuất nội dung từ URL báo chí"""
        try:
            config = Config()
            config.browser_user_agent = self.user_agent
            config.request_timeout = 15
            config.fetch_images = False

            article = Article(url, config=config, language='vi')
            article.download()
            article.parse()

            publish_date = None
            raw_publish_date = getattr(article, "publish_date", None)
            if raw_publish_date:
                if isinstance(raw_publish_date, datetime):
                    publish_date = raw_publish_date.isoformat()
                else:
                    publish_date = str(raw_publish_date)

            if not publish_date:
                meta_candidates = []
                meta_data = getattr(article, "meta_data", {}) or {}
                meta_candidates.extend([
                    meta_data.get("article:published_time"),
                    meta_data.get("og:published_time"),
                    meta_data.get("publish_date"),
                    meta_data.get("pubdate"),
                    meta_data.get("date")
                ])

                top_node = meta_data.get("parsely", {}) if isinstance(meta_data, dict) else {}
                if isinstance(top_node, dict):
                    meta_candidates.extend([
                        top_node.get("pub_date"),
                        top_node.get("published_at"),
                        top_node.get("pubdate")
                    ])

                for candidate in meta_candidates:
                    if candidate:
                        publish_date = str(candidate)
                        break

            if publish_date:
                try:
                    parsed_dt = datetime.fromisoformat(str(publish_date).replace("Z", "+00:00"))
                    publish_date = parsed_dt.isoformat()
                except Exception:
                    pass
            
            # Kết hợp tiêu đề và nội dung để có dữ liệu đầy đủ nhất cho checking
            full_text = f"{article.title}. {article.text}"
            
            if len(article.text) < 50:
                print(f"⚠️ Cảnh báo: Nội dung lấy được quá ngắn ({len(article.text)} ký tự).")
            
            return {
                "success": True,
                "title": article.title,
                "text": full_text,
                "top_image": article.top_image,
                "authors": article.authors,
                "publish_date": publish_date
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }