# Streamlit is provided by the application's runtime environment.
# pyright: reportMissingImports=false
import streamlit as st
import requests  # pyright: ignore[reportMissingModuleSource]
import re
import plotly.graph_objects as go
import streamlit.components.v1 as components
from core.scraper import NewsScraper
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Xác thực Tin tức", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để trang trí đẹp hơn
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    [data-testid="stMain"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #111827;
    }
    .main-header {
        background: linear-gradient(135deg, #1d4ed8 0%, #0f766e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .result-box-fake {
        background: #fee;
        border-left: 5px solid #ff4444;
        padding: 1.5rem;
        border-radius: 5px;
        color: #111827 !important;
    }
    .result-box-real {
        background: #efe;
        border-left: 5px solid #44ff44;
        padding: 1.5rem;
        border-radius: 5px;
        color: #111827 !important;
    }
    .result-box-fake h1, .result-box-fake h2, .result-box-fake h3, .result-box-fake p,
    .result-box-real h1, .result-box-real h2, .result-box-real h3, .result-box-real p {
        color: #111827 !important;
    }
    .reason-item {
        background: #f9f9f9;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        border-radius: 3px;
        color: #111827 !important;
    }
    .reason-item * {
        color: #111827 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #111827 !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #6b7280 !important;
    }
    .stAlert {
        color: #111827 !important;
    }
    .stAlert * {
        color: #111827 !important;
    }
    [data-baseweb="tag"] {
        background: #eef2ff !important;
        color: #1f2937 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Hệ thống Xác thực Tin tức Tiếng Việt</h1>
        <p>Phát hiện tin giả dựa trên AI (PhoBERT) + Kiểm chứng chéo nguồn tin chính thống</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("")

# Sidebar
with st.sidebar:
    st.header("Hướng dẫn Sử dụng")
    st.info("""
    **Bước 1:** Nhập nội dung tin tức hoặc dán link báo chí
    
    **Bước 2:** Nhấn 'Bắt đầu Kiểm tra'
    
    **Bước 3:** Xem kết quả phân tích đa tầng
    
    **Mẹo:** Sử dụng URL từ báo chí để hệ thống tự động cào nội dung
    """)
    
    st.divider()
    st.header("Cài đặt")
    api_url = st.text_input("API Backend URL:", value="http://localhost:8000", key="api_url")
    
    st.divider()
    st.header("Về Hệ thống")
    st.markdown("""
    **Phương pháp phân tích:**
    - **PhoBERT (50%)** - Deep Learning dò tìm mẫu tin giả
    - **Style Analysis (10%)** - Phân tích phong cách viết
    - **Search Verification (40%)** - Kiểm chứng chéo nguồn
    
    **Ngưỡng quyết định:** > 50% = TIN GIẢ
    """)

# Khởi tạo Scraper
scraper = NewsScraper()

# Input section
col1, col2 = st.columns([4, 1])
with col1:
    input_data = st.text_area(
        "Nhập nội dung tin tức hoặc dán URL báo chí:",
        height=150,
        placeholder="Ví dụ: https://vnexpress.net/... hoặc nội dung tin tức..."
    )

with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    check_button = st.button("Bắt đầu Kiểm tra", use_container_width=True)

def is_url(string):
    return re.match(r'http[s]?://', string) is not None


def parse_datetime_safe(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y"
    ]:
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def format_datetime_safe(value):
    dt = parse_datetime_safe(value)
    if not dt:
        return "Không xác định"
    return dt.strftime("%Y-%m-%d %H:%M")

if check_button:
    if not input_data or len(input_data.strip()) < 20:
        st.warning("Vui lòng nhập nội dung ít nhất 20 ký tự để phân tích.")
    else:
        final_text = ""
        source_url = None
        input_publish_date = None
        
        # Xử lý URL
        if is_url(input_data):
            source_url = input_data.strip()
            with st.spinner("🌐 Đang cào dữ liệu từ URL..."):
                scraped = scraper.get_content(input_data)
                if scraped["success"]:
                    final_text = scraped["text"]
                    input_publish_date = scraped.get("publish_date")
                    st.success(f"Đã lấy được bài báo: **{scraped['title']}**")
                else:
                    st.error(f"Lỗi Scraper: {scraped['error']}")
        else:
            final_text = input_data

        if final_text:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🧠 Đang phân tích bằng AI..."):
                try:
                    progress_bar.progress(30)
                    status_text.text("⏳ Xử lý text...")
                    
                    # Gửi yêu cầu đến FastAPI Backend
                    response = requests.post(
                        f"{api_url}/predict", 
                        json={"text": final_text, "source_url": source_url},
                        timeout=60
                    )
                    
                    if response.status_code != 200:
                        st.error(f"Lỗi Backend: {response.text}")
                    else:
                        progress_bar.progress(70)
                        status_text.text("⏳ Sinh bản báo cáo...")
                        
                        result = response.json()
                        progress_bar.progress(100)
                        status_text.text("Phân tích xong!")
                        
                        # --- BẮT ĐẦU PHẦN HIỂN THỊ KẾT QUẢ ---
                        st.divider()
                        
                        # 1. Kết quả tổng quan với biểu đồ gauge
                        risk_score = result['final_risk_score']
                        is_fake = result['is_fake']
                        
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            # Hiển thị Gauge Chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=risk_score * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Mức Độ Rủi Ro (%)"},
                                delta={'reference': 50, 'increasing': {'color': "red"}},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "#90EE90"},   # Green
                                        {'range': [30, 60], 'color': "#FFD700"},  # Yellow
                                        {'range': [60, 100], 'color': "#FF6B6B"}  # Red
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 3},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("")
                            st.markdown("")
                            if is_fake:
                                st.markdown("""
                                    <div class="result-box-fake">
                                    <h3>CẢNH BÁO: TIN GIẢ</h3>
                                    <p>Hệ thống phát hiện khả năng cao đây là tin giả. 
                                    Vui lòng xác minh thêm từ các nguồn tin chính thống.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="result-box-real">
                                    <h3>TIN CẬY: KHẢ NĂNG TIN THẬT</h3>
                                    <p>Hệ thống không phát hiện dấu hiệu tin giả rõ rệt. 
                                    Tuy nhiên, vẫn nên kiểm chứng thêm.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("")
                            st.markdown("")
                            if result["components"].get("has_debunk_source"):
                                st.error("Có báo chính thống bác bỏ")
                            else:
                                st.success("Không có bác bỏ")
                            
                            st.metric(
                                "Thời xử lý",
                                f"{result['components'].get('processing_time_seconds', 0):.2f}s"
                            )

                            source_status = result["components"].get("input_source_status", "no_source_url")
                            source_trust_score = result["components"].get("input_source_trust_score", 0.5)
                            source_domain = result["components"].get("input_source_domain")
                            source_name = result["components"].get("input_source_name")

                            st.metric(
                                "Input Source Trust",
                                f"{source_trust_score:.2f}/1.00",
                                delta=source_status,
                                delta_color="off"
                            )
                            if source_status == "trusted":
                                st.success(f"Nguồn uy tín: {source_name}")
                            elif source_status == "unverified":
                                st.warning(f"Nguồn chưa kiểm chứng: {source_domain}")
                            else:
                                st.info("Không có URL nguồn đầu vào")

                            st.metric(
                                "Ngày đăng input",
                                format_datetime_safe(input_publish_date),
                                delta="Scrape được" if input_publish_date else "Fallback"
                            )

                            if input_publish_date:
                                st.caption(f"Input publish time: {format_datetime_safe(input_publish_date)}")
                            else:
                                st.caption("Không scrape được ngày đăng của bài input, đang fallback theo trạng thái nguồn.")
                        
                        st.divider()
                        
                        # 2. Phần giải thích chi tiết
                        st.subheader("Tại sao hệ thống đánh giá như vậy?")
                        
                        if result["reasons"]:
                            for reason in result["reasons"]:
                                st.markdown(f"""
                                    <div class="reason-item">
                                    {reason}
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Phân tích dựa trên sự kết hợp của các chỉ số kỹ thuật bên dưới.")

                        cross_check = result.get("cross_check", {})

                        # 3. Cross-check nguồn tin (Top 3 scrape đầy đủ + nguồn còn lại)
                        st.subheader("Claim Extraction & Đối Chứng")
                        claims = cross_check.get("claims", []) if isinstance(cross_check, dict) else []

                        if claims:
                            for claim in claims:
                                claim_id = claim.get("claim_id", "?")
                                claim_text = claim.get("claim_text", "")
                                verdict = claim.get("verdict", "no_evidence")
                                verdict_conf = claim.get("verdict_confidence", 0.0)
                                support_sources = claim.get("support_sources", 0)
                                contradict_sources = claim.get("contradict_sources", 0)
                                insufficient_sources = claim.get("insufficient_sources", 0)
                                checkworthiness = claim.get("checkworthiness", 0.0)

                                if verdict == "support":
                                    verdict_badge = "Support"
                                elif verdict == "contradict":
                                    verdict_badge = "Contradict"
                                elif verdict == "insufficient":
                                    verdict_badge = "Insufficient"
                                elif verdict == "unrelated":
                                    verdict_badge = "Unrelated"
                                else:
                                    verdict_badge = "No evidence"

                                with st.expander(f"{claim_id} | {verdict_badge} | conf={verdict_conf:.2f}", expanded=False):
                                    st.write(claim_text)
                                    cols = st.columns(4)
                                    cols[0].metric("Checkworthiness", f"{checkworthiness:.2f}")
                                    cols[1].metric("Support", support_sources)
                                    cols[2].metric("Contradict", contradict_sources)
                                    cols[3].metric("Insufficient", insufficient_sources)

                                    matched_sources = claim.get("matched_sources", [])
                                    if matched_sources:
                                        st.markdown("**Nguồn đối chứng cho claim:**")
                                        for src in matched_sources:
                                            source_name = src.get("source_name", "N/A")
                                            stance = src.get("stance", "n/a")
                                            sim = src.get("similarity", 0.0)
                                            st.caption(f"- {source_name} | stance={stance} | similarity={sim}")
                                    else:
                                        st.info("Claim này chưa có nguồn đối chứng rõ ràng.")
                        else:
                            st.info("Chưa tách được claim rõ ràng từ input.")

                        # 4. Cross-check nguồn tin (Top 3 scrape đầy đủ + nguồn còn lại)
                        st.subheader("Cross-check Nguồn Tin")
                        top_scraped_articles = cross_check.get("top_scraped_articles", [])
                        other_results = cross_check.get("other_results", [])

                        if top_scraped_articles:
                            st.markdown("**Top 3 bài báo được scrape đầy đủ:**")
                            input_dt = parse_datetime_safe(input_publish_date)
                            for article in top_scraped_articles:
                                title = article.get("title", "(Không có tiêu đề)")
                                source_name = article.get("source_name", article.get("domain", "Nguồn không xác định"))
                                similarity = article.get("similarity", 0.0)
                                link = article.get("link", "")
                                rank = article.get("rank", "?")
                                is_debunking = article.get("is_debunking", False)
                                published_at = article.get("published_at") or article.get("scraped", {}).get("publish_date")
                                article_dt = parse_datetime_safe(published_at)
                                temporal_delta_days = None
                                temporal_delta_label = "Không xác định"
                                if input_dt and article_dt:
                                    temporal_delta_days = abs((article_dt - input_dt).days)
                                    temporal_delta_label = f"{temporal_delta_days} ngày"
                                debunk_badge = " | Dấu hiệu bác bỏ" if is_debunking else ""

                                with st.expander(f"#{rank} {title}", expanded=(rank == 1)):
                                    publish_badge = f" | Publish: {format_datetime_safe(published_at)}" if published_at else ""
                                    temporal_badge = f" | Δ thời gian so với input: {temporal_delta_label}"
                                    st.caption(f"Nguồn: {source_name} | Similarity: {similarity}{debunk_badge}{publish_badge}")
                                    st.caption(f"Đối chứng thời gian: {temporal_delta_label} so với bài input")
                                    if link:
                                        st.link_button("Mở link gốc", link)

                                    claim_matches = article.get("claim_matches", [])
                                    if claim_matches:
                                        st.markdown("**Đối chứng theo claim với bài báo này:**")
                                        for m in claim_matches:
                                            st.caption(
                                                f"- {m.get('claim_id')} | stance={m.get('stance')} | "
                                                f"sim={m.get('similarity')} | overlap={m.get('keyword_overlap')} | "
                                                f"conf={m.get('confidence')}"
                                            )

                                    st.markdown("**Xem trực tiếp từ nguồn gốc (embedded):**")
                                    if link:
                                        components.iframe(src=link, height=650, scrolling=True)
                                    else:
                                        st.info("Không có link để nhúng trực tiếp.")

                                    scraped = article.get("scraped", {})
                                    if scraped.get("success"):
                                        if scraped.get("top_image"):
                                            st.image(scraped.get("top_image"), use_container_width=True)
                                        authors = scraped.get("authors") or []
                                        if authors:
                                            st.caption(f"Tác giả: {', '.join(authors)}")
                                        st.markdown("**Nội dung đã scrape (fallback khi trang chặn nhúng):**")
                                        st.write(scraped.get("text", ""))
                                    else:
                                        st.warning(f"Không thể scrape đầy đủ bài này: {scraped.get('error', 'Unknown error')}")
                        else:
                            st.info("Không có dữ liệu top 3 nguồn để hiển thị.")

                        if other_results:
                            st.markdown("**Các nguồn còn lại (title + snippet + link):**")
                            input_dt = parse_datetime_safe(input_publish_date)
                            for item in other_results:
                                rank = item.get("rank", "?")
                                title = item.get("title", "(Không có tiêu đề)")
                                snippet = item.get("snippet", "")
                                link = item.get("link", "")
                                source_name = item.get("source_name", item.get("domain", "Nguồn không xác định"))
                                published_at = item.get("published_at")
                                article_dt = parse_datetime_safe(published_at)
                                if input_dt and article_dt:
                                    delta_days = abs((article_dt - input_dt).days)
                                    temporal_text = f"Δ {delta_days} ngày"
                                else:
                                    temporal_text = "Δ không xác định"
                                st.markdown(f"**#{rank} {title}**")
                                st.caption(f"Nguồn: {source_name} | {temporal_text} | Publish: {format_datetime_safe(published_at)}")
                                if snippet:
                                    st.write(snippet)
                                if link:
                                    st.link_button(f"Mở bài #{rank}", link)
                                st.divider()
                        else:
                            st.info("Không có nguồn bổ sung ngoài top 3.")
                        
                        # 4. Biểu đồ thành phần
                        st.subheader("Đóng góp rủi ro từ các tầng phân tích")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Bar chart
                            components = result["components"]
                            chart_data = {
                                "PhoBERT\n(Deep Learning)": components["phobert_risk"],
                                "Style Analysis\n(ML)": components["style_risk"],
                                "Thiếu bằng chứng\n(Search)": 1 - components["search_evidence_score"]
                            }
                            st.bar_chart(chart_data)
                        
                        with col2:
                            # Metric cards
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric(
                                    "Rủi ro PhoBERT",
                                    f"{components['phobert_risk']*100:.1f}%",
                                    delta=f"{(components['phobert_risk']-0.5)*100:.1f}%" if components['phobert_risk'] != 0.5 else None
                                )
                            with subcol2:
                                st.metric(
                                    "Rủi ro Style",
                                    f"{components['style_risk']*100:.1f}%",
                                    delta=f"{(components['style_risk']-0.5)*100:.1f}%" if components['style_risk'] != 0.5 else None
                                )
                            
                            st.metric(
                                "Điểm bằng chứng",
                                f"{components['search_evidence_score']:.2f}/1.00",
                                delta=f"{'Xác thực' if components['search_evidence_score'] > 0.5 else 'Chưa xác thực'}",
                                delta_color="off"
                            )

                            st.metric(
                                "Uy tín nguồn input",
                                f"{components.get('input_source_trust_score', 0.5):.2f}/1.00",
                                delta=components.get("input_source_status", "no_source_url"),
                                delta_color="off"
                            )
                        
                        # 5. Chi tiết kỹ thuật (Tùy chọn)
                        with st.expander("Chi tiết kỹ thuật (Dành cho chuyên gia)", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Raw Components:**")
                                st.json(result["components"])
                            with col2:
                                st.write("**Toàn bộ Response:**")
                                st.write(result)
                        
                        # 6. Feedback section
                        st.divider()
                        st.subheader("Đánh giá kết quả")
                        col1, col2 = st.columns(2)
                        with col1:
                            feedback = st.radio(
                                "Kết quả này có chính xác không?",
                                ["Chính xác", "Không chính xác", "Không chắc"],
                                horizontal=True
                            )
                        with col2:
                            submit_feedback = st.button("Gửi Feedback")
                            if submit_feedback:
                                st.success(f"Cảm ơn bạn đã gửi feedback: {feedback}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Lỗi kết nối Backend. Đảm bảo bạn đã chạy:\n```\npython api.py\n```")
                except requests.exceptions.Timeout:
                    st.error("Timeout: Phân tích mất quá lâu. Thử lại với nội dung ngắn hơn.")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
