import os
import time
import numpy as np
import pandas as pd
from app.app import app

def run_benchmark():
    res_base = os.environ.get("RESULTS_BASE", "results")
    output_csv = os.path.join(res_base, "tables/api_latency.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    test_csv = "data/splits/test.csv"
    texts = []
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        if 'text' in df.columns:
            texts = df['text'].astype(str).tolist()

    if len(texts) < 100:
        # Bổ sung hoặc tạo danh sách mẫu nếu chưa đủ 100
        base_text = "Đây là bản tin thử nghiệm đánh giá hiệu năng thời gian suy luận của hệ thống nhận diện tin giả PhoBERT-SVM."
        while len(texts) < 100:
            texts.append(base_text + f" Mẫu số {len(texts)}")

    texts = texts[:100]
    client = app.test_client()

    latencies = []
    print(f"⏱️ Đang chạy benchmark 100 yêu cầu tuần tự tới Flask API...")

    for idx, txt in enumerate(texts):
        start_t = time.time()
        res = client.post("/predict", json={"text": txt})
        latency_ms = (time.time() - start_t) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    benchmark_data = [{
        "total_requests": len(latencies),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2)
    }]

    df_bench = pd.DataFrame(benchmark_data)
    df_bench.to_csv(output_csv, index=False)

    print("\n================ KẾT QUẢ BENCHMARK API ================")
    print(f"Tổng số yêu cầu : {len(latencies)}")
    print(f"Độ trễ trung bình: {mean_lat:.2f} ms")
    print(f"Độ trễ trung vị  : {median_lat:.2f} ms")
    print(f"Độ trễ P95       : {p95_lat:.2f} ms")
    print(f"Đã lưu kết quả tại: {output_csv}")
    print("========================================================")

if __name__ == "__main__":
    run_benchmark()
