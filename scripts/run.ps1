param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("install", "test", "data", "features", "train", "evaluate", "baselines", "ablation", "all", "quick", "serve")]
    [string]$Command
)

switch ($Command) {
    "install" {
        Write-Host "📦 Đang cài đặt thư viện từ requirements.txt..." -ForegroundColor Cyan
        python -m pip install -r requirements.txt
    }
    "test" {
        Write-Host "🧪 Đang chạy bộ kiểm thử pytest..." -ForegroundColor Cyan
        python -m pytest tests -q
    }
    "data" {
        Write-Host "📂 Đang phân chia dữ liệu (80/10/10)..." -ForegroundColor Cyan
        python -m src.preprocessing.split_data data/raw/vn_news_merged.csv
    }
    "features" {
        Write-Host "🔄 Đang trích xuất đặc trưng PhoBERT [CLS] 768 chiều..." -ForegroundColor Cyan
        python -m src.features.extract_phobert
    }
    "train" {
        Write-Host "⚙️ Đang huấn luyện và tuning SVM..." -ForegroundColor Cyan
        python -m src.models.train_svm
    }
    "evaluate" {
        Write-Host "📈 Đang đánh giá mô hình trên tập Test..." -ForegroundColor Cyan
        python -m src.evaluation.evaluate
    }
    "baselines" {
        Write-Host "📊 Đang chạy so sánh các mô hình baseline..." -ForegroundColor Cyan
        python -m src.evaluation.compare_models
    }
    "ablation" {
        Write-Host "🔬 Đang chạy nghiên cứu Ablation Study..." -ForegroundColor Cyan
        python -m experiments.ablation_preprocessing
    }
    "all" {
        Write-Host "🚀 Đang chạy toàn bộ quy trình thực nghiệm (Full Mode)..." -ForegroundColor Cyan
        python -m experiments.run_all
    }
    "quick" {
        Write-Host "⚡ Đang chạy quy trình thử nghiệm nhanh (Quick Mode)..." -ForegroundColor Cyan
        python -m experiments.run_all --quick
    }
    "serve" {
        Write-Host "🌐 Đang khởi chạy Flask REST API tại http://127.0.0.1:5000..." -ForegroundColor Cyan
        python -m app.app
    }
}
