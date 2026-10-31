# VietNamFake-News
=======
# Vietnamese Fake News Detection System

A comprehensive fake news detection system for Vietnamese news articles using a multi-layer ensemble approach combining deep learning (PhoBERT), traditional machine learning, and search-based verification.

## 📋 Project Overview

This project implements a transparent, auditable fake news detection pipeline that:
- Analyzes Vietnamese news text using **PhoBERT** (Vietnamese BERT-based transformer)
- Scores writing style patterns using rule-based feature extraction
- Verifies claims through cross-source searching and semantic similarity matching
- Assesses input source credibility against curated domain databases
- Provides detailed reasoning for each prediction

**Ensemble Architecture:**
- **PhoBERT Model**: 50% weight - Deep semantic understanding of Vietnamese text
- **Logistic Regression + TF-IDF**: 10% weight - Style-based feature analysis
- **SerpAPI Search Verification**: 40% weight - Cross-source credibility checking

## ✨ Key Features

### 1. Multi-Layer Analysis
- **Text Classification**: PhoBERT transformer detects semantic indicators of fake news
- **Style Analysis**: Detects sensationalism, aggressive language, caps-lock abuse
- **Cross-Source Verification**: Searches Google for similar articles and debunk sources
- **Source Credibility**: Rates input domain against trusted news outlet database

### 2. User-Friendly Interface
- **Streamlit Web App**: Interactive UI with real-time predictions
- **Visual Reasoning**: Gauge charts, metric cards, expandable explanation panels
- **Cross-Check Browser**: Embedded iframes + scraped content from top 3 matching sources
- **Remaining Sources**: Link table for additional verification sources

### 3. Production-Ready
- **API Caching**: LRU cache (100 entries) to avoid duplicate predictions
- **Logging**: Comprehensive logging to file + console with UTF-8 safe encoding
- **Error Handling**: Graceful error messages with detailed exception tracking
- **Security**: SerpAPI key stored in environment variables, not hardcoded
- **CORS Support**: Multi-source frontend access enabled

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Async REST API framework
- **Python 3.8+**: Core runtime
- **transformers**: Hugging Face PhoBERT model loading
- **scikit-learn**: Logistic Regression, TF-IDF vectorization
- **SerpAPI**: Google Search integration
- **newspaper3k**: News article scraping

### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Gauge charts and visualizations

### NLP & Text Processing
- **transformers**: PhoBERT tokenization and inference
- **pyvi**: Vietnamese text tokenization (ViTokenizer)
- **sklearn.feature_extraction**: TF-IDF vectorization

### Data & Models
- **joblib**: Model serialization/deserialization
- **pandas**: Data processing (notebooks)
- **numpy**: Numerical operations

## 📁 Project Structure

```
fake-news/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variable template
├── requirements.txt                   # Python dependencies
│
├── api.py                             # FastAPI backend with caching
├── app_logic.py                       # FakeNewsSystem ensemble predictor
├── main.py                            # CLI entry point
├── webapp.py                          # Streamlit frontend
│
├── core/                              # Core processing modules
│   ├── features.py                    # StyleFeatureExtractor (rule-based analysis)
│   ├── preprocess.py                  # Text preprocessing utilities
│   ├── scraper.py                     # NewsScraper (article extraction)
│   └── search_engine.py               # NewsVerifier (search + credibility)
│
├── ML core/                           # Jupyter notebooks for ML research
│   ├── 01pre-processing.ipynb
│   ├── 02_ml_text_style_model.ipynb
│   ├── 03_ml_text_style_model.ipynb
│   ├── 04_ml_phobert_classifier.ipynb
│   ├── 06_cross_source_check.ipynb
│   └── dataset-1/, datasetvn/         # Training + validation datasets
│
├── ML/                                # Trained models directory
│   ├── data/
│   │   └── vn_fake_news_combined.csv  # Combined training dataset
│   └── models/
│       └── phobert_fake_news/         # PhoBERT model files
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer_config.json
│           └── ...
│
├── notebook/                          # User-facing notebooks
│   └── fake-checker.ipynb             # Interactive fake news checker
│
└── __pycache__/                       # Python cache (excluded from git)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- SerpAPI API key (free tier available at [serpapi.com](https://serpapi.com))

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd fake-news
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # or
   source .venv/bin/activate       # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env          # Windows
     # or
     cp .env.example .env            # Linux/Mac
     ```
   - Add your SerpAPI key to `.env`:
     ```
     SERPAPI_KEY=your_api_key_here
     ```

### Running the Application

#### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run webapp.py
```
Opens at `http://localhost:8501`

#### Option 2: FastAPI Backend
```bash
python api.py
```
Starts server at `http://localhost:8000`

- API endpoint: `POST /predict`
- Interactive docs: `http://localhost:8000/docs`

#### Option 3: CLI Usage
```bash
python main.py
```

## 📖 API Usage

### Request Format
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d {
    "text": "Một bài báo tiềm năng giả mạo",
    "source_url": "https://example.com"
  }
```

### Request Fields
- **text** (required): Vietnamese news text to analyze
- **source_url** (optional): Source URL for credibility assessment

### Response Format
```json
{
  "final_risk_score": 0.72,
  "is_fake": true,
  "timestamp": "2026-03-19T10:30:45.123456",
  "processing_time_seconds": 8.45,
  "components": {
    "phobert_score": 0.65,
    "style_score": 0.88,
    "search_score": 0.70,
    "input_source_trust": {
      "score": 0.3,
      "status": "unverified"
    }
  },
  "cross_check": {
    "top_scraped_articles": [
      {
        "title": "Article Title",
        "source": "news-outlet.vn",
        "url": "https://...",
        "similarity": 0.85,
        "content": "Full article content..."
      }
    ],
    "other_results": [...]
  },
  "reasons": [
    {
      "component": "phobert",
      "reason": "Multiple indicators of sensationalist language",
      "confidence": 0.87
    }
  ]
}
```

## 🔧 Configuration

### Model Paths
Models are loaded from `ML/models/` directory:
- PhoBERT model: `ML/models/phobert_fake_news/`
- Logistic Regression: Referenced in `app_logic.py`
- TF-IDF Vectorizer: Referenced in `app_logic.py`

### Trusted Domains
Curated list in `core/search_engine.py`:
- Vietnamese trusted outlets: VnExpress, TuoiTre, DanTri, etc.
- International outlets: BBC, Reuters, AP News, etc.

### Ensemble Weights
Adjust in `app_logic.py` `predict()` method:
```python
final_risk_score = (
    phobert_score * 0.5 +    # 50%
    style_score * 0.1 +      # 10%
    search_score * 0.4       # 40%
)
```

## 📊 Understanding Outputs

### Risk Score
- **0.0 - 0.3**: Likely Real (🟢 Green)
- **0.3 - 0.7**: Uncertain (🟡 Yellow)
- **0.7 - 1.0**: Likely Fake (🔴 Red)

### Input Source Trust
- **Trusted (1.0)**: Domain is in trusted outlets database
- **Unverified (0.3)**: Domain not in database; requires additional verification
- **Not Available**: No source URL provided

### Components Breakdown
- **PhoBERT Score**: Deep semantic analysis (transformer-based)
- **Style Score**: Writing pattern detection (caps, shock words, aggressive language)
- **Search Score**: Cross-source credibility and debunk detection
- **Input Source Trust**: Domain reputation assessment

## ⚙️ How It Works

### Prediction Pipeline

```
Input: Text + Optional Source URL
  ↓
1. Text Preprocessing
   - Remove HTML/special chars
   - Vietnamese tokenization (pyvi)
  ↓
2. Component Scoring (Parallel)
   ├─ PhoBERT: Semantic understanding
   ├─ Style Features: Rule-based analysis
   └─ Search Verification: Google API lookup
  ↓
3. Ensemble Combination
   - Weighted averaging (50/10/40)
   - Risk score normalization
  ↓
4. Reasoning Generation
   - Component-specific explanations
   - Confidence levels
  ↓
5. Cross-Check Scraping
   - Top 3 source scraping with full content
   - Remaining sources as link list
   - Semantic similarity ranking
  ↓
Output: Prediction + Reasons + Cross-Check + Component Breakdown
```

### PhoBERT Component
- Pretrained Vietnamese BERT-based transformer
- Detects semantic patterns of fake news
- No fine-tuning (uses general Vietnamese understanding)

### Style Component
- Shock word detection (shocking vocabulary)
- Capitalization ratio analysis
- Exclamation mark frequency
- Aggressive/absolute language detection
- Not connected to ML training (display-only currently)

### Search Component
- Google Search API integration via SerpAPI
- TF-IDF cosine similarity matching
- Debunk source detection
- Trusted outlet cross-referencing

## 📝 Logging

### Log Files
- **fake_news_system.log**: Main system logs
- **api.log**: FastAPI request/response logs
- Both files have UTF-8 encoding for Vietnamese character support

### Log Rotation
Currently append-only. Consider adding `RotatingFileHandler` for production.

## 🐛 Known Limitations

1. **Style Features Integration**: StyleFeatureExtractor rules don't feed into LR training (display-only for now)
2. **Cross-Check Timeout**: No retry logic for scraping failures (may fail on slow networks)
3. **Model Fine-Tuning**: Using pretrained PhoBERT; no fine-tuning on Vietnamese fake news corpus
4. **Caching**: In-memory LRU cache (cleared on API restart)
5. **Key Rotation**: If SerpAPI key was hardcoded in git history, regenerate from SerpAPI dashboard
6. **Fact Verification**: No deep fact extraction or knowledge graph verification (Level 1 enhancement)

## 🚀 Future Enhancements

- [ ] Integrate StyleFeatureExtractor into LR model training
- [ ] Add retry/timeout logic to cross-check scraping
- [ ] Implement Knowledge Graph verification for dates/entities
- [ ] User feedback loop with model retraining
- [ ] Docker containerization for deployment
- [ ] Database persistence for caching (Redis)
- [ ] Fine-tune PhoBERT on Vietnamese fake news corpus
- [ ] Multi-language support (Thai, Burmese, etc.)

## 📚 References

### Models & Datasets
- **PhoBERT**: [VinAI Research](https://github.com/VinAI/PhoBERT)
- **Vietnamese Fake News Datasets**: 
  - VFND (Vietnamese Fake News Datasets)
  - Sameer's Fake News Detection dataset
  - Local Vietnamese medical fake news dataset

### APIs & Libraries
- **SerpAPI**: [serpapi.com](https://serpapi.com)
- **Transformers**: [Hugging Face](https://huggingface.co)
- **Streamlit**: [streamlit.io](https://streamlit.io)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

## 👨‍💻 Development

### Running Tests
```bash
pytest tests/  # (Not currently implemented; add test suite)
```

### Code Standards
- Python 3.8+ syntax
- UTF-8 encoding for all files
- Type hints recommended for new code
- Logging for all major operations

## 📄 License

[Specify your license here]

## 👥 Contributing

[Add contribution guidelines if applicable]

## 📧 Support

For issues or questions:
- Check logs in `fake_news_system.log`
- Verify environment variables in `.env`
- Ensure SerpAPI key is valid and has remaining quota

---

**Last Updated**: March 2026  
**Project Status**: Active Development

