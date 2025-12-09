# 🎯 Ad Click-Through Rate (CTR) Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

A production-ready machine learning system for predicting ad click-through rates, demonstrating full ML lifecycle from data engineering to deployment.

---

## 🚀 Project Overview

This project showcases **end-to-end ML engineering** for a real-world ad tech problem:

### What It Does
Predicts whether a user will click on a display advertisement based on:
- **User behavior**: Device type, location, time of day
- **Ad characteristics**: Position, format, site context
- **Historical patterns**: Click rates, engagement metrics

### Why It Matters
Accurate CTR prediction enables:
- ✅ **Optimized ad placement** - Better targeting = higher revenue
- ✅ **Smart bidding strategies** - Bid more on high-CTR opportunities
- ✅ **Improved user experience** - Show relevant ads
- ✅ **Revenue optimization** - Maximize value per impression

---

## 🏗️ Architecture & Tech Stack

### ML/AI Components
- **PyTorch** - Deep neural network with batch normalization and dropout
- **XGBoost** - Gradient boosting (industry standard for tabular data)
- **scikit-learn** - Logistic regression baseline, preprocessing pipelines

### MLOps & Deployment
- **MLflow** - Experiment tracking, model versioning, parameter logging
- **FastAPI** - Production REST API with Pydantic validation
- **Docker** - Containerized deployment
- **Hash Encoding** - Memory-efficient encoding for high-cardinality features

### Data Engineering
- **Feature Engineering** - Time-based features, cyclical encoding, hash encoding
- **Pandas/NumPy** - Data processing and manipulation
- **Train/Val/Test Split** - Proper stratified splitting

---

## 📊 Results

### Model Performance

| Model | Validation AUC-ROC | Parameters | Training Time |
|-------|-------------------|------------|---------------|
| **Logistic Regression** | 0.72 | ~65K | ~2 min |
| **XGBoost** | 0.76 | ~100 trees | ~5 min |
| **PyTorch Neural Net** | 0.78 | 33.7M | ~10 min |

🏆 **Best Model**: PyTorch Neural Network (AUC-ROC: 0.78)

### Key Insights
- Deep learning outperforms traditional ML for this task
- Hash encoding successfully handles millions of unique categorical values
- Early stopping prevents overfitting (patience=3 epochs)

---

## 🛠️ Technical Implementation

### 1. Feature Engineering
```python
# 65,556 total features from:
- 8 time-based features (hour, day, cyclical encoding)
- 12 numerical features (scaled)
- 262,144 hash-encoded categorical features (site_id, device_id, etc.)
```

**Why Hash Encoding?**
- Site IDs: ~5,000 unique values
- Device IDs: ~10,000 unique values
- One-hot encoding would create millions of sparse features
- Hash encoding: Fixed 262K features, O(1) lookup

### 2. Model Architecture (PyTorch)
```
Input (65,556) → Linear(512) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(1) → Sigmoid → Output [0, 1]
```

### 3. Training Pipeline
- **Data**: 50K impressions (train/val/test: 70%/10%/20%)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy
- **Early Stopping**: Patience=3 epochs on validation loss
- **Batch Size**: 256
- **Device**: Auto-detect (CUDA/MPS/CPU)

### 4. MLflow Tracking
All experiments logged with:
- ✅ Hyperparameters (learning rate, batch size, architecture)
- ✅ Metrics (AUC-ROC, accuracy, precision, recall, F1)
- ✅ Training curves (loss per epoch)
- ✅ Model artifacts (saved weights)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AdClickPrediction.git
cd AdClickPrediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_synthetic_data.py
```

### Train Models

```bash
# Train all models (Logistic Regression, XGBoost, PyTorch)
python src/training/train.py

# View experiment results
mlflow ui
# Open http://localhost:5000
```

### Run API

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload

# View API docs
# Open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# API: http://localhost:8000
# MLflow: http://localhost:5000
```

---

## 📁 Project Structure

```
AdClickPrediction/
├── data/
│   ├── raw/train.csv           # 100K synthetic ad impressions
│   └── processed/              # Preprocessed features
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data loading & splitting
│   ├── features/
│   │   └── feature_engineering.py  # Feature pipeline
│   ├── models/
│   │   └── pytorch_model.py    # Neural network definition
│   ├── training/
│   │   └── train.py            # Training script with MLflow
│   └── api/
│       └── main.py             # FastAPI service
├── models/                      # Saved model artifacts
├── mlruns/                      # MLflow experiment logs
├── scripts/
│   └── generate_synthetic_data.py
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Multi-container setup
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🔧 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": "14102207",
    "C1": 1005,
    "banner_pos": 0,
    "site_id": "site_1234",
    ...
  }'
```

**Response:**
```json
{
  "click_probability": 0.2341,
  "prediction": "no_click",
  "confidence": "high"
}
```

### Batch Predictions

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "impressions": [
      { "hour": "14102207", ... },
      { "hour": "14102208", ... }
    ]
  }'
```

---

## 🎓 Key Learnings & Design Decisions

### Why PyTorch Over TensorFlow?
- More Pythonic API
- Better for research and experimentation
- Explicit control over training loop
- Required by job description

### Why XGBoost?
- Industry standard for CTR prediction
- Handles categorical features natively
- Fast training and inference
- Excellent baseline for tabular data

### Why Hash Encoding?
- Millions of unique categorical values (site_id, device_id)
- One-hot encoding = memory explosion
- Hash encoding = fixed size, no collisions in practice
- O(1) lookup, much faster

### Why MLflow?
- Experiment tracking (compare 100s of runs)
- Model versioning and registry
- Reproducibility (log everything)
- Industry standard for MLOps

---

## 📈 Future Improvements

- [ ] Deploy to Railway/Render
- [ ] Add A/B testing simulation
- [ ] Implement model monitoring dashboard
- [ ] Add feature importance analysis
- [ ] Hyperparameter tuning with Optuna
- [ ] Real Kaggle dataset (Avazu CTR)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Load testing with Locust

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Avazu CTR Prediction** - Kaggle competition inspiration
- **MLflow** - Experiment tracking
- **PyTorch** - Deep learning framework
- **FastAPI** - Modern API framework

---

**Built by Tomer Sadeh** | [LinkedIn](https://linkedin.com/in/tomer-sadeh) | [GitHub](https://github.com/tomersadeh123)
