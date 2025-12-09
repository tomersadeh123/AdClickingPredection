# Setup Instructions

## Step 1: Get the Dataset

### Option 1: Kaggle Dataset (Recommended - Real Data)

1. **Create Kaggle Account**
   - Go to https://www.kaggle.com
   - Sign up or log in

2. **Download Avazu CTR Dataset**
   - Visit: https://www.kaggle.com/c/avazu-ctr-prediction/data
   - Click "Download All"
   - Accept competition rules
   - Extract `train.csv` to `data/raw/`

3. **Install Kaggle API (Optional - Faster)**
   ```bash
   pip install kaggle

   # Get API credentials from https://www.kaggle.com/settings
   # Place kaggle.json in ~/.kaggle/

   kaggle competitions download -c avazu-ctr-prediction
   unzip avazu-ctr-prediction.zip -d data/raw/
   ```

### Option 2: Generate Synthetic Data (Quick Start)

If you want to start immediately without Kaggle:

```bash
cd /Users/tomersadehdev/PycharmProjects/AdClickPrediction
python scripts/generate_synthetic_data.py
```

This creates a realistic synthetic CTR dataset for development.

## Step 2: Set Up Python Environment

```bash
cd /Users/tomersadehdev/PycharmProjects/AdClickPrediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults are fine for local dev)
```

## Step 4: Verify Setup

```bash
# Test data loading
python src/data/data_loader.py

# Start Jupyter for exploration
jupyter notebook notebooks/
```

## Step 5: Start MLflow UI (Optional)

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

## Project Structure

```
AdClickPrediction/
├── data/
│   ├── raw/           # Place train.csv here
│   └── processed/     # Preprocessed data will go here
├── notebooks/         # Jupyter notebooks for EDA
├── src/
│   ├── data/         # Data loading
│   ├── features/     # Feature engineering
│   ├── models/       # Model definitions
│   ├── training/     # Training scripts
│   └── api/          # FastAPI app
├── mlruns/           # MLflow experiments (auto-created)
├── models/           # Saved models (auto-created)
└── requirements.txt
```

## Next Steps

1. **Explore Data**: Open `notebooks/01_eda.ipynb`
2. **Train Models**: Run `python src/training/train.py`
3. **Start API**: Run `uvicorn src.api.main:app --reload`

## Troubleshooting

**"Data file not found"**
- Make sure `train.csv` is in `data/raw/`
- Or use synthetic data generator

**MLflow connection error**
- MLflow will auto-create `mlruns/` directory
- If issues, check `.env` file

**PyTorch installation issues**
- Use pip install torch (CPU version)
- For GPU: Follow https://pytorch.org/get-started/locally/
