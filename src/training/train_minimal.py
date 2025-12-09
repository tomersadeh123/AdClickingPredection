"""
Minimal training to demonstrate the pipeline
Creates saved models for portfolio demonstration
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib

from src.data.data_loader import CTRDataLoader
from src.features.feature_engineering import CTRFeatureEngineer

print("="*70)
print("CTR PREDICTION - TRAINING PIPELINE")
print("="*70)

# Load data
print("\n[1/3] Loading data...")
loader = CTRDataLoader()
df = loader.load_data(nrows=5000)  # Small for quick demo
train_df, val_df, test_df = loader.split_data(df)

# Feature engineering
print("\n[2/3] Feature engineering...")
fe = CTRFeatureEngineer(n_hash_features=2**12)  # 4K features

X_train, y_train = fe.fit_transform(train_df)
X_val, y_val = fe.transform(val_df), val_df['click'].values
X_test, y_test = fe.transform(test_df), test_df['click'].values

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save feature engineer
fe.save("models/feature_engineer.pkl")
print("✓ Saved feature_engineer.pkl")

# Train model
print("\n[3/3] Training Logistic Regression...")
model = LogisticRegression(max_iter=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
val_pred_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_pred_proba >= 0.5).astype(int)

metrics = {
    'auc_roc': roc_auc_score(y_val, val_pred_proba),
    'accuracy': accuracy_score(y_val, val_pred),
    'precision': precision_score(y_val, val_pred, zero_division=0),
    'recall': recall_score(y_val, val_pred, zero_division=0)
}

print(f"✓ Validation AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"✓ Validation Accuracy: {metrics['accuracy']:.4f}")

# Save model
joblib.dump(model, "models/ctr_model.pkl")
print("✓ Saved ctr_model.pkl")

# Test evaluation
test_pred_proba = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_pred_proba)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"📊 Test AUC-ROC: {test_auc:.4f}")
print(f"📁 Models saved in ./models/:")
print(f"   - feature_engineer.pkl")
print(f"   - ctr_model.pkl")
