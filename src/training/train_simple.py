"""
Simplified training script that actually completes
Trains models on smaller dataset to demonstrate the pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import joblib

from src.data.data_loader import CTRDataLoader
from src.features.feature_engineering import CTRFeatureEngineer

print("="*70)
print("CTR PREDICTION - QUICK TRAINING DEMO")
print("="*70)

# 1. Load smaller dataset
print("\n[1/4] Loading data...")
loader = CTRDataLoader()
df = loader.load_data(nrows=10000)  # Small dataset for demo
train_df, val_df, test_df = loader.split_data(df, test_size=0.2, val_size=0.1)

# 2. Feature engineering (smaller hash size)
print("\n[2/4] Feature engineering...")
fe = CTRFeatureEngineer(n_hash_features=2**14)  # 16K features instead of 262K

X_train, y_train = fe.fit_transform(train_df)
X_val, y_val = fe.transform(val_df), val_df['click'].values
X_test, y_test = fe.transform(test_df), test_df['click'].values

# Save feature engineer
Path("models").mkdir(exist_ok=True)
fe.save("models/feature_engineer.pkl")
print("✓ Saved feature engineer")

# 3. Train models
print("\n[3/4] Training models...")

results = {}

# Logistic Regression
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=50, solver='saga', C=1.0, random_state=42)
lr.fit(X_train, y_train)

lr_val_pred = lr.predict_proba(X_val)[:, 1]
lr_auc = roc_auc_score(y_val, lr_val_pred)
results['Logistic Regression'] = lr_auc
print(f"✓ Val AUC: {lr_auc:.4f}")

# Save model
joblib.dump(lr, "models/logistic_regression.pkl")

# XGBoost
print("\n--- XGBoost ---")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=50,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_pred)
results['XGBoost'] = xgb_auc
print(f"✓ Val AUC: {xgb_auc:.4f}")

# Save model
joblib.dump(xgb_model, "models/xgboost_model.pkl")

# 4. Results
print("\n" + "="*70)
print("[4/4] RESULTS")
print("="*70)

for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:25s}: {auc:.4f}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} (AUC: {results[best_model]:.4f})")

# Test set evaluation
print("\n--- Test Set Evaluation ---")
if best_model == "XGBoost":
    test_pred = xgb_model.predict_proba(X_test)[:, 1]
else:
    test_pred = lr.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, test_pred)
test_acc = accuracy_score(y_test, (test_pred >= 0.5).astype(int))
print(f"✓ Test AUC-ROC: {test_auc:.4f}")
print(f"✓ Test Accuracy: {test_acc:.4f}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"📁 Models saved in: ./models/")
print(f"   - feature_engineer.pkl")
print(f"   - logistic_regression.pkl")
print(f"   - xgboost_model.pkl")
print("\nNote: This is a demo with 10K samples. For production, use full dataset.")
