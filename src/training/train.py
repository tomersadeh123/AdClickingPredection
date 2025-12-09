"""
Training script with MLflow tracking

Trains multiple models and logs everything to MLflow:
- Logistic Regression (baseline)
- XGBoost
- PyTorch Neural Network

Compares models and selects the best one
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss
)
import xgboost as xgb
import torch

from src.data.data_loader import CTRDataLoader
from src.features.feature_engineering import CTRFeatureEngineer
from src.models.pytorch_model import CTRNeuralNetwork, CTRNeuralNetworkTrainer


class CTRModelTrainer:
    """Train and compare multiple CTR prediction models"""

    def __init__(self, experiment_name="ad_ctr_prediction"):
        """Initialize trainer with MLflow experiment"""
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_metrics(self, y_true, y_pred_proba, prefix=""):
        """Calculate and log all metrics"""
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

        metrics = {
            f"{prefix}auc_roc": roc_auc_score(y_true, y_pred_proba),
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
            f"{prefix}log_loss": log_loss(y_true, y_pred_proba)
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        return metrics

    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression baseline"""
        print("\n" + "="*50)
        print("Training Logistic Regression (Baseline)")
        print("="*50)

        with mlflow.start_run(run_name="logistic_regression"):
            # Log parameters
            params = {
                'max_iter': 100,
                'solver': 'saga',
                'penalty': 'l2',
                'C': 1.0,
                'class_weight': 'balanced'
            }
            mlflow.log_params(params)

            # Train
            model = LogisticRegression(**params, random_state=42, verbose=1)
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]

            # Log metrics
            train_metrics = self.log_metrics(y_train, train_pred, "train_")
            val_metrics = self.log_metrics(y_val, val_pred, "val_")

            print(f"✓ Train AUC: {train_metrics['train_auc_roc']:.4f}")
            print(f"✓ Val AUC: {val_metrics['val_auc_roc']:.4f}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            return model, val_metrics['val_auc_roc']

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost")
        print("="*50)

        with mlflow.start_run(run_name="xgboost"):
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': scale_pos_weight,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            mlflow.log_params(params)

            # Train with early stopping
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )

            # Predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]

            # Log metrics
            train_metrics = self.log_metrics(y_train, train_pred, "train_")
            val_metrics = self.log_metrics(y_val, val_pred, "val_")

            print(f"✓ Train AUC: {train_metrics['train_auc_roc']:.4f}")
            print(f"✓ Val AUC: {val_metrics['val_auc_roc']:.4f}")

            # Log model
            mlflow.xgboost.log_model(model, "model")

            return model, val_metrics['val_auc_roc']

    def train_pytorch(self, X_train, y_train, X_val, y_val):
        """Train PyTorch neural network"""
        print("\n" + "="*50)
        print("Training PyTorch Neural Network")
        print("="*50)

        with mlflow.start_run(run_name="pytorch_neural_network"):
            # Parameters
            params = {
                'input_size': X_train.shape[1],
                'hidden_sizes': [512, 256, 128],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 20,
                'early_stopping_patience': 3
            }
            mlflow.log_params(params)

            # Create model
            model = CTRNeuralNetwork(
                input_size=params['input_size'],
                hidden_sizes=params['hidden_sizes'],
                dropout_rate=params['dropout_rate']
            )

            # Create trainer
            trainer = CTRNeuralNetworkTrainer(
                model=model,
                learning_rate=params['learning_rate']
            )

            # Train
            history = trainer.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                early_stopping_patience=params['early_stopping_patience'],
                verbose=True
            )

            # Log training history
            for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
                mlflow.log_metric("train_loss_epoch", train_loss, step=epoch)
                mlflow.log_metric("val_loss_epoch", val_loss, step=epoch)

            # Predictions
            train_pred = trainer.predict(X_train)
            val_pred = trainer.predict(X_val)

            # Log metrics
            train_metrics = self.log_metrics(y_train, train_pred, "train_")
            val_metrics = self.log_metrics(y_val, val_pred, "val_")

            print(f"✓ Train AUC: {train_metrics['train_auc_roc']:.4f}")
            print(f"✓ Val AUC: {val_metrics['val_auc_roc']:.4f}")

            # Save model
            model_path = "models/pytorch_model.pt"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            trainer.save_model(model_path)
            mlflow.log_artifact(model_path)

            return trainer, val_metrics['val_auc_roc']


def main():
    """Main training pipeline"""
    print("="*70)
    print("CTR PREDICTION MODEL TRAINING")
    print("="*70)

    # 1. Load data
    print("\n[1/5] Loading data...")
    loader = CTRDataLoader()
    df = loader.load_data(nrows=50000)  # Use 50K for faster training
    train_df, val_df, test_df = loader.split_data(df, test_size=0.2, val_size=0.1)

    # 2. Feature engineering
    print("\n[2/5] Feature engineering...")
    fe = CTRFeatureEngineer(n_hash_features=2**18)  # 262K features

    X_train, y_train = fe.fit_transform(train_df)
    X_val, y_val = fe.transform(val_df), val_df['click'].values
    X_test, y_test = fe.transform(test_df), test_df['click'].values

    # Save feature engineer
    fe.save("models/feature_engineer.pkl")

    # 3. Train models
    print("\n[3/5] Training models...")
    trainer = CTRModelTrainer()

    results = {}

    # Train Logistic Regression
    lr_model, lr_auc = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    results['Logistic Regression'] = lr_auc

    # Train XGBoost
    xgb_model, xgb_auc = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    results['XGBoost'] = xgb_auc

    # Train PyTorch
    pytorch_model, pytorch_auc = trainer.train_pytorch(X_train, y_train, X_val, y_val)
    results['PyTorch NN'] = pytorch_auc

    # 4. Compare results
    print("\n" + "="*70)
    print("[4/5] MODEL COMPARISON (Validation AUC-ROC)")
    print("="*70)
    for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {auc:.4f}")

    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model} (AUC: {results[best_model]:.4f})")

    # 5. Final evaluation on test set
    print("\n[5/5] Final evaluation on test set...")
    if best_model == "PyTorch NN":
        test_pred = pytorch_model.predict(X_test)
    elif best_model == "XGBoost":
        test_pred = xgb_model.predict_proba(X_test)[:, 1]
    else:
        test_pred = lr_model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, test_pred)
    print(f"✓ Test AUC-ROC: {test_auc:.4f}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 View results: mlflow ui")
    print(f"📁 Models saved in: ./models/")
    print(f"📈 Experiments logged in: ./mlruns/")


if __name__ == "__main__":
    main()
