"""
PyTorch Neural Network for CTR Prediction

A feed-forward neural network with:
- Multiple hidden layers
- Batch normalization
- Dropout for regularization
- Binary classification output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CTRNeuralNetwork(nn.Module):
    """
    Deep neural network for CTR prediction

    Architecture:
    - Input layer (dynamic based on feature size)
    - Hidden layers with BatchNorm and Dropout
    - Output layer (sigmoid for binary classification)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize neural network

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(CTRNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)

    def predict_proba(self, x):
        """Predict probabilities (for sklearn compatibility)"""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            probs = self.forward(x).numpy()
        return probs

    def get_config(self):
        """Get model configuration for logging"""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


class CTRNeuralNetworkTrainer:
    """Trainer for PyTorch CTR model"""

    def __init__(
        self,
        model: CTRNeuralNetwork,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.model = model

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"✓ Training on device: {self.device}")

    def train_epoch(self, X_train, y_train, batch_size=256):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        # Create batches
        n_samples = len(X_train)
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            # Get batch
            batch_indices = indices[i:i + batch_size]
            X_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)
            y_batch = torch.FloatTensor(y_train[batch_indices]).unsqueeze(1).to(self.device)

            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, X_val, y_val, batch_size=256):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            n_samples = len(X_val)
            for i in range(0, n_samples, batch_size):
                X_batch = torch.FloatTensor(X_val[i:i + batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y_val[i:i + batch_size]).unsqueeze(1).to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=256,
        early_stopping_patience=3,
        verbose=True
    ):
        """
        Train model with early stopping

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Print progress
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)

            # Validate
            val_loss = self.evaluate(X_val, y_val, batch_size)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)

        return history

    def predict(self, X, batch_size=256):
        """Predict probabilities"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            n_samples = len(X)
            for i in range(0, n_samples, batch_size):
                X_batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                preds = self.model(X_batch).cpu().numpy()
                predictions.append(preds)

        return np.concatenate(predictions)

    def save_model(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'device': str(self.device)
        }, path)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load_model(path, device=None):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location='cpu')

        # Recreate model
        config = checkpoint['model_config']
        model = CTRNeuralNetwork(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create trainer
        trainer = CTRNeuralNetworkTrainer(model, device=device)

        return trainer


# Quick test
if __name__ == "__main__":
    import numpy as np

    # Test model creation
    model = CTRNeuralNetwork(
        input_size=65556,  # From feature engineering
        hidden_sizes=[512, 256, 128],
        dropout_rate=0.3
    )

    print("✓ PyTorch Model Created")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Model config: {model.get_config()}")

    # Test forward pass
    dummy_input = torch.randn(32, 65556)  # Batch of 32
    output = model(dummy_input)
    print(f"✓ Output shape: {output.shape}")  # Should be (32, 1)
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")  # Should be [0, 1]
