"""
Feature engineering for CTR prediction

Handles:
- Time-based features from 'hour' field
- Hash encoding for high-cardinality categorical features
- Feature scaling for numerical features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from typing import Tuple, List
import joblib
from pathlib import Path


class CTRFeatureEngineer:
    """Feature engineering pipeline for CTR data"""

    def __init__(self, n_hash_features: int = 2**18):
        """
        Initialize feature engineer

        Args:
            n_hash_features: Number of features for hash encoding (default: 262,144)
        """
        self.n_hash_features = n_hash_features
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        self.time_cols = []

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from 'hour' column

        Hour format: YYMMDDHH
        """
        df = df.copy()

        # Convert hour to string if not already
        df['hour'] = df['hour'].astype(str)

        # Extract components
        df['hour_of_day'] = df['hour'].str[-2:].astype(int)
        df['day_of_month'] = df['hour'].str[-4:-2].astype(int)
        df['month'] = df['hour'].str[-6:-4].astype(int)
        df['year'] = df['hour'].str[:-6].astype(int)

        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Day of week approximation (assuming some starting date)
        # This is simplified - in production, convert to actual date
        df['day_of_week'] = (df['day_of_month'] % 7)

        # Is weekend (simplified)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time period buckets
        df['time_period'] = pd.cut(
            df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )

        return df

    def identify_column_types(self, df: pd.DataFrame, target_col: str = 'click'):
        """Identify categorical, numerical, and time columns"""

        # Time features we created
        self.time_cols = [
            'hour_of_day', 'day_of_month', 'month', 'year',
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
        ]

        # Categorical columns (object dtype + low cardinality int)
        self.categorical_cols = [
            col for col in df.columns
            if df[col].dtype == 'object' and col != target_col
        ]

        # Numerical columns (int/float, not target or id)
        self.numerical_cols = [
            col for col in df.columns
            if df[col].dtype in ['int64', 'float64']
            and col not in [target_col, 'id', 'hour']
            and col not in self.time_cols
        ]

        print(f"Categorical features ({len(self.categorical_cols)}): {self.categorical_cols[:5]}...")
        print(f"Numerical features ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"Time features ({len(self.time_cols)}): {self.time_cols}")

    def hash_encode_categoricals(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> np.ndarray:
        """
        Hash encoding for high-cardinality categorical features

        This is more memory-efficient than one-hot encoding for features
        with millions of unique values (like device_id, site_id)
        """
        hasher = FeatureHasher(
            n_features=self.n_hash_features,
            input_type='dict'
        )

        # Convert to list of dicts format for FeatureHasher
        records = df[categorical_cols].astype(str).to_dict('records')

        # Hash encode
        hashed_features = hasher.transform(records).toarray()

        return hashed_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'click'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform training data

        Returns:
            X: Feature matrix
            y: Target vector
        """
        print("Starting feature engineering...")

        # Extract time features
        df = self.extract_time_features(df)

        # Identify column types
        self.identify_column_types(df, target_col)

        # Get target
        y = df[target_col].values

        # Process numerical features
        numerical_features = df[self.numerical_cols + self.time_cols].values
        numerical_features = self.scaler.fit_transform(numerical_features)

        # Hash encode categorical features
        categorical_features = self.hash_encode_categoricals(df, self.categorical_cols)

        # Combine all features
        X = np.hstack([numerical_features, categorical_features])

        print(f"✓ Final feature matrix shape: {X.shape}")
        print(f"✓ Target distribution - Click: {y.mean():.4f}, No-click: {1-y.mean():.4f}")

        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted parameters"""

        # Extract time features
        df = self.extract_time_features(df)

        # Process numerical features
        numerical_features = df[self.numerical_cols + self.time_cols].values
        numerical_features = self.scaler.transform(numerical_features)

        # Hash encode categorical features
        categorical_features = self.hash_encode_categoricals(df, self.categorical_cols)

        # Combine
        X = np.hstack([numerical_features, categorical_features])

        return X

    def save(self, path: str):
        """Save feature engineer to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        print(f"✓ Saved feature engineer to {save_path}")

    @staticmethod
    def load(path: str) -> 'CTRFeatureEngineer':
        """Load feature engineer from disk"""
        return joblib.load(path)


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data.data_loader import CTRDataLoader

    # Load sample data
    loader = CTRDataLoader()
    df = loader.get_sample_data(n_samples=5000)

    # Create feature engineer
    fe = CTRFeatureEngineer(n_hash_features=2**16)  # Smaller for testing

    # Fit and transform
    X, y = fe.fit_transform(df)

    print(f"\n✓ Feature engineering complete!")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    print(f"✓ Click rate: {y.mean():.4f}")
