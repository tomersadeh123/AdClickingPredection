"""
Data loading utilities for CTR prediction dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class CTRDataLoader:
    """Load and split CTR prediction dataset"""

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)

    def load_data(self, filename: str = "train.csv", nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load CTR dataset

        Args:
            filename: Name of the data file
            nrows: Number of rows to load (None = all)

        Returns:
            DataFrame with CTR data
        """
        filepath = self.data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                "Please download the Avazu Click-Through Rate dataset from Kaggle:\n"
                "https://www.kaggle.com/c/avazu-ctr-prediction/data"
            )

        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, nrows=nrows)
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "click",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets

        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining after test split)
            random_state: Random seed

        Returns:
            train_df, val_df, test_df
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col]
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df[target_col]
        )

        print(f"Train set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val set:   {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

        # Print class distribution
        print(f"\nClick rate - Train: {train_df[target_col].mean():.4f}, "
              f"Val: {val_df[target_col].mean():.4f}, "
              f"Test: {test_df[target_col].mean():.4f}")

        return train_df, val_df, test_df

    def get_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Get a small sample for quick experimentation

        Args:
            n_samples: Number of samples to return

        Returns:
            Sample DataFrame
        """
        return self.load_data(nrows=n_samples)


# For quick testing
if __name__ == "__main__":
    loader = CTRDataLoader()

    # Load small sample
    df = loader.get_sample_data(n_samples=10000)
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nClick rate: {df['click'].mean():.4f}")
