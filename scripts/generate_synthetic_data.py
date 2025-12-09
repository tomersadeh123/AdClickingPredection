"""
Generate synthetic CTR data for quick development
This creates a realistic fake dataset that mimics the Avazu CTR structure
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_ctr_data(n_samples=100000, output_path="data/raw/train.csv"):
    """Generate synthetic ad click data"""

    print(f"Generating {n_samples:,} synthetic ad impressions...")

    np.random.seed(42)

    # Realistic CTR: ~17% (typical for display ads)
    click_rate = 0.17

    # Generate features
    data = {
        'id': range(n_samples),

        # Target
        'click': np.random.binomial(1, click_rate, n_samples),

        # Time features (YYMMDDHH format)
        'hour': [
            f"{np.random.randint(14, 15):02d}"  # Year 2014
            f"{np.random.randint(10, 11):02d}"  # October
            f"{np.random.randint(21, 31):02d}"  # Days 21-30
            f"{np.random.randint(0, 24):02d}"   # Hours 0-23
            for _ in range(n_samples)
        ],

        # Categorical features
        'C1': np.random.randint(1000, 1010, n_samples),
        'banner_pos': np.random.randint(0, 8, n_samples),
        'site_id': np.random.choice([f'site_{i}' for i in range(5000)], n_samples),
        'site_domain': np.random.choice([f'domain_{i}.com' for i in range(100)], n_samples),
        'site_category': np.random.choice([f'cat_{i}' for i in range(50)], n_samples),
        'app_id': np.random.choice([f'app_{i}' for i in range(8000)], n_samples),
        'app_domain': np.random.choice([f'appdomain_{i}.com' for i in range(200)], n_samples),
        'app_category': np.random.choice([f'appcat_{i}' for i in range(40)], n_samples),

        # Device features
        'device_id': np.random.choice([f'device_{i}' for i in range(10000)], n_samples),
        'device_ip': np.random.choice([f'ip_{i}' for i in range(20000)], n_samples),
        'device_model': np.random.choice([f'model_{i}' for i in range(500)], n_samples),
        'device_type': np.random.randint(0, 5, n_samples),
        'device_conn_type': np.random.randint(0, 5, n_samples),

        # Additional categorical features
        'C14': np.random.randint(10000, 20000, n_samples),
        'C15': np.random.randint(300, 350, n_samples),
        'C16': np.random.randint(50, 150, n_samples),
        'C17': np.random.randint(1000, 3000, n_samples),
        'C18': np.random.randint(0, 5, n_samples),
        'C19': np.random.randint(30, 200, n_samples),
        'C20': np.random.randint(-1, 1000, n_samples),
        'C21': np.random.randint(10, 100, n_samples),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some correlation between features and clicks for realism
    # Higher banner positions get more clicks
    df.loc[df['banner_pos'] == 0, 'click'] = np.random.binomial(1, 0.25, (df['banner_pos'] == 0).sum())
    df.loc[df['banner_pos'] > 5, 'click'] = np.random.binomial(1, 0.10, (df['banner_pos'] > 5).sum())

    # Certain device types perform better
    df.loc[df['device_type'] == 1, 'click'] = np.random.binomial(1, 0.22, (df['device_type'] == 1).sum())

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(df):,} rows to {output_file}")
    print(f"✓ Click rate: {df['click'].mean():.2%}")
    print(f"✓ Features: {len(df.columns)}")
    print(f"✓ File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return df


if __name__ == "__main__":
    # Generate 100K samples (fast for development)
    # For production-like testing, use n_samples=1000000 (1M)
    df = generate_synthetic_ctr_data(n_samples=100000)

    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
