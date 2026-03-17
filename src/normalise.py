import pandas as pd
import numpy as np

def remove_outliers_zscore(df, threshold=3):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())

    # Keep rows where ALL z-scores < threshold
    filtered_df = df[(z_scores < threshold).sum(axis=1) > (len(numeric_cols) * 0.9)]

    print(f"Original shape: {df.shape}")
    print(f"After outlier removal: {filtered_df.shape}")

    return filtered_df