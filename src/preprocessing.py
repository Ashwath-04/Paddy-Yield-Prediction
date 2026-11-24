"""Cleaning, target construction, splitting and categorical encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicate rows (features AND target identical) would otherwise land on
    both sides of a split and inflate test scores."""
    return df.drop_duplicates().reset_index(drop=True)


def add_per_ha_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.TARGET_PER_HA] = df[config.TARGET_RAW] / df["hectares"]
    return df


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Compass labels -> (sin, cos) so N and NNW are close, unlike one-hot/ordinal."""
    df = df.copy()
    for c in config.WIND_DIR_COLUMNS:
        rad = np.deg2rad(df[c].map(config.COMPASS_DEG))
        if rad.isna().any():
            raise ValueError(f"Unknown compass code in {c}")
        df[f"{c}_sin"] = np.sin(rad).round(6)
        df[f"{c}_cos"] = np.cos(rad).round(6)
    return df.drop(columns=config.WIND_DIR_COLUMNS)


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Hold-out split stratified on farm size x location so both sides cover every
    (hectares, agriblock) cell. No temporal column exists, so a time split is impossible."""
    strata = df["hectares"].astype(str) + "_" + df["agriblock"]
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)


class CategoricalEncoder:
    """One-hot encoder fitted on training categories only (binary columns -> one dummy)."""

    def __init__(self, columns=config.CATEGORICAL_COLUMNS):
        self.columns = list(columns)
        self.categories_: dict[str, list] = {}

    def fit(self, df: pd.DataFrame):
        self.categories_ = {c: sorted(df[c].unique()) for c in self.columns}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=self.columns).copy()
        for c in self.columns:
            cats = self.categories_[c]
            keep = cats[1:] if len(cats) == 2 else cats
            for v in keep:
                out[f"{c}={v}"] = (df[c] == v).astype(np.int8)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
