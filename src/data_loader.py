"""Download (if needed) and load the UCI Paddy Dataset with clean column names."""
import io
import urllib.request
import zipfile

import pandas as pd

from src import config


def download_raw(force: bool = False) -> None:
    """Fetch the official UCI zip (id 1186) into data/raw/."""
    if config.RAW_FILE.exists() and not force:
        return
    with urllib.request.urlopen(config.DATASET_URL, timeout=60) as resp:
        payload = resp.read()
    (config.DATA_RAW / "paddy_dataset.zip").write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(config.DATA_RAW)


def load_raw() -> pd.DataFrame:
    """Raw CSV exactly as published (header whitespace stripped)."""
    download_raw()
    df = pd.read_csv(config.RAW_FILE)
    df.columns = df.columns.str.strip()
    return df


def load_clean() -> pd.DataFrame:
    """Raw data with snake_case columns and string categoricals stripped."""
    df = load_raw()
    missing = set(config.COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    df = df.rename(columns=config.COLUMN_MAP)
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip()
    return df
