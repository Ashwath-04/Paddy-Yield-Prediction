"""Agronomic feature engineering and feature selection.

Every derived feature is documented in FEATURE_DOCS (also rendered in the README).
Fertilizer quantities are assumed to be in kg (the source gives no unit for
DAP/Urea/Potash); nutrient contents use standard fertilizer grades:
Urea 46% N, DAP 18-46-0, Muriate of Potash 60% K2O.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src import config

FEATURE_DOCS = {
    "n_applied_kg": "Nitrogen applied = 0.46*Urea + 0.18*DAP (kg)",
    "n_per_ha": "Nitrogen intensity, kg N / ha",
    "p2o5_per_ha": "Phosphate intensity = 0.46*DAP / ha (kg P2O5/ha)",
    "k2o_per_ha": "Potash intensity = 0.60*Potash / ha (kg K2O/ha)",
    "n_to_k_ratio": "N : K2O balance (nutrient ratio)",
    "total_rain_mm": "Season rainfall, sum of the 4 crop-stage windows (D0-105)",
    "total_irrigation_mm": "Season 'AI' water (source abbreviation; complements rain to a fixed stage requirement)",
    "rain_share": "Fraction of season water supplied by rain = rain / (rain + AI)",
    "tmean_season": "Mean of (Tmin+Tmax)/2 over the 4 thirty-day windows (deg C)",
    "diurnal_range_mean": "Mean (Tmax - Tmin) over the season (deg C)",
    "gdd_base10": "Growing degree days, base 10 C, 4 x 30-day windows",
    "heat_stress_windows": "Number of 30-day windows with Tmax >= 35 C (spikelet sterility risk)",
    "rh_mean": "Season mean relative humidity (%)",
    "wind_mean_kn": "Season mean instantaneous wind speed (knots)",
    "rain_x_tmean": "Rainfall-temperature interaction = total_rain_mm * tmean_season",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h = df["hectares"]
    df["n_applied_kg"] = 0.46 * df["urea_40d"] + 0.18 * df["dap_20d"]
    df["n_per_ha"] = df["n_applied_kg"] / h
    df["p2o5_per_ha"] = 0.46 * df["dap_20d"] / h
    df["k2o_per_ha"] = 0.60 * df["potash_50d"] / h
    df["n_to_k_ratio"] = df["n_per_ha"] / df["k2o_per_ha"]

    df["total_rain_mm"] = df[config.RAIN_COLUMNS].sum(axis=1)
    df["total_irrigation_mm"] = df[config.AI_COLUMNS].sum(axis=1)
    df["rain_share"] = df["total_rain_mm"] / (df["total_rain_mm"] + df["total_irrigation_mm"])

    tmin, tmax = df[config.TMIN_COLUMNS].to_numpy(), df[config.TMAX_COLUMNS].to_numpy()
    tavg = (tmin + tmax) / 2
    df["tmean_season"] = tavg.mean(axis=1)
    df["diurnal_range_mean"] = (tmax - tmin).mean(axis=1)
    df["gdd_base10"] = (np.clip(tavg - 10, 0, None) * 30).sum(axis=1)
    df["heat_stress_windows"] = (tmax >= 35).sum(axis=1)

    df["rh_mean"] = df[config.RH_COLUMNS].mean(axis=1)
    df["wind_mean_kn"] = df[config.WIND_SPEED_COLUMNS].mean(axis=1)
    df["rain_x_tmean"] = df["total_rain_mm"] * df["tmean_season"]
    # Round away float noise (e.g. 162.78/6) so truly constant ratios are detected as constant.
    df[list(FEATURE_DOCS)] = df[list(FEATURE_DOCS)].round(6)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except targets and post-harvest leakage columns."""
    drop = {config.TARGET_RAW, config.TARGET_PER_HA, *config.LEAKAGE_COLUMNS}
    return [c for c in df.columns if c not in drop]


# --------------------------------------------------------------------------- selection
def redundancy_filter(X: pd.DataFrame, threshold: float = 0.99):
    """Drop constant columns, then greedily drop any column whose |Pearson r| with an
    already-kept column is >= threshold. Column order sets priority (raw first).
    Returns (kept, dropped_info)."""
    dropped = {}
    nunique = X.nunique()
    for c in nunique[nunique <= 1].index:
        dropped[c] = "constant"
    cand = [c for c in X.columns if c not in dropped]
    corr = X[cand].corr().abs()
    kept = []
    for c in cand:
        hit = next((k for k in kept if corr.loc[c, k] >= threshold), None)
        if hit is None:
            kept.append(c)
        else:
            dropped[c] = f"|r|>={threshold} with {hit} (r={corr.loc[c, hit]:.4f})"
    return kept, dropped


def mutual_information(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED) -> pd.Series:
    mi = mutual_info_regression(X, y, random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def model_importance(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED) -> pd.Series:
    """Gain importance from a default LightGBM (normalised to sum 1)."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1)
    m.fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return (imp / imp.sum()).sort_values(ascending=False)
