"""Data validation and leakage diagnostics. Every function returns plain dicts /
DataFrames so results can be serialised to reports/metrics/."""
import numpy as np
import pandas as pd

from src import config


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique(),
    })
    num = df.select_dtypes("number")
    rep.loc[num.columns, "min"] = num.min()
    rep.loc[num.columns, "max"] = num.max()
    rep["role"] = "numeric feature"
    rep.loc[[c for c in config.CATEGORICAL_COLUMNS + config.WIND_DIR_COLUMNS if c in df], "role"] = "categorical feature"
    rep.loc[[c for c in config.GEO_COLUMNS if c in df], "role"] = "geographic (categorical)"
    rep.loc[[c for c in config.LEAKAGE_COLUMNS if c in df], "role"] = "post-harvest (leakage)"
    rep.loc[[c for c in (config.TARGET_RAW, config.TARGET_PER_HA) if c in df], "role"] = "target"
    return rep


def invalid_value_report(df: pd.DataFrame) -> dict:
    """Physically impossible values."""
    out = {}
    num = df.select_dtypes("number")
    out["negative_values"] = int((num < 0).sum().sum())
    out["rh_out_of_0_100"] = int(((df[config.RH_COLUMNS] < 0) | (df[config.RH_COLUMNS] > 100)).sum().sum())
    out["tmin_gt_tmax"] = int(sum((df[mn] > df[mx]).sum()
                                  for mn, mx in zip(config.TMIN_COLUMNS, config.TMAX_COLUMNS)))
    out["zero_or_negative_yield"] = int((df[config.TARGET_RAW] <= 0).sum())
    out["zero_hectares"] = int((df["hectares"] <= 0).sum())
    out["unknown_wind_direction_codes"] = sorted(
        set(np.unique(df[config.WIND_DIR_COLUMNS].values)) - set(config.COMPASS_DEG))
    return out


def duplicate_report(df: pd.DataFrame, feature_cols: list[str], target: str) -> dict:
    exact = int(df.duplicated().sum())
    configs = df.groupby(feature_cols, observed=True)[target]
    n_configs = configs.ngroups
    conflicting = int((configs.nunique() > 1).sum())
    return {
        "n_rows": len(df),
        "exact_duplicate_rows": exact,
        "rows_after_dedup": len(df) - exact,
        "unique_feature_configurations": n_configs,
        "configs_with_conflicting_targets": conflicting,
    }


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def outlier_report(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes("number")
    return num.apply(lambda s: int(iqr_outliers(s).sum())).sort_values(ascending=False)


def is_determined_by(df: pd.DataFrame, col: str, key) -> bool:
    """True if `col` is a deterministic function of `key` (functional dependency)."""
    return bool((df.groupby(key, observed=True)[col].nunique() <= 1).all())


def leakage_report(df: pd.DataFrame) -> dict:
    """Structural leakage / confounding diagnostics on the raw (clean-named) data."""
    t, h = df[config.TARGET_RAW], df["hectares"]
    yph = t / h
    rep = {}

    # 1. Farm size: input quantities are hectares x constant.
    per_ha_unique = {c: int((df[c] / h).round(6).nunique()) for c in config.INPUT_QTY_COLUMNS}
    rep["input_qty_unique_values_per_hectare"] = per_ha_unique
    rep["corr_yield_kg_vs_hectares"] = float(t.corr(h))
    rep["r2_yield_kg_from_hectares_linear"] = float(t.corr(h) ** 2)

    # 2. Post-harvest variable.
    trash_ph = df["trash_bundles"] / h
    rep["corr_trash_vs_yield_kg"] = float(df["trash_bundles"].corr(t))
    rep["trash_per_ha_by_variety"] = {
        v: g.value_counts().to_dict() for v, g in trash_ph.groupby(df["variety"])}

    # 3. Weather is a function of location.
    rep["n_distinct_weather_profiles"] = int(df[config.WEATHER_COLUMNS].drop_duplicates().shape[0])
    rep["weather_cols_determined_by_agriblock"] = int(sum(
        is_determined_by(df, c, "agriblock") for c in config.WEATHER_COLUMNS))
    rep["n_weather_cols"] = len(config.WEATHER_COLUMNS)

    # 4. Rain + "AI" is constant per stage -> AI is complementary water to a fixed requirement.
    rep["rain_plus_ai_unique_per_stage"] = {
        s: sorted((df[f"rain_{s}_mm"] + df[f"ai_{s}_mm"]).round(3).unique().tolist())
        for s in config.RAIN_STAGES}

    # 5. Time: no year / season column exists.
    rep["temporal_columns_present"] = [c for c in df.columns if any(k in c.lower() for k in ("year", "season", "date"))]

    # 6. Irreducible noise: share of per-ha variance among rows with identical features.
    feat = [c for c in df.columns if c not in (config.TARGET_RAW, *config.LEAKAGE_COLUMNS)]
    within = yph - yph.groupby([df[c] for c in feat]).transform("mean")
    rep["per_ha_within_config_variance_share"] = float(within.var() / yph.var())
    rep["per_ha_r2_ceiling_estimate"] = float(1 - within.var() / yph.var())

    # 7. Correlation of each numeric with both targets.
    num = df.select_dtypes("number").drop(columns=[config.TARGET_RAW])
    rep["top_corr_with_yield_kg"] = num.corrwith(t).abs().sort_values(ascending=False).head(15).round(4).to_dict()
    rep["top_corr_with_yield_per_ha"] = num.corrwith(yph).abs().sort_values(ascending=False).head(15).round(4).to_dict()
    return rep
