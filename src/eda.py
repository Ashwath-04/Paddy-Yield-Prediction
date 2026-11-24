"""Exploratory figures. Correlation plots are descriptive only - no causal claims."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.plotting import BLUE, CATEGORICAL, DIVERGING, INK_2, ORANGE, save

TPH = config.TARGET_PER_HA
TKG = config.TARGET_RAW


def _strip(ax, x, y, color=BLUE, label_means=True):
    """Jittered strip for a discrete x with a mean marker - avoids overplotting."""
    rng = np.random.default_rng(0)
    xs = np.asarray(x, float)
    levels = np.unique(xs)
    span = np.diff(levels).min() if len(levels) > 1 else 1.0
    ax.scatter(xs + rng.uniform(-0.18, 0.18, len(xs)) * span, y, s=6, alpha=0.25, color=color, lw=0)
    means = pd.Series(np.asarray(y)).groupby(xs).mean()
    ax.plot(means.index, means.values, "o-", color=INK_2, ms=5, lw=1.2, label="group mean")
    if label_means:
        for xv, mv in means.items():
            ax.annotate(f"{mv:,.0f}", (xv, mv), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK_2)


def target_distribution(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df[TKG], bins=50, color=BLUE)
    axes[0].set(title="Total yield (kg) — multimodal: one mode per farm size", xlabel="yield_kg", ylabel="rows")
    for i, (h, g) in enumerate(df.groupby("hectares")):
        axes[1].hist(g[TKG], bins=20, color=CATEGORICAL[i % 8], alpha=0.85, label=f"{h} ha")
    axes[1].set(title="Total yield (kg) by hectares", xlabel="yield_kg")
    axes[1].legend(ncols=2, fontsize=8)
    axes[2].hist(df[TPH], bins=40, color=BLUE)
    axes[2].axvline(df[TPH].mean(), color=ORANGE, lw=1.5, label=f"mean {df[TPH].mean():,.0f}")
    axes[2].set(title="Yield per hectare (kg/ha)", xlabel="yield_kg_per_ha")
    axes[2].legend()
    fig.tight_layout(); save(fig, path)


def feature_distributions(df, path):
    num = df.select_dtypes("number").drop(columns=[TKG, TPH], errors="ignore")
    cols = list(num.columns)
    n = len(cols); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 2.3 * nrow))
    for ax, c in zip(axes.flat, cols):
        vc = num[c].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=BLUE, width=0.7)
        ax.set_title(f"{c}  (n_unique={len(vc)})", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=6.5, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions — every feature takes only 3–36 distinct values", y=1.0)
    fig.tight_layout(); save(fig, path)


def correlation_matrix(df, path):
    num = df.select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    ax.set_title("Correlation matrix (numeric columns + both targets)")
    fig.tight_layout(); save(fig, path)
    return corr


def yield_vs(df, xcol, xlabel, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _strip(axes[0], df[xcol], df[TPH])
    axes[0].set(xlabel=xlabel, ylabel="yield (kg/ha)", title=f"Yield per ha vs {title}")
    sc = axes[1].scatter(df[xcol], df[TKG], c=df["hectares"], cmap="viridis", s=8, alpha=0.5)
    fig.colorbar(sc, ax=axes[1], label="hectares")
    axes[1].set(xlabel=xlabel, ylabel="yield (kg)", title=f"Total yield vs {title} (coloured by farm size)")
    r1, r2 = df[xcol].corr(df[TPH]), df[xcol].corr(df[TKG])
    fig.suptitle(f"Pearson r: per-ha {r1:+.3f} | total {r2:+.3f}  (association, not causation)", fontsize=9, color=INK_2)
    fig.tight_layout(); save(fig, path)


def categorical_effects(df, path):
    cats = ["hectares", "agriblock", "variety", "soil_type", "nursery_type"]
    fig, axes = plt.subplots(1, len(cats), figsize=(20, 4), sharey=True)
    for ax, c in zip(axes, cats):
        groups = sorted(df[c].unique())
        data = [df.loc[df[c] == g, TPH] for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups], widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor="#d6e6f8", edgecolor=BLUE),
                   medianprops=dict(color=ORANGE, lw=1.5), flierprops=dict(markersize=3))
        ax.set_title(f"yield/ha by {c}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("kg/ha")
    fig.tight_layout(); save(fig, path)


def outlier_plot(df, path):
    from src.validation import iqr_outliers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    out_global = iqr_outliers(df[TPH])
    axes[0].boxplot(df[TPH], vert=False, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="#d6e6f8", edgecolor=BLUE), medianprops=dict(color=ORANGE))
    axes[0].set(title=f"yield/ha — IQR outliers (global): {int(out_global.sum())}", xlabel="kg/ha")
    resid_kg = df[TKG] - df["hectares"] * df[TPH].mean()
    flag = iqr_outliers(df[TPH] - df.groupby("hectares")[TPH].transform("median"))
    axes[1].scatter(df["hectares"], resid_kg, s=6, alpha=0.3, color=np.where(flag, ORANGE, BLUE))
    axes[1].set(title=f"Total-yield deviation from area x mean yield\norange = IQR outlier within farm-size group ({int(flag.sum())})",
                xlabel="hectares", ylabel="kg")
    fig.tight_layout(); save(fig, path)


def weather_by_block(df, path):
    """Shows that the 28 weather columns collapse to one profile per agriblock."""
    w = [c for c in config.WEATHER_COLUMNS if c not in config.WIND_DIR_COLUMNS]
    prof = df.groupby("agriblock")[w].first()
    z = (prof - prof.mean()) / prof.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(z))); ax.set_yticklabels(z.index)
    ax.set_xticks(range(len(w))); ax.set_xticklabels(w, rotation=90, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="z-score across blocks")
    ax.set_title("Weather profile per agriblock — each block has exactly one weather vector (weather ≡ location)")
    fig.tight_layout(); save(fig, path)


def run_all(df: pd.DataFrame, fig_dir=config.FIG_DIR):
    """df: deduplicated data with per-ha target and engineered features."""
    target_distribution(df, fig_dir / "eda_target_distribution.png")
    raw_num = [c for c in config.COLUMN_MAP.values() if c in df and pd.api.types.is_numeric_dtype(df[c])]
    feature_distributions(df[raw_num], fig_dir / "eda_feature_distributions.png")
    corr = correlation_matrix(df[raw_num + [TPH]], fig_dir / "eda_correlation_matrix.png")
    yield_vs(df, "total_rain_mm", "season rainfall (mm)", fig_dir / "eda_yield_vs_rainfall.png", "rainfall")
    yield_vs(df, "tmean_season", "season mean temperature (°C)", fig_dir / "eda_yield_vs_temperature.png", "temperature")
    yield_vs(df, "n_applied_kg", "nitrogen applied (kg N)", fig_dir / "eda_yield_vs_nitrogen.png", "nitrogen")
    yield_vs(df, "n_per_ha", "nitrogen intensity (kg N/ha)", fig_dir / "eda_yield_vs_nitrogen_per_ha.png", "N intensity")
    yield_vs(df, "rh_mean", "season mean relative humidity (%)", fig_dir / "eda_yield_vs_humidity.png", "humidity")
    yield_vs(df, "hectares", "hectares", fig_dir / "eda_yield_vs_hectares.png", "farm size")
    categorical_effects(df, fig_dir / "eda_categorical_effects.png")
    outlier_plot(df, fig_dir / "eda_outliers.png")
    weather_by_block(df, fig_dir / "eda_weather_by_block.png")
    return corr
