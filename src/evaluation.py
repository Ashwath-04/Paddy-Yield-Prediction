"""Metrics, cross-validation, diagnostic plots and error analysis."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from src import config
from src.plotting import BLUE, INK_2, ORANGE, save


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    # MAPE is well-defined here: yield is strictly positive (min 5,410 kg/ha).
    out["MAPE_%"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if (y_true > 0).all() else np.nan
    return out


def cv_evaluate(model, X, y, cv=None, groups=None) -> dict:
    """Per-fold metrics -> mean and std. Fold metrics (not pooled) so the std is meaningful."""
    cv = cv or KFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    rows = []
    for tr, va in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        rows.append(regression_metrics(y.iloc[va], m.predict(X.iloc[va])))
    f = pd.DataFrame(rows)
    return {**{f"cv_{k}": v for k, v in f.mean().items()},
            **{f"cv_{k}_std": v for k, v in f.std().items()}}


def oof_predictions(model, X, y, seed=config.SEED) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=KFold(config.CV_FOLDS, shuffle=True, random_state=seed))


# ------------------------------------------------------------------------ plots
def plot_pred_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color=BLUE)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color=INK_2, lw=1)
    m = regression_metrics(y_true, y_pred)
    ax.set(xlabel="Actual", ylabel="Predicted",
           title=f"{title}\nR²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")
    fig.tight_layout(); save(fig, path)


def plot_residuals(y_true, y_pred, title, path):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, res, s=10, alpha=0.5, color=BLUE)
    axes[0].axhline(0, color=INK_2, lw=1, ls="--")
    axes[0].set(xlabel="Predicted", ylabel="Residual (actual - predicted)", title="Residuals vs predicted")
    axes[1].hist(res, bins=40, color=BLUE, alpha=0.8)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    fig.suptitle(title); fig.tight_layout(); save(fig, path)


# --------------------------------------------------------------- error analysis
def error_analysis(meta: pd.DataFrame, y_true, y_pred, target_name: str, fig_path=None) -> dict:
    """meta must contain hectares/agriblock/variety/soil_type/nursery_type for slicing."""
    d = meta.copy()
    d["actual"], d["predicted"] = np.asarray(y_true), np.asarray(y_pred)
    d["residual"] = d["actual"] - d["predicted"]
    d["abs_error"] = d["residual"].abs()
    d["ape_%"] = d["abs_error"] / d["actual"] * 100
    out = {
        "n": len(d),
        "underpredicted_n": int((d["residual"] > 0).sum()),
        "overpredicted_n": int((d["residual"] < 0).sum()),
        "mean_underprediction": float(d.loc[d.residual > 0, "residual"].mean()),
        "mean_overprediction": float(-d.loc[d.residual < 0, "residual"].mean()),
        "abs_error_p50": float(d["abs_error"].median()),
        "abs_error_p90": float(d["abs_error"].quantile(0.9)),
    }
    slices = {}
    for c in ("agriblock", "variety", "hectares", "soil_type", "nursery_type"):
        g = d.groupby(c).agg(n=("abs_error", "size"), MAE=("abs_error", "mean"),
                             bias=("residual", "mean"), mean_actual=("actual", "mean"))
        slices[c] = g.round(2)
    out["slices"] = slices
    cols = ["hectares", "agriblock", "variety", "soil_type", "nursery_type", "actual", "predicted", "residual", "ape_%"]
    out["top_errors"] = d.sort_values("abs_error", ascending=False).head(15)[cols].round(1)
    out["lowest_errors"] = d.sort_values("abs_error").head(15)[cols].round(1)
    # Characterise high-error vs low-error deciles.
    hi = d[d.abs_error >= d.abs_error.quantile(0.9)]
    lo = d[d.abs_error <= d.abs_error.quantile(0.1)]
    out["high_vs_low_error_profile"] = pd.DataFrame({
        "high_error_decile": hi[["hectares", "actual"]].mean(),
        "low_error_decile": lo[["hectares", "actual"]].mean(),
    }).round(1)

    if fig_path is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex="col")
        for j, c in enumerate(("agriblock", "hectares", "variety")):
            s = slices[c]
            x = s.index.astype(str)
            axes[0, j].bar(x, s["MAE"], color=BLUE, width=0.6)
            axes[0, j].set(title=f"Test error by {c}", ylabel="MAE" if j == 0 else None)
            axes[1, j].bar(x, s["bias"], color=np.where(s["bias"] >= 0, BLUE, ORANGE), width=0.6)
            axes[1, j].axhline(0, color=INK_2, lw=0.8)
            axes[1, j].set(ylabel="Bias = mean(actual − pred)" if j == 0 else None)
            axes[1, j].tick_params(axis="x", rotation=30)
        axes[1, 0].annotate("blue = under-prediction, orange = over-prediction", (0, -0.42),
                            xycoords="axes fraction", fontsize=8, color=INK_2)
        fig.suptitle(f"Error slices — {target_name}")
        fig.tight_layout(); save(fig, fig_path)
    return out
