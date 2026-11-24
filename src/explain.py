"""SHAP explanations for tree models (TreeExplainer: exact, CPU-cheap)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.plotting import save


def shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def global_importance(sv: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)


def grouped_importance(imp: pd.Series) -> pd.Series:
    """Sum one-hot dummies (col=value) and wind sin/cos back to their source variable."""
    def root(c):
        c = c.split("=")[0]
        return c[:-4] if c.endswith(("_sin", "_cos")) else c
    return imp.groupby(imp.index.map(root)).sum().sort_values(ascending=False)


def save_plots(sv: shap.Explanation, prefix, fig_dir, title, rows: dict | None = None, max_display=15):
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title(f"SHAP summary — {title}"); save(plt.gcf(), fig_dir / f"{prefix}_summary.png")

    plt.figure()
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title(f"Mean |SHAP| — {title}"); save(plt.gcf(), fig_dir / f"{prefix}_bar.png")

    for name, row in (rows or {"typical": 0}).items():
        plt.figure()
        shap.plots.waterfall(sv[row], max_display=12, show=False)
        plt.title(f"Individual explanation ({name} test sample) — {title}")
        save(plt.gcf(), fig_dir / f"{prefix}_waterfall_{name}.png")

    grp = grouped_importance(global_importance(sv)).head(max_display)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(grp) + 1))
    ax.barh(grp.index, grp.values, color="#2a78d6", height=0.6)
    ax.set(xlabel="mean |SHAP| (summed over encoded columns)", title=f"Global importance by source variable — {title}")
    fig.tight_layout(); save(fig, fig_dir / f"{prefix}_grouped_importance.png")
