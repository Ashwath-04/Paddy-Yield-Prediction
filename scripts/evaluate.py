"""Final evaluation on the untouched hold-out test set + SHAP + error analysis.

    python scripts/evaluate.py

Requires the artefacts written by scripts/train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root (this file lives in scripts/)

import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.evaluation import error_analysis, plot_pred_vs_actual, plot_residuals, regression_metrics
from src.explain import global_importance, grouped_importance, save_plots, shap_values

M, F = config.METRICS_DIR, config.FIG_DIR
TITLES = {"total": "total yield (kg)", "per_ha": "yield per hectare (kg/ha)"}


def load_split(name):
    d = pd.read_csv(config.DATA_PROCESSED / f"{name}.csv")
    meta = d[[c for c in d if c.startswith("meta__")]].rename(columns=lambda c: c[6:])
    X = d[[c for c in d if not c.startswith("meta__")]]
    return X, meta


def main():
    Xte, meta_te = load_split("test")
    Xtr, _ = load_split("train")
    summary = json.loads((M / "training_summary.json").read_text())
    final = {}

    for tname, tcol in config.TARGETS.items():
        y_te = meta_te[tcol]
        models = joblib.load(config.MODELS_DIR / f"models_{tname}.joblib")
        best_name = summary[tname]["best_model_by_cv"]
        cv = pd.read_csv(M / f"cv_results_{tname}.csv").set_index("model")

        rows = []
        for name, model in models.items():
            m = regression_metrics(y_te, model.predict(Xte))
            rows.append({"model": name, "cv_R2": cv.loc[name, "cv_R2"], "cv_RMSE": cv.loc[name, "cv_RMSE"],
                         "test_R2": m["R2"], "test_RMSE": m["RMSE"], "test_MAE": m["MAE"], "test_MAPE_%": m["MAPE_%"]})
        tab = pd.DataFrame(rows)
        tab.round(4).to_csv(M / f"test_results_{tname}.csv", index=False)

        best = models[best_name]
        pred = best.predict(Xte)
        plot_pred_vs_actual(y_te, pred, f"{best_name} — {TITLES[tname]} (test)", F / f"pred_vs_actual_{tname}.png")
        plot_residuals(y_te, pred, f"{best_name} — {TITLES[tname]} (test)", F / f"residuals_{tname}.png")

        ea = error_analysis(meta_te, y_te, pred, TITLES[tname], F / f"error_slices_{tname}.png")
        ea["top_errors"].to_csv(M / f"top_errors_{tname}.csv", index=False)
        ea["lowest_errors"].to_csv(M / f"lowest_errors_{tname}.csv", index=False)
        for k, v in ea.pop("slices").items():
            v.to_csv(M / f"error_by_{k}_{tname}.csv")
        ea["high_vs_low_error_profile"] = ea["high_vs_low_error_profile"].to_dict()
        ea.pop("top_errors"); ea.pop("lowest_errors")

        # SHAP: explain the best tree model; if the CV-best is not tree-based, fall back to
        # the best tree model so TreeExplainer applies.
        tree_names = [n for n in tab.sort_values("test_RMSE").model if n not in
                      ("Linear Regression", "Mean predictor (reference)", "Hectares-only OLS (reference)")]
        shap_name = best_name if best_name in tree_names else tree_names[0]
        sv = shap_values(models[shap_name], Xte)
        abs_err = np.abs(y_te.to_numpy() - models[shap_name].predict(Xte))
        typical = int(np.argmin(np.abs(abs_err - np.median(abs_err))))
        save_plots(sv, f"shap_{tname}", F, f"{shap_name}, {TITLES[tname]}",
                   rows={"typical": typical, "worst": int(np.argmax(abs_err))})
        imp = global_importance(sv)
        grp = grouped_importance(imp)
        pd.DataFrame({"mean_abs_shap": imp}).round(4).to_csv(M / f"shap_importance_{tname}.csv")
        pd.DataFrame({"mean_abs_shap": grp, "share_%": 100 * grp / grp.sum()}).round(3) \
            .to_csv(M / f"shap_importance_grouped_{tname}.csv")

        final[tname] = {
            "best_model_by_cv": best_name,
            "test_metrics": regression_metrics(y_te, pred),
            "shap_model": shap_name,
            "shap_top_features": imp.head(10).round(3).to_dict(),
            "shap_top_source_variables_share_%": (100 * grp / grp.sum()).head(10).round(2).to_dict(),
            "error_analysis": ea,
        }
        print(f"\n=== {TITLES[tname]} ===")
        print(tab.round(4).to_string(index=False))
        print(f"Best (by CV): {best_name} -> test {final[tname]['test_metrics']}")
        print("SHAP share by source variable (%):", final[tname]["shap_top_source_variables_share_%"])

    (M / "final_results.json").write_text(json.dumps(final, indent=2, default=float))


if __name__ == "__main__":
    main()
