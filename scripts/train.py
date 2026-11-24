"""End-to-end training pipeline.

    python scripts/train.py                 # full run (40 Optuna trials / model / target)
    python scripts/train.py --trials 10     # quick run

Stages: validation -> EDA -> preprocessing -> feature engineering -> leakage analysis
-> feature selection (train only) -> split -> baselines (CV) -> Optuna (CV on train)
-> refit on full train -> save models. Test-set scoring happens in scripts/evaluate.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root (this file lives in scripts/)

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src import config, eda
from src.data_loader import load_clean
from src.evaluation import cv_evaluate
from src.features import (FEATURE_DOCS, add_engineered_features, feature_columns,
                          model_importance, mutual_information, redundancy_filter)
from src.models import baseline_models, hectares_only_linear, make_lgbm
from src.preprocessing import (CategoricalEncoder, add_per_ha_target, drop_exact_duplicates,
                               encode_wind_direction, split)
from src.tuning import build_tuned, tune
from src.validation import (duplicate_report, invalid_value_report, leakage_report,
                            outlier_report, schema_report)

M = config.METRICS_DIR
META_COLS = ["hectares", "agriblock", "variety", "soil_type", "nursery_type"]


def dump(obj, name):
    (M / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def design_matrix(df: pd.DataFrame, include_trash=False):
    """Full model matrix for an arbitrary frame (used for the leakage ablation)."""
    d = encode_wind_direction(add_engineered_features(add_per_ha_target(df)))
    cols = feature_columns(d) + (["trash_bundles"] if include_trash else [])
    return CategoricalEncoder().fit_transform(d[cols]), d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(n_trials: int):
    # ------------------------------------------------------------------ 1. validation
    raw = load_clean()
    log(f"Loaded raw data {raw.shape}")
    schema_report(raw).to_csv(M / "schema_report.csv")
    raw_feats = [c for c in raw.columns if c not in (config.TARGET_RAW, *config.LEAKAGE_COLUMNS)]
    validation = {
        "shape": list(raw.shape),
        "n_predictors_in_source": raw.shape[1] - 1,
        "missing_values_total": int(raw.isna().sum().sum()),
        "invalid_values": invalid_value_report(raw),
        "duplicates": duplicate_report(raw, raw_feats, config.TARGET_RAW),
        "target_yield_kg": raw[config.TARGET_RAW].describe().round(2).to_dict(),
        "target_yield_kg_skew": float(raw[config.TARGET_RAW].skew()),
    }
    outlier_report(raw).to_csv(M / "outlier_counts_iqr.csv", header=["n_iqr_outliers"])
    leak_raw = leakage_report(raw)

    # ----------------------------------------------------- 2. clean + features + EDA
    df = add_per_ha_target(drop_exact_duplicates(raw))
    leak_dedup = leakage_report(df.drop(columns=[config.TARGET_PER_HA]))
    validation["target_yield_kg_per_ha"] = df[config.TARGET_PER_HA].describe().round(2).to_dict()
    validation["duplicates_after_dedup"] = duplicate_report(df, raw_feats, config.TARGET_RAW)
    dump(validation, "validation.json")
    dump({"raw": leak_raw, "deduplicated": leak_dedup}, "leakage_report.json")
    log(f"Dedup: {len(raw)} -> {len(df)} rows; per-ha R2 ceiling (dedup) ~ {leak_dedup['per_ha_r2_ceiling_estimate']:.3f}")

    df = add_engineered_features(df)
    corr = eda.run_all(df)
    corr.to_csv(M / "correlation_matrix.csv")
    log("EDA figures written")

    # --------------------------------------------------------------- 3. split + encode
    dfw = encode_wind_direction(df)
    train_df, test_df = split(dfw)
    feats = feature_columns(dfw)
    enc = CategoricalEncoder().fit(train_df)
    Xtr_full = enc.transform(train_df[feats])
    Xte_full = enc.transform(test_df[feats])
    log(f"Split: train {len(train_df)}, test {len(test_df)}; encoded features {Xtr_full.shape[1]}")

    # ------------------------------------------------------------- 4. feature selection
    kept, dropped = redundancy_filter(Xtr_full)
    sel = {
        "n_source_predictors": raw.shape[1] - 1,
        "n_leakage_removed": len(config.LEAKAGE_COLUMNS),
        "n_engineered": len(FEATURE_DOCS),
        "n_encoded_features_before_selection": Xtr_full.shape[1],
        "n_selected": len(kept),
        "reduction_pct": round(100 * (1 - len(kept) / Xtr_full.shape[1]), 1),
        "selected": kept,
        "dropped": dropped,
    }
    sel_rows = []
    for tname, tcol in config.TARGETS.items():
        ytr = train_df[tcol]
        mi = mutual_information(Xtr_full[kept], ytr)
        gi = model_importance(Xtr_full[kept], ytr)
        pd.DataFrame({"mutual_information": mi, "lgbm_gain_share": gi}).sort_values(
            "mutual_information", ascending=False).round(5).to_csv(M / f"feature_scores_{tname}.csv")
        for label, cols in (("all encoded features", list(Xtr_full.columns)), ("selected features", kept)):
            r = cv_evaluate(make_lgbm(), Xtr_full[cols], ytr)
            sel_rows.append({"target": tname, "feature_set": label, "n_features": len(cols),
                             "cv_R2": r["cv_R2"], "cv_RMSE": r["cv_RMSE"]})
    sel["cv_check"] = sel_rows
    dump(sel, "feature_selection.json")
    pd.DataFrame(sel_rows).round(4).to_csv(M / "feature_selection_cv_check.csv", index=False)
    log(f"Feature selection: {Xtr_full.shape[1]} -> {len(kept)} ({sel['reduction_pct']}% reduction)")

    Xtr, Xte = Xtr_full[kept], Xte_full[kept]
    pd.concat([train_df[META_COLS + list(config.TARGETS.values())].add_prefix("meta__"), Xtr], axis=1) \
        .to_csv(config.DATA_PROCESSED / "train.csv", index=False)
    pd.concat([test_df[META_COLS + list(config.TARGETS.values())].add_prefix("meta__"), Xte], axis=1) \
        .to_csv(config.DATA_PROCESSED / "test.csv", index=False)
    joblib.dump({"encoder": enc, "selected": kept}, config.MODELS_DIR / "preprocessor.joblib")

    # --------------------------------------------- 5. leakage ablation (CV, LightGBM)
    abl = []
    X_dup_trash, d_dup = design_matrix(raw, include_trash=True)
    X_ded_trash, d_ded = design_matrix(drop_exact_duplicates(raw), include_trash=True)
    compact = [c for c in X_ded_trash.columns if c == "hectares" or "=" in c]
    for tname, tcol in config.TARGETS.items():
        settings = [
            ("raw incl. duplicates + trash (naive)", X_dup_trash, d_dup[tcol], make_lgbm()),
            ("deduplicated + trash", X_ded_trash, d_ded[tcol], make_lgbm()),
            ("deduplicated, no trash (all features)", X_ded_trash.drop(columns="trash_bundles"), d_ded[tcol], make_lgbm()),
            ("compact: hectares+agriblock+variety+soil+nursery", X_ded_trash[compact], d_ded[tcol], make_lgbm()),
            ("hectares only (OLS)", X_ded_trash[["hectares"]], d_ded[tcol], hectares_only_linear()),
        ]
        for name, X, y, model in settings:
            r = cv_evaluate(model, X, y)
            abl.append({"target": tname, "setting": name, "n_rows": len(X),
                        "cv_R2": r["cv_R2"], "cv_RMSE": r["cv_RMSE"], "cv_MAE": r["cv_MAE"]})
    pd.DataFrame(abl).round(4).to_csv(M / "leakage_ablation.csv", index=False)

    # ---------------------------------------------- 6. split-strategy sensitivity (train)
    # Weather is a function of agriblock and inputs a function of hectares, so these five
    # columns identify the full feature configuration of a row.
    cfg_groups = train_df[META_COLS].astype(str).agg("|".join, axis=1)
    schemes = {
        "Random 5-fold": (KFold(5, shuffle=True, random_state=config.SEED), None),
        "Leave-one-agriblock-out (6 folds)": (GroupKFold(6), train_df["agriblock"]),
        "Group by management config (5 folds)": (GroupKFold(5), cfg_groups),
    }
    sens = []
    for tname, tcol in config.TARGETS.items():
        for sname, (cv, groups) in schemes.items():
            r = cv_evaluate(make_lgbm(), Xtr, train_df[tcol], cv=cv, groups=groups)
            sens.append({"target": tname, "scheme": sname, "cv_R2": r["cv_R2"],
                         "cv_R2_std": r["cv_R2_std"], "cv_RMSE": r["cv_RMSE"]})
    pd.DataFrame(sens).round(4).to_csv(M / "split_sensitivity.csv", index=False)
    log("Leakage ablation + split sensitivity done")

    # ------------------------------------------------- 7. baselines + Optuna per target
    summary = {}
    for tname, tcol in config.TARGETS.items():
        ytr = train_df[tcol]
        rows, fitted = [], {}
        for name, model in baseline_models().items():
            r = cv_evaluate(model, Xtr, ytr)
            rows.append({"model": name, "stage": "baseline", **r})
            fitted[name] = model
            log(f"  [{tname}] {name:32s} CV R2={r['cv_R2']:.4f} RMSE={r['cv_RMSE']:.2f}")

        for name in ("LightGBM", "XGBoost"):
            t0 = time.time()
            study = tune(name, Xtr, ytr, n_trials=n_trials)
            study.trials_dataframe().to_csv(M / f"optuna_trials_{tname}_{name}.csv", index=False)
            joblib.dump(study, config.MODELS_DIR / f"optuna_study_{tname}_{name}.joblib")
            dump({"best_params": study.best_params, "best_cv_rmse": study.best_value,
                  "n_trials": len(study.trials), "seconds": round(time.time() - t0, 1)},
                 f"optuna_best_{tname}_{name}.json")
            model = build_tuned(name, study.best_params)
            r = cv_evaluate(model, Xtr, ytr)
            tuned = f"{name} (Optuna)"
            rows.append({"model": tuned, "stage": "optuna", **r})
            fitted[tuned] = model
            log(f"  [{tname}] {tuned:32s} CV R2={r['cv_R2']:.4f} RMSE={r['cv_RMSE']:.2f} ({time.time()-t0:.0f}s)")

        cvtab = pd.DataFrame(rows)
        cvtab.round(4).to_csv(M / f"cv_results_{tname}.csv", index=False)
        cand = cvtab[~cvtab.model.str.contains("reference")]
        best = cand.sort_values("cv_RMSE").iloc[0]["model"]

        for name, model in fitted.items():
            model.fit(Xtr, ytr)
        joblib.dump(fitted, config.MODELS_DIR / f"models_{tname}.joblib")
        joblib.dump(fitted[best], config.MODELS_DIR / f"best_model_{tname}.joblib")
        summary[tname] = {"target_column": tcol, "best_model_by_cv": best}
        log(f"[{tname}] best by CV RMSE: {best}")

    dump(summary, "training_summary.json")
    log("Training complete. Next: python scripts/evaluate.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=config.OPTUNA_TRIALS)
    main(ap.parse_args().trials)
