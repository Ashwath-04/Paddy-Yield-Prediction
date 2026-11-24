"""Generates the five analysis notebooks (python notebooks/build_notebooks.py), then
execute them with:  jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
Notebooks are a presentation layer over src/ and the artefacts written by scripts/train.py /
scripts/evaluate.py, so they run in seconds and never re-tune or touch the test set."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
from src import config
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)
M, F = config.METRICS_DIR, config.FIG_DIR
show = lambda name, w=900: display(Image(filename=str(F / name), width=w))"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c)
                  for c in cells]
    return book


NOTEBOOKS = {}

NOTEBOOKS["01_eda.ipynb"] = [
    "md:# 01 · Data validation & EDA\nUCI **Paddy Dataset** (id 1186, DOI 10.24432/C55W3J, CC BY 4.0). "
    "This notebook validates the raw file, identifies target / feature types and runs the leakage "
    "diagnostics that shape every later decision.",
    SETUP,
    "from src.data_loader import load_raw, load_clean\nraw_orig = load_raw()\nraw = load_clean()\n"
    "print(raw.shape)\nraw_orig.head()",
    "md:## Schema, feature roles and cardinality",
    "from src.validation import schema_report\nschema_report(raw)",
    "md:## Validation summary (missing, invalid, duplicates, target)",
    "v = json.loads((M/'validation.json').read_text())\n"
    "print(json.dumps({k: v[k] for k in ['shape','missing_values_total','invalid_values','duplicates','duplicates_after_dedup']}, indent=1))\n"
    "pd.DataFrame({'yield_kg': v['target_yield_kg'], 'yield_kg_per_ha (dedup)': v['target_yield_kg_per_ha']})",
    "md:## Outliers (1.5×IQR per numeric column, raw data)",
    "pd.read_csv(M/'outlier_counts_iqr.csv', index_col=0).query('n_iqr_outliers > 0')",
    "show('eda_outliers.png')",
    "md:## Leakage & structural diagnostics",
    "leak = json.loads((M/'leakage_report.json').read_text())\nprint(json.dumps(leak['deduplicated'], indent=1))",
    "show('eda_weather_by_block.png')",
    "md:## Target distribution",
    "show('eda_target_distribution.png')",
    "md:## Feature distributions & correlation",
    "show('eda_feature_distributions.png', 1100)",
    "show('eda_correlation_matrix.png', 900)",
    "md:## Yield vs agronomic variables\nLeft panels: yield per hectare (jittered strips, group means). Right: total yield "
    "coloured by farm size. These are associations only; no causal claim is made.",
    *[f"show('eda_yield_vs_{k}.png')" for k in ("rainfall", "temperature", "nitrogen", "nitrogen_per_ha", "humidity", "hectares")],
    "show('eda_categorical_effects.png', 1100)",
    "md:## Findings\n{{FINDINGS_01}}",
]

NOTEBOOKS["02_feature_engineering.ipynb"] = [
    "md:# 02 · Feature engineering, leakage ablation, feature selection & split strategy",
    SETUP,
    "from src.features import FEATURE_DOCS, add_engineered_features\n"
    "from src.data_loader import load_clean\nfrom src.preprocessing import drop_exact_duplicates, add_per_ha_target\n"
    "df = add_engineered_features(add_per_ha_target(drop_exact_duplicates(load_clean())))\n"
    "pd.DataFrame({'definition': FEATURE_DOCS, 'n_unique': df[list(FEATURE_DOCS)].nunique(),\n"
    "              'corr_with_yield_per_ha': df[list(FEATURE_DOCS)].corrwith(df[config.TARGET_PER_HA]).round(3)})",
    "md:## Feature selection (fitted on the training split only)",
    "sel = json.loads((M/'feature_selection.json').read_text())\n"
    "print({k: sel[k] for k in ['n_source_predictors','n_leakage_removed','n_engineered','n_encoded_features_before_selection','n_selected','reduction_pct']})\n"
    "pd.Series(sel['dropped'], name='reason').to_frame()",
    "pd.read_csv(M/'feature_selection_cv_check.csv')",
    "md:### Mutual information vs LightGBM gain (per-hectare target)",
    "pd.read_csv(M/'feature_scores_per_ha.csv', index_col=0).head(20)",
    "pd.read_csv(M/'feature_scores_total.csv', index_col=0).head(10)",
    "md:## Leakage ablation (5-fold CV, default LightGBM)",
    "pd.read_csv(M/'leakage_ablation.csv')",
    "md:## Split-strategy sensitivity (training data)",
    "pd.read_csv(M/'split_sensitivity.csv')",
    "md:## Findings\n{{FINDINGS_02}}",
]

NOTEBOOKS["03_baselines.ipynb"] = [
    "md:# 03 · Baseline models (5-fold CV on train; hold-out test reported once)",
    SETUP,
    "import matplotlib.pyplot as plt\nfrom src.plotting import BLUE, ORANGE\n"
    "for t in ('total', 'per_ha'):\n"
    "    cv = pd.read_csv(M/f'cv_results_{t}.csv')\n"
    "    display(Markdown(f'### Target: `{t}`'))\n"
    "    display(cv[['model','stage','cv_R2','cv_R2_std','cv_RMSE','cv_MAE','cv_MAPE_%']].round(4))",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "for ax, t in zip(axes, ('total', 'per_ha')):\n"
    "    cv = pd.read_csv(M/f'cv_results_{t}.csv').query('stage == \"baseline\"').sort_values('cv_R2')\n"
    "    ax.barh(cv.model, cv.cv_R2, xerr=cv.cv_R2_std, color=BLUE, height=0.6)\n"
    "    ax.set(title=f'CV R² — {t}', xlabel='R²', xlim=(min(0, cv.cv_R2.min()), 1))\n"
    "fig.tight_layout(); fig.savefig(F/'baseline_cv_r2.png', dpi=130)",
    "md:## Findings\n{{FINDINGS_03}}",
]

NOTEBOOKS["04_optuna.ipynb"] = [
    "md:# 04 · Optuna hyperparameter optimisation\nTPE sampler (seed 42), objective = 5-fold CV RMSE on the "
    "training split. The test split is never used during tuning.",
    SETUP,
    "import joblib, matplotlib.pyplot as plt\nfrom src.plotting import BLUE, ORANGE\n"
    "rows = []\nfor t in ('total','per_ha'):\n    for m in ('LightGBM','XGBoost'):\n"
    "        b = json.loads((M/f'optuna_best_{t}_{m}.json').read_text())\n"
    "        rows.append({'target': t, 'model': m, 'trials': b['n_trials'], 'best_cv_RMSE': round(b['best_cv_rmse'],2), 'seconds': b['seconds'], **b['best_params']})\n"
    "pd.DataFrame(rows)",
    "fig, axes = plt.subplots(2, 2, figsize=(13, 8))\n"
    "for i, t in enumerate(('total','per_ha')):\n    for j, m in enumerate(('LightGBM','XGBoost')):\n"
    "        s = joblib.load(config.MODELS_DIR/f'optuna_study_{t}_{m}.joblib')\n"
    "        v = np.array([tr.value for tr in s.trials])\n"
    "        ax = axes[i, j]\n"
    "        ax.scatter(range(len(v)), v, s=14, color=BLUE, alpha=0.6, label='trial')\n"
    "        ax.plot(np.minimum.accumulate(v), color=ORANGE, lw=2, label='best so far')\n"
    "        ax.set(title=f'{m} — {t}', xlabel='trial', ylabel='CV RMSE', ylim=(v.min()*0.98, np.quantile(v, 0.9)*1.02)); ax.legend()\n"
    "fig.tight_layout(); fig.savefig(F/'optuna_history.png', dpi=130)",
    "from optuna.importance import get_param_importances\n"
    "imp = {}\nfor t in ('total','per_ha'):\n    for m in ('LightGBM','XGBoost'):\n"
    "        s = joblib.load(config.MODELS_DIR/f'optuna_study_{t}_{m}.joblib')\n"
    "        imp[f'{m}/{t}'] = get_param_importances(s)\n"
    "pd.DataFrame(imp).round(3)",
    "md:## Baseline vs tuned (CV on train)",
    "for t in ('total','per_ha'):\n    cv = pd.read_csv(M/f'cv_results_{t}.csv').set_index('model')\n"
    "    display(Markdown(f'**{t}**'))\n"
    "    display(cv.loc[['LightGBM','LightGBM (Optuna)','XGBoost','XGBoost (Optuna)'], ['cv_R2','cv_RMSE','cv_MAE']].round(4))",
    "md:## Findings\n{{FINDINGS_04}}",
]

NOTEBOOKS["05_shap.ipynb"] = [
    "md:# 05 · Final test evaluation, SHAP & error analysis",
    SETUP,
    "final = json.loads((M/'final_results.json').read_text())\n"
    "for t in ('total','per_ha'):\n    display(Markdown(f'### Target `{t}` — best by CV: **{final[t][\"best_model_by_cv\"]}**'))\n"
    "    display(pd.read_csv(M/f'test_results_{t}.csv').round(4))",
    "show('pred_vs_actual_total.png', 450); show('pred_vs_actual_per_ha.png', 450)",
    "show('residuals_per_ha.png')",
    "md:## SHAP — yield per hectare",
    "show('shap_per_ha_bar.png', 650); show('shap_per_ha_summary.png', 650)",
    "show('shap_per_ha_grouped_importance.png', 650)",
    "pd.read_csv(M/'shap_importance_grouped_per_ha.csv', index_col=0).head(12)",
    "show('shap_per_ha_waterfall_typical.png', 650); show('shap_per_ha_waterfall_worst.png', 650)",
    "md:## SHAP — total yield",
    "show('shap_total_bar.png', 650); show('shap_total_summary.png', 650)",
    "pd.read_csv(M/'shap_importance_grouped_total.csv', index_col=0).head(10)",
    "show('shap_total_waterfall_typical.png', 650)",
    "md:## Error analysis (test set, per-hectare model)",
    "print(json.dumps({k: v for k, v in final['per_ha']['error_analysis'].items()}, indent=1))",
    "show('error_slices_per_ha.png', 1100)",
    "for k in ('agriblock','hectares','variety','soil_type','nursery_type'):\n"
    "    display(Markdown(f'**by {k}**')); display(pd.read_csv(M/f'error_by_{k}_per_ha.csv'))",
    "display(Markdown('**15 largest errors**')); display(pd.read_csv(M/'top_errors_per_ha.csv'))\n"
    "display(Markdown('**15 smallest errors**')); display(pd.read_csv(M/'lowest_errors_per_ha.csv'))",
    "md:## Findings\n{{FINDINGS_05}}",
]


def build(findings: dict | None = None):
    findings = findings or {}
    for name, cells in NOTEBOOKS.items():
        key = "FINDINGS_" + name[:2]
        cells = [c.replace("{{" + key + "}}", findings.get(key, "_See README._")) for c in cells]
        nbf.write(nb(cells), HERE / name)


if __name__ == "__main__":
    import importlib.util
    f = HERE / "findings.py"
    found = {}
    if f.exists():
        spec = importlib.util.spec_from_file_location("findings", f)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        found = mod.FINDINGS
    build(found)
    print("notebooks written")
