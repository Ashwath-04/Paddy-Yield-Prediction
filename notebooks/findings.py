"""Narrative findings injected into the notebooks by build_notebooks.py. Numbers come from
reports/metrics/* of the reference run (seed 42, 40 Optuna trials)."""

FINDINGS = {
    "FINDINGS_01": """
* **Target**: `Paddy yield(in Kg)` — total paddy harvested per farm record, in kg (5,410–38,814). No missing values, no physically invalid values.
* **Feature types**: 44 predictors = 36 numeric + 8 categorical (`Agriblock`, `Variety`, `Soil Types`, `Nursery`, 4× wind direction). Geographic: `Agriblock` (6 blocks of Cuddalore/Kallakurichi districts, Tamil Nadu). **Temporal: none** — the `D1_D30 … D91_D120` columns are crop-stage windows within a season, not calendar time; there is no year/season column.
* **451 exact duplicate rows (16.2%)** → removed before splitting (2,338 rows remain).
* **Farm size drives everything**: all 10 input-quantity columns (seed, DAP, urea, potash, micronutrients, weedicide, pesticide, nursery area, land-prep) are *exactly* `hectares × constant`. Every farm applies identical per-hectare doses, so fertilizer/nitrogen carries **no information beyond farm size**. `corr(yield_kg, hectares) = 0.994` → hectares alone gives R² = 0.989.
* **Weather ≡ location**: all 28 weather columns take exactly one value per agriblock (6 weather profiles total). Weather effects cannot be separated from location effects. `rain + AI` is constant per stage (40 / 458 / 417 / 125 mm), so "AI" is complementary water and adds no information.
* **Post-harvest leakage**: `Trash(in bundles)` is a harvest by-product (r = 0.957 with yield) — excluded. Here it equals hectares × a variety constant (80/90/100 bundles/ha, one exception), so it happens not to leak extra signal, but it would be unavailable at prediction time.
* **Per-hectare yield** (5,410–6,469 kg/ha, mean ≈ 5,985) is the agronomically meaningful quantity. ~41% of its variance occurs between rows with *identical* features → estimated R² ceiling ≈ 0.59.
* EDA associations: per-ha yield is ~5,740 kg/ha for 1–3 ha farms and ~6,050–6,200 for 4–6 ha farms; differences across variety, soil, nursery, block (and therefore weather) are small. These are associations, not causal effects.
""",
    "FINDINGS_02": """
* 15 agronomic features were engineered and documented. **4 are constant** (N, P₂O₅, K₂O per ha and the N:K ratio — every farm uses the same dose), **5 are ≥0.99-correlated duplicates** of existing columns (N applied ≡ hectares; season rain/irrigation/rain-share ≡ one rain column; GDD ≡ mean temperature). The 6 survivors carry ~0 SHAP importance. Feature engineering therefore **did not add predictive signal** on this dataset — reported, not hidden.
* **Feature selection** (constant + |r|≥0.99 redundancy filter, fit on train only): 69 encoded features → **44 (36.2% reduction)** with **identical CV score** (LightGBM CV R² 0.4501 vs 0.4501 per-ha; 0.9911 vs 0.9911 total). MI and LightGBM gain both rank `hectares` far above everything else.
* **Leakage ablation** (CV, LightGBM): keeping the duplicate rows inflates per-ha R² from 0.435 → 0.488 — duplicate leakage is real and measurable. Trash adds nothing here. A compact model with only *hectares + agriblock + variety + soil + nursery* (5 source variables) matches the full feature set (0.448 vs 0.435 per-ha; 0.9907 vs 0.9905 total) — the 44 columns hold about 5 variables of information.
* **Split decision**: no time column → time split impossible. Random 5-fold, leave-one-agriblock-out and group-by-configuration CV give consistent per-ha R² (0.45 / 0.48 / 0.45), so random splitting does not leak through location here. Final protocol: deduplicate → stratified (hectares × agriblock) 80/20 hold-out → 5-fold CV on train for all model selection & tuning.
""",
    "FINDINGS_03": """
* **Total yield**: every model, including a one-variable OLS on hectares (CV R² 0.9888), scores R² ≈ 0.99. Tree ensembles reduce RMSE from ~978 kg to ~840 kg (≈14%) — the real gain of ML over the trivial baseline.
* **Per-hectare yield**: OLS ≈ 0.35, Random Forest 0.44, LightGBM 0.45, Gradient Boosting 0.50 (CV R²). Non-linear models help because the farm-size effect is a step (1–3 ha vs 4–6 ha) plus variety×size interactions.
""",
    "FINDINGS_04": """
* 40 TPE trials per model per target (~1 min each on CPU). Optuna favoured **many shallow trees with a low learning rate** (LightGBM: 5–6 leaves, lr ≈ 0.005–0.009, 850–1,100 trees), consistent with a low-dimensional signal plus noise.
* Gains: per-ha LightGBM CV RMSE 212.9 → 200.2 (R² 0.450 → 0.513); XGBoost 219.0 → 201.4 (0.418 → 0.507). Total-yield LightGBM 873.8 → 830.5 kg (R² 0.9911 → 0.9919).
* Tuning closes the gap to the (untuned) sklearn GradientBoosting but improvements are modest — the problem is bounded by noise (ceiling ≈ 0.59), not by model capacity.
""",
    "FINDINGS_05": """
* **Hold-out test (468 rows, never used in tuning)** — best model by CV = LightGBM (Optuna):
  * Total yield: **R² 0.991, RMSE 881 kg, MAE 624 kg, MAPE 2.77%** (hectares-only OLS: R² 0.988, RMSE 1,010 kg).
  * Yield per hectare: **R² 0.496, RMSE 210 kg/ha, MAE 163 kg/ha, MAPE 2.77%** (mean predictor RMSE 296; hectares-only 238).
* **SHAP (per-ha)**: hectares 82.8% of mean |SHAP|, variety 9.2%, soil 1.7%, nursery 0.8%, and all location/weather columns combined 5.5%. Nitrogen/fertilizer columns have **zero** importance (they are constant per ha or identical to hectares). For total yield, hectares = 97%.
* Weather columns appearing in the SHAP ranking (`wind_kn_d1_30`, `tmax_d61_90`, …) should be read as **agriblock identifiers**, because weather is a deterministic function of block.
* **Error analysis**: errors are symmetric (244 under- vs 224 over-predictions). The largest errors are *low-yield farms of 4–6 ha* (actual ≈ 5,400–5,500 kg/ha, predicted ≈ 6,100) — the model predicts the group mean and cannot see what made those farms under-perform. 4-ha and 6-ha farms have the highest MAE (226 / 207 kg/ha) because yields in those groups are most dispersed; `ponmani` has the highest MAE and a slight over-prediction bias. Per-block MAE ranges only 153–172 kg/ha → no strong regional failure. No temporal analysis is possible (no time column).
""",
}
