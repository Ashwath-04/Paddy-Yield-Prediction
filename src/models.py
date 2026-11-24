"""Model zoo: reference baselines, classical regressors and gradient boosters."""
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def make_lgbm(**params) -> lgb.LGBMRegressor:
    base = dict(n_estimators=400, learning_rate=0.05, subsample_freq=1,
                random_state=config.SEED, n_jobs=-1, verbose=-1)
    base.update(params)
    return lgb.LGBMRegressor(**base)


def make_xgb(**params) -> xgb.XGBRegressor:
    base = dict(n_estimators=400, learning_rate=0.05, max_depth=6, tree_method="hist",
                random_state=config.SEED, n_jobs=-1)
    base.update(params)
    return xgb.XGBRegressor(**base)


def hectares_only_linear() -> Pipeline:
    """Reference model: OLS on farm size only. Quantifies how much of the total-yield
    R^2 is explained by area alone."""
    sel = ColumnTransformer([("h", "passthrough", ["hectares"])], remainder="drop")
    return Pipeline([("select", sel), ("ols", LinearRegression())])


def baseline_models() -> dict:
    return {
        "Mean predictor (reference)": DummyRegressor(strategy="mean"),
        "Hectares-only OLS (reference)": hectares_only_linear(),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                               random_state=config.SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                           random_state=config.SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=config.SEED),
        "LightGBM": make_lgbm(),
        "XGBoost": make_xgb(),
    }
