"""Optuna hyperparameter search. The objective is K-fold CV RMSE on the TRAINING
set only; the hold-out test set is never touched during tuning."""
import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score

from src import config
from src.models import make_lgbm, make_xgb


def lgbm_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1200, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 128, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def xgb_space(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1200, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


SPACES = {"LightGBM": (lgbm_space, make_lgbm), "XGBoost": (xgb_space, make_xgb)}


def tune(name: str, X, y, n_trials: int = config.OPTUNA_TRIALS,
         seed: int = config.SEED) -> optuna.Study:
    space, factory = SPACES[name]
    cv = KFold(config.CV_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        model = factory(**space(trial))
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        trial.set_user_attr("cv_rmse_std", float(np.std(scores)))
        return float(-np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def build_tuned(name: str, params: dict):
    return SPACES[name][1](**params)
