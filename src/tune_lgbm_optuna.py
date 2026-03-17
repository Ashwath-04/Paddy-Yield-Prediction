import optuna
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def objective(trial, X, y):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
        "verbose": -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def tune_lgbm(X, y):
    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)

    print("Best Params:", study.best_params)

    best_model = lgb.LGBMRegressor(**study.best_params)
    best_model.fit(X, y)

    return best_model, study