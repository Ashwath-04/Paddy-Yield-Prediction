import optuna
from xgboost import XGBRegressor
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
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }

    model = XGBRegressor(**params, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def tune_xgb(X, y):
    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)

    print("Best XGB Params:", study.best_params)

    best_model = XGBRegressor(**study.best_params)
    best_model.fit(X, y)

    return best_model, study