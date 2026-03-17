from xgboost import XGBRegressor

def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5
    )

    model.fit(X_train, y_train)
    return model