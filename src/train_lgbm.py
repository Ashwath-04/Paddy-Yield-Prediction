import lightgbm as lgb

def train_lgbm(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=20,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )

    model.fit(X_train, y_train)
    return model