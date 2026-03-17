import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

def train_tabnet(X_train, y_train, X_test, y_test):

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    model = TabNetRegressor()

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=10,
        batch_size=256
    )

    return model