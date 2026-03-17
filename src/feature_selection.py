from sklearn.feature_selection import VarianceThreshold

def select_features(X):
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    return X_selected