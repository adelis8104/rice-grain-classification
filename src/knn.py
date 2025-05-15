from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_knn(X_train, y_train, param_grid=None, cv=3, n_jobs=-1):
    """
    Train and tune a KNN classifier using grid search.
    Falls back to a default KNN if grid search fails.
    Returns the trained model with a `best_params_` attribute.
    """
    print(f"[KNN] Starting grid search with CV={cv} and n_jobs={n_jobs}...")
    if param_grid is None:
        param_grid = {"n_neighbors": [4], "weights": ["uniform"]}
    try:
        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=2,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best.best_params_ = grid.best_params_
        print(f"[KNN] Best parameters: {best.best_params_}")
        return best
    except Exception as e:
        print(f"[KNN] Grid search failed ({e}), using default model")
        model = KNeighborsClassifier(n_neighbors=4, weights="uniform")
        model.fit(X_train, y_train)
        model.best_params_ = {"n_neighbors": 4, "weights": "uniform"}
        print(f"[KNN] Fallback parameters: {model.best_params_}")
        return model
