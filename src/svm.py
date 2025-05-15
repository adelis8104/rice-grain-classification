from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from joblib import parallel_backend


def train_svm(X_train, y_train, param_grid=None, cv=5, n_jobs=-1):
    """
    Train and tune an SVM classifier using grid search.
    Falls back to a default SVM if grid search fails.
    Returns the trained model with a `best_params_` attribute.
    """
    print(f"[SVM] Starting grid search with CV={cv} and n_jobs={n_jobs}...")
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }
    try:
        with parallel_backend("loky", n_jobs=n_jobs):
            grid = GridSearchCV(
                SVC(), param_grid, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=2
            )
            grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best.best_params_ = grid.best_params_
        print(f"[SVM] Best parameters: {best.best_params_}")
        return best
    except Exception as e:
        print(f"[SVM] Grid search failed ({e}), using default model")
        model = SVC(C=1, kernel="linear", gamma="scale")
        model.fit(X_train, y_train)
        model.best_params_ = {"C": 1, "kernel": "linear", "gamma": "scale"}
        print(f"[SVM] Fallback parameters: {model.best_params_}")
        return model
