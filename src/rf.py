from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train, param_grid=None, cv=3, n_jobs=-1):
    """
    Train and tune a Random Forest classifier using grid search.
    Falls back to a default Random Forest if grid search fails.
    Returns the trained model with a `best_params_` attribute.
    """
    print(f"[RF] Starting grid search with CV={cv} and n_jobs={n_jobs}...")
    if param_grid is None:
        param_grid = {
            "n_estimators": [100],
            "max_depth": [None],
            "max_features": ["sqrt"],
        }
    try:
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=2,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best.best_params_ = grid.best_params_
        print(f"[RF] Best parameters: {best.best_params_}")
        return best
    except Exception as e:
        print(f"[RF] Grid search failed ({e}), using default model")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=None, max_features="sqrt", random_state=42
        )
        model.fit(X_train, y_train)
        model.best_params_ = {
            "n_estimators": 100,
            "max_depth": None,
            "max_features": "sqrt",
        }
        print(f"[RF] Fallback parameters: {model.best_params_}")
        return model
