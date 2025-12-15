# tune_model.py
from preprocess import load_and_preprocess
from train_models import preprocessor

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

X_train, _, y_train, _ = load_and_preprocess()

param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5]
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

grid = GridSearchCV(
    pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best F1:", grid.best_score_)
