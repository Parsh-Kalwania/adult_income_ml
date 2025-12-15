# evaluate_best_model.py
from preprocess import load_and_preprocess
from train_models import preprocessor, numeric_features, categorical_features

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd

X_train, X_test, y_train, y_test = load_and_preprocess()

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

feature_names = list(numeric_features) + list(
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(categorical_features)
)

importances = best_model.named_steps["model"].feature_importances_

fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
print(fi.sort_values("Importance", ascending=False).head(10))
