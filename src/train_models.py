# train_models.py
from preprocess import load_and_preprocess

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

X_train, X_test, y_train, y_test = load_and_preprocess()

numeric_features = [
    "age", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]

categorical_features = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race",
    "sex", "native_country"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values("F1-score", ascending=False))
