import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# Paths
PROCESSED_PATH = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.set_tracking_uri("file:///c:/Users/yijin/mlops-customer-churn/mlruns")

def load_data():
    X_train = joblib.load(os.path.join(PROCESSED_PATH, "X_train.pkl"))
    X_test = joblib.load(os.path.join(PROCESSED_PATH, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_PATH, "y_train.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_PATH, "y_test.pkl"))
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    experiments = [
        {
            "model_type": "LogisticRegression",
            "name": "LogisticRegression",
            "model": LogisticRegression(max_iter=1000),
            "params": {"max_iter": 1000}
        },
        {
            "model_type": "RandomForestClassifier",
            "name": "RandomForest_100_5",
            "model": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced"}
        },
        {
            "model_type": "RandomForestClassifier",
            "name": "RandomForest_200_10",
            "model": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
            "params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "model_type": "XGBoost",
            "name": "XGBoost_100_5_0.1",
            "model": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        }
    ]

    mlflow.set_experiment("Customer-Churn-Classifier")

    for exp in experiments:
        model = exp["model"]
        model_type = exp["model_type"]
        model_name = exp["name"]
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.log_model(model, "model")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Log params with fallback
            mlflow.log_param("model_type", model_type)
            for param in ["n_estimators", "max_depth", "learning_rate"]:
                mlflow.log_param(param, exp["params"].get(param, "N/A"))

            # Also log all other params
            for key, value in exp["params"].items():
                if key not in ["n_estimators", "max_depth", "learning_rate"]:
                    mlflow.log_param(key, value)

            print(f"[{model_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
            filename = f"{model_name.lower()}_model.pkl"
            joblib.dump(model, os.path.join(MODEL_DIR, filename))

if __name__ == "__main__":
    train_and_evaluate()
