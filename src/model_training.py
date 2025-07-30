# src/model_training.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import inspect

def _get_project_root():
    """
    Detect project root dynamically:
    - If running as a script: use __file__
    - If running from a notebook: use current working directory
    """
    try:
        # Works in .py scripts
        current_file = Path(__file__).resolve()
        return current_file.parent.parent
    except NameError:
        # In Jupyter (no __file__)
        return Path.cwd().parent

def train_and_evaluate(data_path, target_col, model_name_prefix):
    project_root = _get_project_root()
    model_dir = project_root / "models"
    results_dir = project_root / "results"

    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[INFO] Data loaded from: {data_path}")
    print(f"[INFO] Shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)

    lr_metrics = {
        "f1_score": f1_score(y_test, y_pred_lr),
        "auc_pr": average_precision_score(y_test, log_reg.predict_proba(X_test)[:, 1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist()
    }
    print(f"[INFO] F1 Score: {lr_metrics['f1_score']:.4f}, AUC-PR: {lr_metrics['auc_pr']:.4f}")
    
    joblib.dump(log_reg, model_dir / f"{model_name_prefix}_logistic_regression.pkl")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        "f1_score": f1_score(y_test, y_pred_rf),
        "auc_pr": average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }
    print(f"[INFO] F1 Score: {rf_metrics['f1_score']:.4f}, AUC-PR: {rf_metrics['auc_pr']:.4f}")

    joblib.dump(rf, model_dir / f"{model_name_prefix}_random_forest.pkl")

    # Save metrics
    metrics_path = results_dir / f"metrics_{model_name_prefix}.json"
    with open(metrics_path, "w") as f:
        json.dump({"logistic_regression": lr_metrics, "random_forest": rf_metrics}, f, indent=4)

    print(f"[INFO] Logistic Regression metrics: {lr_metrics}")
    print(f"[INFO] Random Forest metrics: {rf_metrics}")
    print(f"[INFO] Models saved to: {model_dir}")
    print(f"[INFO] Metrics saved to: {metrics_path}")
