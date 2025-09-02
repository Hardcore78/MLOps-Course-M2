import argparse
import mlflow
from mlflow import sklearn as mlflow_sklearn
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils import load_config, load_csv, get_features, split_xy, train_test_split_strat
from pipeline import build_pipeline

def main(args):
    cfg = load_config(args.config)
    df = load_csv(cfg.data["csv_path"])
    numeric, categorical, target = get_features(cfg)
    X, y = split_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split_strat(X, y, cfg)
    pipe = build_pipeline(numeric, categorical, cfg.model["type"])
    if cfg.model["type"] == "logreg":
        param_grid = cfg.model["params"]["logreg"]
    else:
        param_grid = cfg.model["params"]["random_forest"]
    cv = StratifiedKFold(n_splits=cfg.cv.get("n_splits", 5), shuffle=True, random_state=cfg.data.get("random_state", 42))
    mlflow.autolog(log_models=True)
    with mlflow.start_run(run_name=f"train_{cfg.model['type']}"):
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring=cfg.cv.get("scoring", "roc_auc"), cv=cv, n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("test_roc_auc", float(test_auc))
        mlflow_sklearn.log_model(best_model, "model")
    print(f"Test ROC-AUC: {test_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
