import argparse
import os
import mlflow
from mlflow import sklearn as mlflow_sklearn

from utils import load_config, load_csv, get_features, split_xy, train_test_split_strat, plot_and_save_roc, ensure_dir

ARTIFACT_DIR = "artifacts"

def main(args):
    cfg = load_config(args.config)
    df = load_csv(cfg.data["csv_path"])
    numeric, categorical, target = get_features(cfg)
    X, y = split_xy(df, target)
    _, X_test, _, y_test = train_test_split_strat(X, y, cfg)
    model_uri = f"models:/{cfg.model['registry_name']}/Staging"
    model = mlflow_sklearn.load_model(model_uri)
    y_proba = model.predict_proba(X_test)[:, 1]
    ensure_dir(ARTIFACT_DIR)
    roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    auc = plot_and_save_roc(y_test, y_proba, roc_path)
    with mlflow.start_run(run_name="evaluate", nested=True):
        mlflow.log_metric("eval_roc_auc", float(auc))
        mlflow.log_artifact(roc_path)
    print(f"Eval ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
