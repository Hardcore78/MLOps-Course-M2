import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

class Config:
    def __init__(self, data, features, model, cv):
        self.data = data
        self.features = features
        self.model = model
        self.cv = cv

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    return df

def get_features(cfg: Config):
    numeric = cfg.features["numeric"]
    categorical = cfg.features["categorical"]
    target = cfg.data["target"]
    return numeric, categorical, target

def split_xy(df: pd.DataFrame, target: str):
    y = df[target]
    if y.dtype == "O":
        y = y.astype(str).str.lower().map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}).fillna(y)
    X = df.drop(columns=[target])
    return X, y

def train_test_split_strat(X, y, cfg: Config):
    return train_test_split(
        X, y,
        test_size=cfg.data.get("test_size", 0.2),
        random_state=cfg.data.get("random_state", 42),
        stratify=y,
    )

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_and_save_roc(y_true, y_proba, out_path: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return auc
