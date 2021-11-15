from typing import Dict, Union, Callable
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import metrics

OUTPUT_DIR = Path("./output/")
EVAL_DICT = {
    "accuracy": metrics.accuracy_score,
    "f1": metrics.f1_score,
    "roc_auc": metrics.roc_auc_score,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score,
}


def evaluate_preds(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    eval_dict: Dict[str, Callable],
) -> Dict[str, float]:
    return {metric: func(y_true, y_pred) for metric, func in eval_dict.items()}


# Evaluating sentiment
big_preds = pd.read_csv(OUTPUT_DIR / "big_preds.csv")
y_true = big_preds["Sentiment"]
y_pred = big_preds["pred"]

# Evaluating my model
topic_preds = pd.read_csv(OUTPUT_DIR / "topic_preds.csv")
topic_y_true = topic_preds["y_true"]

W
