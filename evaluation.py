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


# Evaluating my model
topic_preds = pd.read_csv(OUTPUT_DIR / "topic_preds.csv")
topic_y_true = topic_preds["y_true"]

# Evaluating sentiment on same data!
big_preds = pd.read_csv(OUTPUT_DIR / "big_preds.csv")
big_preds["id"] = big_preds["id"].astype(np.uint64)
small_preds = big_preds[big_preds["id"].isin(topic_preds["id"])]

y_true = small_preds["Sentiment"]
y_pred = small_preds["pred"]

print("Evaluation of topic model")
evaluate_preds(topic_preds["y_true"], topic_preds["y_pred"], EVAL_DICT)
print("Evaluation of big model")
evaluate_preds(y_true, y_pred, EVAL_DICT)
