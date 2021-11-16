import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

DATA_DIR = Path("./output")


def logodds_to_probs(coefs):
    return [np.round(np.exp(x) / (1 + np.exp(x)), 5) for x in coefs[0]]


all_embs = np.load(DATA_DIR / "topic_embs.npy")
train_df = pd.read_csv(DATA_DIR / "clean_tweets.csv")
train_df.head()

fullX = all_embs
fullY = train_df["Sentiment"].values
X_train, X_test, Y_train, Y_test = train_test_split(
    fullX, fullY, test_size=10000, random_state=42
)

model = LogisticRegression()
X_train_norm = normalize(X_train[:, 1:])
model.fit(X_train_norm, Y_train)
y_preds = model.predict(normalize(X_test[:, 1:]))
test_ids = pd.Series(np.rint(X_test[:, 0])).astype(np.uint64).astype(str)
pd.DataFrame({"id": test_ids, "y_true": Y_test, "y_pred": y_preds}).to_csv(
    DATA_DIR / "topic_preds.csv", index=False
)

logodds_to_probs(model.coef_)
