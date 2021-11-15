import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("./output")

all_embs = np.load(DATA_DIR / "topic_embs.npy")
train_df = pd.read_csv(DATA_DIR / "clean_tweets.csv")
train_df.head()

fullX = all_embs[:, 1:]
fullY = train_df["Sentiment"].values
X_train, X_test, Y_train, Y_test = train_test_split(
    fullX, fullY, test_size=10000, random_state=42
)

model = LogisticRegression()
model.fit(X_train, Y_train)
y_preds = model.predict(X_test)
pd.DataFrame({"y_true": Y_test, "y_pred": y_preds}).to_csv(
    DATA_DIR / "topic_preds.csv", index=False
)
