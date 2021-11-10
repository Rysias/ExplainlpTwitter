"""
Clean the twitter data for embedding
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("./output")


def find_file(path, pattern="*"):
    return next(path.glob(pattern))


df = pd.read_csv(
    find_file(DATA_DIR, pattern="*training*"), encoding="latin-1", header=None
)
df.columns = ["Sentiment", "id", "Date", "Query", "User", "Tweet"]


df = pd.read_csv(find_file(DATA_DIR, "*training*"))
df.head()
model = SentenceTransformer("Maltehb/-l-ctra-danish-electra-small-cased")
embs = model.encode(df["text"])


full_embs = np.hstack((df["status_id"].values.reshape(-1, 1), embs))
np.save(DATA_DIR / "embs.npy", full_embs)
