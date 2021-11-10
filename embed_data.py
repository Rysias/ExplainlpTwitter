"""
Embeds the data using ÆLÆCTRA and saves it locally
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./output")
df = pd.read_csv(DATA_DIR / "full_tweets.csv")
df.head()
model = SentenceTransformer("Maltehb/-l-ctra-danish-electra-small-cased")
embs = model.encode(df["text"])


full_embs = np.hstack((df["status_id"].values.reshape(-1, 1), embs))
np.save(DATA_DIR / "embs.npy", full_embs)
