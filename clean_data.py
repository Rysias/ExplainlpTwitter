"""
Clean the twitter data for embedding
"""
from os import replace
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("./output")


def find_file(path, pattern="*"):
    return next(path.glob(pattern))


def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


df = pd.read_csv(
    find_file(DATA_DIR, pattern="*training*"), encoding="latin-1", header=None
)
df.columns = ["Sentiment", "id", "Date", "Query", "User", "Tweet"]

df["cleantext"] = df["Tweet"].apply(lambda x: preprocess(x))
df["Sentiment"] = (df["Sentiment"] / 4).astype(np.uint8)

df[["id", "cleantext", "Sentiment"]].to_csv(DATA_DIR / "clean_tweets.csv")
