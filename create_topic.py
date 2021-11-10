from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = Path("./output")


def load_text_url(text_url: str):
    """Inputs a URL of a newline seperated text-file and returns a list"""
    response = requests.get(text_url)
    return response.text.split("\n")


embs = np.load(DATA_DIR / "embs.npy")
emb_id = embs[:, 0]
embeddings = embs[:, 1:]


docs = pd.read_csv(DATA_DIR / "full_tweets.csv")["text"]

STOP_WORD_URL = "https://gist.githubusercontent.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b/raw/305d8e3930cc419e909d49d4b489c9773f75b2d6/stopord.txt"
STOP_WORDS = load_text_url(STOP_WORD_URL)
vectorizer_model = CountVectorizer(stop_words=STOP_WORDS)

topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=3)

topics, probs = topic_model.fit_transform(docs, embeddings)

topic_model.get_topics()
docs[2085]
