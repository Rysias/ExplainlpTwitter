import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
from explainlp.explainlp import ClearSearch


TOPIC_PATH = Path("./output/topic_model")
DATA_DIR = Path("./output")
TOPIC_PATH.exists()

topic_model = BERTopic.load(
    str(TOPIC_PATH), embedding_model="finiteautomata/bertweet-base-sentiment-analysis"
)
topic_embedder = ClearSearch(
    "finiteautomata/bertweet-base-sentiment-analysis", topic_model=topic_model
)

# Reducing topics
df = pd.read_csv(DATA_DIR / "doc_topics.csv")
embeddings = np.load(DATA_DIR / "small_embs.npy")
idxs = df["topic"] != -1
topics = df.loc[idxs, "topic"]
probs = df.loc[idxs, "prob"]
embs = embeddings[idxs, 1:]

centroids = topic_embedder.calculate_centroids(topics, probs, embeddings[idxs, :])

# Transforming all the tweets!
all_embs = np.load(DATA_DIR / "embeddings.npy")
new_features = topic_embedder.transform_many(all_embs[:, 1:])

feat_with_id = np.hstack((all_embs[:, 0].reshape(-1, 1), new_features))

# Write to disk
np.save(DATA_DIR / "topic_embs.npy", feat_with_id)
