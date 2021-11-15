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
embeddings = np.load(DATA_DIR / "embeddings.npy")
idxs = df["topic"] != -1


topic_embedder.calculate_centroids()
