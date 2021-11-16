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

# Explore the topics #
# Explore topic words
topic_dict = {
    top: [word[0] for word in words]
    for top, words in topic_model.get_topics().items()
    if top != -1
}
print(topic_dict)

# Explore example tweets
for top in topic_dict.keys():
    print(f"examples for topic {top}")
    example_tweets = df.loc[df["topic"] == top, "doc"].sample(10).tolist()
    print(example_tweets)

# Names to topic (manually)
topic_names = {
    0: "Popular media",
    1: "Local talk",
    2: "Media",
    3: "User small talk",
    4: "Social experiences",
    5: "Wishes and dreams",
    6: "Negative feelings",
    7: "Expressing Feelings",
    8: "User encouragment",
    9: "User greetings",
}
