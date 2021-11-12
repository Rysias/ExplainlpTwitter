from pathlib import Path
from bertopic import BERTopic
from explainlp.explainlp import ClearSearch
import joblib


TOPIC_PATH = Path("./output/topic_model")
TOPIC_PATH.exists()

topic_model = BERTopic.load(
    str(TOPIC_PATH), embedding_model="finiteautomata/bertweet-base-sentiment-analysis"
)
topic_embedder = ClearSearch(
    "finiteautomata/bertweet-base-sentiment-analysis", topic_model=TOPIC_PATH
)
