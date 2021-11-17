import numpy as np
from bertopic import BERTopic
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Iterable, Optional, Union, List, Sequence


class ClearSearch:
    def __init__(
        self,
        topic_model: Union[BERTopic, str, Path],
    ) -> None:

        self.topic_model = self._load_topic_model(topic_model)
        self.nr_topics = self.topic_model.nr_topics

    @staticmethod
    def _load_topic_model(topic_model: Union[BERTopic, str, Path]):
        if isinstance(topic_model, BERTopic):
            return topic_model
        else:
            return BERTopic.load(str(topic_model))

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transforms one document (with potentially multiple paragraphs) into features
        args:
            embeddings: np.ndarray (n_paragraphs, embedding_size)
        returns:
            The similarity to each of the clusters (n_cluster, )
        """
        try:
            return np.mean(cosine_similarity(embeddings, self.centroids), axis=0)
        except ValueError as e:
            if "Reshape" in str(e):
                # Reshape to (1, n)-array
                return self.transform(embeddings.reshape(1, -1))
            elif "0 sample" in str(e):
                nan_result = np.zeros((self.centroids.shape[0],))
                nan_result.fill(np.nan)
                return nan_result
            else:
                raise e

    def transform_many(
        self, doc_embeddings: Union[Sequence[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        featurizes multiple documents where each document is a numpy array of shape (n_paragraphs, embedding_dim)
        Also works if input is a (n_docs, embedding_dim) numpy array
        """
        if type(doc_embeddings) == np.ndarray:
            print("numpy array!")
            return cosine_similarity(doc_embeddings, self.centroids)
        transformed_embeddings = np.zeros((len(doc_embeddings), self.nr_topics))
        for i, embeddings in enumerate(doc_embeddings):
            transformed_embeddings[i, :] += self.transform(embeddings)
        return transformed_embeddings

    def get_topics(self) -> Dict[int, List[str]]:
        raw_topics = self.topic_model.get_topics()
        return {
            k: [topic[0] for topic in topics]
            for k, topics in raw_topics.items()
            if k != -1
        }

    def _get_topic_range(self, topics: Iterable[int]) -> List[int]:
        max_topic = max(topics)
        return list(range(max_topic + 1))

    def calculate_centroids(
        self, topics: np.ndarray, probs: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        unique_topics = self._get_topic_range(topics)
        self.centroids = np.zeros(
            (len(unique_topics), embeddings.shape[1])
        )  # Centroids need dimensions (number of topics, embedding-dimensionality)
        for i in unique_topics:
            self.centroids[i, :] += self._find_centroid(embeddings, topics, probs, i)
        return self.centroids

    def weighted_mean(
        self, X: np.ndarray, weights: Sequence[Union[int, float]]
    ) -> np.ndarray:
        return np.dot(X.T, weights) / np.sum(weights)

    def _find_centroid(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        probs: np.ndarray,
        target_topic: int,
    ) -> np.ndarray:
        """
        Arguments:
            embeddings: 2d with dimensions (num_documents, num_dimensions)
            topics: list of length num documents
            probs: np.array of length num_documents showing the probability of the assigned topic
            target_topic: the topic, we want to find the centroid for
        returns:
            The centroid for the cluster
        """
        # Filtering the embeddings
        filtered_embeddings = embeddings[topics == target_topic, :]
        filtered_probs = probs[topics == target_topic]

        # Calculating the centroid
        return self.weighted_mean(filtered_embeddings, filtered_probs)


class Clearsifier:
    def __init__(self, clf=None):
        self.clf = clf
