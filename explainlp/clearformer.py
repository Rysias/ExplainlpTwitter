import numpy as np
from bertopic import BERTopic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, Sequence, Iterable, Dict, List


class Clearformer(BaseEstimator, TransformerMixin):
    def __init__(self, topic_model: BERTopic) -> None:
        self.topic_model = topic_model
        self.nr_topics = self.topic_model.nr_topics

    def fit(self, X: np.ndarray):
        """Fits the centroids to the data
        X should have the following cols:
            0: topic_num
            1: probs
            2-n: embeddings
        """
        topics = X[:, 0]
        probs = X[:, 1]
        embeddings = X[:, 2:]

        self.centroids = np.zeros(
            (self.nr_topics, embeddings.shape[1])
        )  # Centroids need dimensions (number of topics, embedding-dimensionality)
        for i in range(self.nr_topics):
            self.centroids[i, :] += self._find_centroid(embeddings, topics, probs, i)

    def transform_row(self, embeddings: np.ndarray) -> np.ndarray:
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        featurizes multiple documents where each document is a numpy array of shape (n_paragraphs, embedding_dim)
        Also works if input is a (n_docs, embedding_dim) numpy array
        """
        try:
            return cosine_similarity(X, self.centroids)
        except ValueError as e:
            if "Incompatible dimension" in str(e):
                return cosine_similarity(X[:, 2:], self.centroids)
            else:
                raise

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
