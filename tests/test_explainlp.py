import pytest
import numpy as np
import pickle
from pathlib import Path
from bertopic import BERTopic
from explainlp.explainlp import ClearSearch
from explainlp.clearsify import Clearsifier


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def topics():
    return np.random.choice(list(range(-1, 5)), 50)


@pytest.fixture(scope="session")
def real_embeddings():
    return read_pickle(Path("./tests/example_embeddings.pkl"))


@pytest.fixture(scope="session")
def real_docs():
    return read_pickle(Path("./tests/example_docs.pkl"))


@pytest.fixture(scope="session")
def embeddings():
    """random embeddings in the range (-5, 5)"""
    return (np.random.rand(50, 256) - 0.5) * 10


@pytest.fixture(scope="session")
def probs():
    return np.random.rand(50)


@pytest.fixture(scope="session")
def topic_model():
    return BERTopic.load(str(Path("../TransTopicXAI/models/topic_model")))


@pytest.fixture(scope="session")
def model(topic_model):
    return ClearSearch(
        topic_model=topic_model,
    )


def test_topic_model_created(model):
    assert model.topic_model is not None


def test_create_centroids(model, topics, probs, embeddings):
    model.calculate_centroids(topics, probs, embeddings)
    assert hasattr(model, "centroids")
    assert model.centroids.shape == (5, 256)
    assert model.centroids.max() < 5


def test_featurize_document(model, topics, probs, embeddings):
    model.calculate_centroids(topics, probs, embeddings)
    featurized_doc = model.transform(embeddings)  # Check up on the use of "transform"
    assert isinstance(featurized_doc, np.ndarray)
    assert featurized_doc.shape == (5,)
    assert np.all(
        (featurized_doc >= 0) & (featurized_doc <= 1)
    )  # within cosine similarity


def test_featurize_documents(model, embeddings):
    multiple_docs = {
        1: embeddings[:3, :],
        2: embeddings[4, :],
        3: embeddings[10:17, :],
        4: np.zeros((0, 256)),
    }
    featurized_docs = model.transform_many(multiple_docs.values())
    assert featurized_docs.shape == (4, 5)
    assert np.all(np.isnan(featurized_docs[3, :]))


def test_featurize_documents_array(model, embeddings):
    featurized_docs = model.transform_many(embeddings)
    assert featurized_docs.shape == (50, 5)


def test_get_topic_dict(model):
    real_topics = model.get_topics()
    assert isinstance(real_topics, dict)
    assert type(real_topics[0]) == list
    assert type(real_topics[0][0]) == str
    assert len(real_topics) == model.nr_topics


def test_init_clearsifier(topic_model):
    clearsifier = Clearsifier(topic_model=topic_model)