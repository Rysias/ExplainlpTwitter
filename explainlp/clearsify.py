"""
Extension of clearsearch for a classifier
"""
from typing import Dict
from explainlp.explainlp import ClearSearch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


class Clearsifier(ClearSearch):
    def __init__(
        self,
        topic_model,
        clf=LogisticRegression,
        preprocessor=Normalizer,
        topic_dict: Dict[int, str] = None,
    ):
        super().__init__(topic_model)
        self.clf = self._create_pipeline(preprocessor, clf)
        self.topic_dict = topic_dict if topic_dict else self._empty_topic_dict()

    def _create_pipeline(self, preprocessor, clf):
        return make_pipeline(preprocessor(), clf())

    def _empty_topic_dict(self):
        return {i: None for i in range(self.nr_topics)}

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        X_new = self.transform_many(X)
        return self.clf.predict(X_new)
