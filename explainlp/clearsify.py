"""
Extension of clearsearch for a classifier
"""
from explainlp.explainlp import ClearSearch


class Clearsifier(ClearSearch):
    def __init__(self, topic_model, clf=None):
        super().__init__(topic_model)
        self.clf = clf
