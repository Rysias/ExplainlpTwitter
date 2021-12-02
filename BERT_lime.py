from typing import Dict, List, Union
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import dill
from pathlib import Path
from pysentimiento import create_analyzer
from lime.lime_text import LimeTextExplainer
from pysentimiento.analyzer import AnalyzerOutput


def sort_sentiment(res: AnalyzerOutput) -> np.array:
    vals = [res.probas[k] for k in sentiment.id2label.values()]
    return np.array(vals).reshape(1, -1)


def list_to_arr(result: List[AnalyzerOutput]) -> np.ndarray:
    return np.vstack([sort_sentiment(out) for out in result])


def format_output(result: Union[List[AnalyzerOutput], AnalyzerOutput]) -> np.ndarray:
    try:
        return sort_sentiment(result)
    except AttributeError:
        return list_to_arr(result)


def dict_to_arr(dct: dict) -> np.ndarray:
    n_feats = len(dct.values())
    return np.array(list(dct.values())).reshape((-1, n_feats))


def predict_pos_proba(sentence: str) -> np.ndarray:
    pred = sentiment.predict(sentence)
    return format_output(pred)


sentiment = create_analyzer("sentiment", lang="en")
sentence = ["I'm tweeting and I'm happy!", "I'm sad"]
output = sentiment.predict(sentence)

predict_pos_proba(sentence)
labels = list(sentiment.id2label.values())

list_to_arr(output)

explainer = LimeTextExplainer(class_names=labels)

explains = explainer.explain_instance(sentence[0], predict_pos_proba, num_features=3)

# Test on real data
test_dat = pd.read_csv(
    Path("./output/clean_tweets.csv"),
    header=0,
    skiprows=lambda i: i > 0 and random.random() > 0.01,
)

testy = test_dat.sample(1)
sentence = testy["cleantext"].tolist()[0]


top_label = np.argmax(predict_pos_proba(sentence))


explains = explainer.explain_instance(
    sentence, predict_pos_proba, num_features=5, labels=[top_label]
)
explains.as_list()
fig = explains.as_pyplot_figure()
plt.show()
