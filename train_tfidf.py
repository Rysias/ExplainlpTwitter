import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np

nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

tfidfconverter = TfidfVectorizer(
    max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words("english")
)


DATA_DIR = Path("./output")

train_df = pd.read_csv(DATA_DIR / "clean_tweets.csv")


fullX = train_df["cleantext"]
fullY = train_df["Sentiment"].values
X_train, X_test, Y_train, Y_test = train_test_split(
    fullX, fullY, test_size=10000, random_state=42
)
grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)


pipeline = make_pipeline(tfidfconverter, logreg, verbose=True)

pipeline.fit(X_train, Y_train)

tf_idf_preds = pipeline.predict(X_test)

pd.DataFrame({"y_true": Y_test, "preds": tf_idf_preds}).to_csv(
    DATA_DIR / "tf_idf_preds.csv", index=False
)

tfidfconverter.fit_transform(X_test)
