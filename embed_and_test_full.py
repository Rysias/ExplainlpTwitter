from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import argparse
from pysentimiento import SentimentAnalyzer
from scipy.special import softmax
from pathlib import Path


def predict_text(model, tokenizer, text_arr: np.ndarray) -> np.ndarray:
    encoded_input = tokenizer(text_arr, padding=True, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0].detach().numpy()
    scores = softmax(scores, axis=1)
    return np.around(scores[:, 2])


def predict_sentiment(analyzer: SentimentAnalyzer, text: List[str]) -> List[int]:
    output = analyzer.predict(text)
    return [int(np.around(pred.probas["POS"])) for pred in output]


def create_emb_obj(embeddings: np.array, id_col: pd.Series) -> np.array:
    return np.hstack((id_col.values.reshape(-1, 1), embeddings))


def main(args):
    df = pd.read_csv(args.data_path)
    print("Loading models")
    analyzer = SentimentAnalyzer(lang="en")
    sent_model = SentenceTransformer("finiteautomata/bertweet-base-sentiment-analysis")
    print("Embedding docs")
    docs = df["cleantext"]
    embeddings = sent_model.encode(docs, show_progress_bar=True, batch_size=128)
    emb_obj = create_emb_obj(embeddings, df["id"])
    np.save(Path(args.embedding_path) / "embeddings.npy", emb_obj)
    print("predicting sentiment")
    preds = predict_sentiment(analyzer, docs.tolist())
    pred_df = df[["id", "Sentiment"]].assign(pred=preds)
    pred_df.to_csv(Path(args.embedding_path) / "big_preds.csv")
    print("all done!")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    my_parser.add_argument(
        "--data_path", type=str, help="Gives the path to the data file (a csv)"
    )
    my_parser.add_argument(
        "-emb",
        "--embedding_path",
        type=str,
        required=False,
        help="Gives the directory where to save embeddings",
    )
    args = my_parser.parse_args()
    main(args)
