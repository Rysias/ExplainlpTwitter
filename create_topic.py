from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import argparse
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def load_docs(data_path: Union[str, Path], text_col="cleantext") -> np.ndarray:
    df = pd.read_csv(Path(data_path))
    return df[text_col].values


def load_embeds(embed_path: Union[str, Path]):
    raw_embeds = np.load(embed_path)
    return raw_embeds[:, 1:]


def get_random_idx(
    col: Union[pd.Series, np.array], sample_size: Union[int, float] = 0.1
) -> np.array:
    all_idx = np.arange(len(col))
    if sample_size < 1:
        sample_size = int(len(col) * sample_size)
    return np.random.choice(all_idx, size=int(sample_size), replace=False)


def main(args):
    print("loadin data...")
    docs = load_docs(args.data_path)
    embeddings = load_embeds(args.embedding_path)
    np.random.seed(0)
    sample_idx = get_random_idx(docs, sample_size=args.data_size)
    small_docs = docs[sample_idx]
    small_embs = embeddings[sample_idx]
    print("bootin model...")
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words="english", min_df=100
    )
    topic_model = BERTopic(
        low_memory=True,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False,
        nr_topics=10,
    )
    print("fittin model...")
    topics, probs = topic_model.fit_transform(small_docs, small_embs)
    print("savin data...")
    preds_df = pd.DataFrame(
        list(zip(topics, probs, small_docs)), columns=["topic", "prob", "doc"]
    )
    preds_df.to_csv(Path(args.save_path) / "doc_topics.csv", index=False)
    np.save(Path(args.save_path) / "small_embs.npy", small_embs)
    print("savin model...")
    topic_model.save(
        str((Path(args.save_path) / "topic_model")), save_embedding_model=False
    )
    print("all done!")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    my_parser.add_argument(
        "--data_path", type=str, help="Gives the path to the data file (a csv)"
    )
    my_parser.add_argument(
        "--embedding_path",
        type=str,
        help="Gives the path to the embeddings (.npy)",
    )
    my_parser.add_argument(
        "--save_path",
        type=str,
        help="Path to directory to save stuff",
    )
    my_parser.add_argument(
        "--data_size",
        type=float,
        help="Size of the data to use (100k seems to be a hard limit)",
    )
    args = my_parser.parse_args()
    main(args)
