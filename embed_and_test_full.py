from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import argparse
from scipy.special import softmax

task = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
DATA_PATH = "./output/clean_tweets.csv"


def predict_text(model, tokenizer, text_arr: np.ndarray) -> np.ndarray:
    encoded_input = tokenizer(text_arr, padding=True, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0].detach().numpy()
    scores = softmax(scores, axis=1)
    return np.around(scores[:, 2])


def main(args):
    df = pd.read_csv(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis"
    )
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis"
    )
    sent_model = SentenceTransformer("finiteautomata/bertweet-base-sentiment-analysis")


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Embed Documents")
    my_parser.add_argument(
        "--data_path", type=str, help="Gives the path to the data file (a pickle)"
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
