"""
Clean the twitter data for embedding
"""
from os import replace
from pathlib import Path
import pandas as pd
import re
import numpy as np
import functools

DATA_DIR = Path("./output")


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)


def find_file(path, pattern="*"):
    return next(path.glob(pattern))


def remove_pattern(text_col: pd.Series, pattern: str) -> pd.Series:
    return text_col.str.replace(pattern, "")


def remove_usernames(text_col: pd.Series) -> pd.Series:
    pattern = "@\w+"
    return remove_pattern(text_col, pattern)


def remove_urls(text_col: pd.Series) -> pd.Series:
    pattern = "https?://\w+\\.\w+/\w+"
    return remove_pattern(text_col, pattern)


def clean_col(col, *functions):
    clean_pipeline = compose(*functions)
    return clean_pipeline(col)


def clean_textcol(text_col):
    return clean_col(text_col, remove_urls, remove_usernames)


df = pd.read_csv(
    find_file(DATA_DIR, pattern="*training*"), encoding="latin-1", header=None
)
df.columns = ["Sentiment", "id", "Date", "Query", "User", "Tweet"]

df["cleantext"] = clean_textcol(df["Tweet"])


test_name = "@switchfoot http://twitpic.com/2y1zl - Awww"
re.sub("https?://\w+\\.\w+/\w+", "", test_name)