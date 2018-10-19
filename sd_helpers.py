from json import load
import os
import csv
import numpy as np
import nltk
import plotly
from plotly.graph_objs import Bar, Layout, Scatterpolar, Figure
from nltk.tokenize import TweetTokenizer, sent_tokenize
from sklearn.linear_model import LinearRegression
from nltk.stem import PorterStemmer


DICTIONARY_FILE = "data/stemmed_liwc.json"
LABELS_FILE = "data/labels.txt"

with open(LABELS_FILE, "r") as f:
    LABELS = [l[:-1] for l in list(f)]
with open(DICTIONARY_FILE, "r") as f:
    DIC = load(f)


def extract_features(input_string):
    """Extract feature scores from a string. Uses global variables LABELS and
    DIC from helpers.py

    Args:
        input_string (string): the string to be scored

    Returns:
        numpy array: the corresponding score values
    """

    tknzr = TweetTokenizer()
    ps = PorterStemmer()

    stemmed_and_tokenized = [ps.stem(k) for k in tknzr.tokenize(input_string)]
    words_per_sentence = len(stemmed_and_tokenized) / len(sent_tokenize(input_string))

    labels = tuple(LABELS)
    scores = [0] * len(labels)
    for word in stemmed_and_tokenized:
        if word in DIC:
            for label in DIC[word]:
                scores[labels.index(label)] += 1
    scores = [(s * 100) / len(stemmed_and_tokenized) for s in scores]

    # post processing add-ins
    scores[labels.index("WPS")] = words_per_sentence

    return np.array(scores)
