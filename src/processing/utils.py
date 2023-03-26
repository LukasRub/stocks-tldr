import os
import logging
import json
from pathlib import Path

import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class plotting():
    rc = {
        "axes.grid": False,
        "axes.edgecolor": "darkslategrey",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "xtick.color": "darkslategrey",
        "xtick.labelsize": "medium",
        "xtick.labelcolor": "darkslategrey",
        "xtick.bottom": True,
        "xtick.major.size": 3,
        "ytick.color": "darkslategrey",
        "ytick.major.size": 3,
        "ytick.left": True,
        "ytick.labelsize": "medium"
    }
    discrete_palette = sns.color_palette("tab10")
    # signed_discrete_palette = sns.color_palette("Spectral", n_colors=5) 
    signed_discrete_palette = sns.color_palette("RdYlGn", n_colors=5) 
    signed_palette = sns.color_palette("BrBG")
    unsigned_palette = sns.color_palette("viridis")
    



def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokenized = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered


def get_doc2vec_model_str_repr(model, epochs_param_str_marker="ep"):
    model_str = str(model).replace("/", "-")
    epochs = model.epochs
    return f"{model_str[:-1]},{epochs_param_str_marker}{epochs})"


def multiprocess_log(status_msg, lock=None):
    if lock:
        with lock:
            logging.info("[PID: {}] {}".format(os.getpid(), status_msg))
    else:
        logging.info(status_msg) 


def read_json(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(path, "r") as fp:
        return json.load(fp)


def read_jsonl(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(path, "r", encoding="utf8") as fp:
        return [json.loads(line) for line in fp]
