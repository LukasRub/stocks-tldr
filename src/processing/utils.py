import os
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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