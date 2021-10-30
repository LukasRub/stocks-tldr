import os
import re
import json
import logging
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION


PATH_TO_ENTITIES = "data/samples/sample_entities/revolut.2021-08-19.top49.entities.jsonl"
PATH_TO_MODELS = "models/"
PATH_TO_ARTICLES = "data/processed/articles/test"
PATH_TO_VECTORS = "data/test/doc2vec/vectors"

ARTICLE_VECTORS_DIR = "article_vectors"
ENTITY_VECTORS_DIR = "entity_vectors"

LOCK = Lock()


def filter_stocks(stocks):
    return [s for s in stocks if s["wiki"] is not None]


def read_jsonl_file(filename, processing_func=None):
    objects = list()
    with open(filename, "r", encoding="utf8") as fp:
        for line in fp:
            obj = json.loads(line)
            objects.append(obj)
    if processing_func:
        objects = processing_func(objects)
    return objects


def tokenize(text):
    stop_words = set(stopwords.words("english"))
    tokenized = [token.lower() for token in word_tokenize(text) 
                 if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered


def read_and_process_articles(article_paths):
    articles = list()
    for i, path in enumerate(article_paths):
        
        # Assering that loop index matches with filename of the article
        assert i == int(path.stem)
        
        with open(path, "r", encoding="utf-8") as fp:
            full_txt = fp.read()
            
        articles.append(tokenize(full_txt))
    
    return articles


def get_model_str_repr(model, epochs_param_str_marker="ep"):
    model_str = str(model).replace("/", "-")
    epochs = model.epochs
    return f"{model_str[:-1]},{epochs_param_str_marker}{epochs})"


def multiproccess_print(status_msg, lock=LOCK):
    if lock:
        with lock:
            logging.info("[PID: {}] {}".format(os.getpid(), status_msg))
    else:
        logging.info(status_msg)


def infer_article_vectors(model, articles, lock=LOCK, vectors_per_article=100, 
                          epochs=20):
    
    # Initializing empty array for article vectors
    avs = np.empty((len(articles), vectors_per_article, model.vector_size))
    
    # Inferring vectors for articles
    for i, article in enumerate(articles):
        for j in range(vectors_per_article):
            status_msg = ("Inferring vector #{} for article {}.txt with model {}"
                              .format(j+1, i, get_model_str_repr(model)))
            multiproccess_print(status_msg, lock)  # Displaying status message
            avs[i, j, :] = model.infer_vector(article, epochs=epochs)
    
    return avs


def infer_entity_vectors(model, entity, lock=LOCK, vectors_per_entity=10, epochs=20):
    
    # Initializing empty arrays for entity vectors
    evs_full = np.empty((vectors_per_entity, model.vector_size))
    evs_summary = np.empty((vectors_per_entity, model.vector_size))
    evs_child = np.empty((len(entity["entities"]), vectors_per_entity, 
                          model.vector_size))
    
    # Displaying status message
    status_msg = ("Inferring entity {} vectors with model {}"
                    .format(entity["ticker"], get_model_str_repr(model)))
    multiproccess_print(status_msg, lock)

    # Inferring vectors for the parent entity
    for i in range(vectors_per_entity):
        evs_full[i, :] = model.infer_vector(entity["content"], epochs=epochs)
        evs_summary[i, :] = model.infer_vector(entity["summary"], epochs=epochs)
        
    # Inferring vectors for child entities
    for i, child_entity in enumerate(entity["entities"]):
        for j in range(vectors_per_entity):
            evs_child[i, j, :] = model.infer_vector(child_entity["summary"], 
                                                    epochs=epochs)
    
    return evs_full, evs_summary, evs_child

            
def infer_vectors(articles, entities, model_path, lock=LOCK, vectors_per_entity=10, 
                  vectors_per_article=100, epochs=20):
    # Load model
    model = Doc2Vec.load(str(model_path))
    model_str_repr = get_model_str_repr(model)
    
    # Set up directories for article vectors
    avs_filename = ("vpa{vpa}.ep{ep}.av.vectors.npy"
                        .format(vpa=vectors_per_article, ep=epochs))
    avs_path = (Path(PATH_TO_VECTORS) 
                   / Path(model_str_repr)  
                   / Path(ARTICLE_VECTORS_DIR)
                   / avs_filename)
    
    # Infer article vectors and save them to a file if they don't already exist
    if not avs_path.exists():
        avs_path.parent.mkdir(parents=True, exist_ok=True)
        avs = infer_article_vectors(model, articles, lock=lock, 
                                    vectors_per_article=vectors_per_article, 
                                    epochs=epochs)
        multiproccess_print(("Saving article vectors for {} to {}..."
                                .format(model_str_repr, avs_path)), lock)
        np.save(avs_path, avs)
    else:
        status_msg = f"Article vectors for {model_str_repr} already exist."
        multiproccess_print(status_msg, lock)
    
    for entity in entities:
        # Set up directories for entity vectors
        entity_ticker = entity["ticker"]
        evs_filename = ("{ticker}.vpe{vpe}.ep{ep}.ev.vectors.npy"
                            .format(ticker=entity_ticker, 
                                    vpe=vectors_per_entity, 
                                    ep=epochs))
        evs_path = (Path(PATH_TO_VECTORS) 
                       / Path(model_str_repr)  
                       / Path(ENTITY_VECTORS_DIR)
                       / evs_filename)
        
        # Infer entity vectors and save them to a file if they don't already exist
        if not Path(str(evs_path) + ".npz").exists(): # NumPy array archives  
                                                      # have a .npz suffix
            evs_path.parent.mkdir(parents=True, exist_ok=True)
            evs_full, evs_summary, evs_child = \
                infer_entity_vectors(model, entity, lock=lock, 
                                     vectors_per_entity=vectors_per_entity,
                                     epochs=epochs)
            multiproccess_print(("Saving entity {} vectors for {} to {}..."
                                     .format(entity_ticker, model_str_repr, evs_path)), lock)
            np.savez(evs_path, evs_full=evs_full, evs_summary=evs_summary, 
                     evs_child=evs_child)
        else:
            status_msg = f"Entity {entity_ticker} vectors for {model_str_repr} already exist."
            multiproccess_print(status_msg, lock)
        

def main(vectors_per_article=100, vectors_per_entity=50):
    # Setting up paths to various data resources
    entities_path = Path(PATH_TO_ENTITIES)
    # model_paths = sorted([path for path in Path(PATH_TO_MODELS).glob("*/*")
    #                      if path.suffix == ""], 
    #                      key=lambda x: int(re.search("w\d{1}", str(x)).group()[1:]))

    model_paths = [
        Path("models/Doc2Vec(dm-c,d100,n30,w1,mc2,s0.0001,t16,ep40)/Doc2Vec(dm-c,d100,n30,w1,mc2,s0.0001,t16,ep40)"),
        Path("models/Doc2Vec(dm-c,d100,n30,w2,mc2,s0.0001,t16,ep40)/Doc2Vec(dm-c,d100,n30,w2,mc2,s0.0001,t16,ep40)")
    ]

    article_paths = sorted(list(Path(PATH_TO_ARTICLES).glob("*.txt")), 
                           key=lambda x: int(x.stem))

    # Reading data
    entities = read_jsonl_file(entities_path, processing_func=filter_stocks)
    articles = read_and_process_articles(article_paths)

    # Main logic, parallelized
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=40, 
                            batch_size=1, max_nbytes=None, mmap_mode=None)
    partial_infer_vectors = partial(infer_vectors, articles, entities)
    parallelized(delayed(partial_infer_vectors)(model_path) for model_path in model_paths)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"gensim FAST_VERSION: {FAST_VERSION}")
    main(vectors_per_article=100, vectors_per_entity=10)