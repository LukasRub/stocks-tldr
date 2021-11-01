import os
import json
import logging
from time import time
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION

from utils import tokenize, multiproccess_log
from utils import get_doc2vec_model_str_repr as model_str_repr


PATH_TO_ARTICLES = "data/processed/articles/test"
PATH_TO_ENTITIES = "data/processed/entities"
MODEL_DIR = "models/Doc2Vec(dm-c,d100,n30,w2,mc2,s0.0001,t16,ep40)"

PATH_TO_VECTORS = "data/test/user_study"
ARTICLE_VECTORS_DIR = "article_vectors"
ENTITY_VECTORS_DIR = "entity_vectors"

ARTICLE_VECTORS_FILENAME_FMT = "vpa{vecs_per_article}.ep{epochs}.av.vectors.npy"
ENTITY_VECTORS_FILENAME_FMT = "{ticker_sybmol}.vpe{vecs_per_ent}.ep{epochs}.ev.vectors.npy"

LOCK = Lock()


def infer_single_article_vectors(model, article_path, lock=LOCK, 
                                 vectors_per_article=100):
    # Read and tokenize article text
    with open(article_path, "r", encoding="utf-8") as fp:
        article_txt = fp.read()

    article_tokenized = tokenize(article_txt)

    # Initializing empty arrays for article vectors
    avs = np.empty((vectors_per_article, model.vector_size))

    # Displaying a status message
    status_msg = f"Inferring article {article_path.name} vectors..."
    multiproccess_log(status_msg, lock)

    t0 = time()

    # Inferring a specified number of article vectors
    for i in range(vectors_per_article):
        avs[i, :] = model.infer_vector(article_tokenized)
    
    t1 = time() - t0
    
    # Displaying a status message 
    fmt = (article_path.name, t1)
    status_msg = "Inferring vectors for article {0} finished, took {1:.0f} seconds"
    multiproccess_log(status_msg.format(*fmt), lock)

    return (article_path.name, avs)


def infer_single_entity_vectors(model, entity_path, lock=LOCK, vectors_per_entity=10):
    # Setting up representations for the entity and model in question
    entity_ticker = entity_path.name.split(".")[0]
    model_name = model_str_repr(model)

    # Setting up path for entity vectors
    fmt = dict(ticker_sybmol=entity_ticker, vecs_per_ent=vectors_per_entity,
               epochs=model.epochs)

    evs_path = (Path(PATH_TO_VECTORS) 
                    / Path(model_name)  
                    / Path(ENTITY_VECTORS_DIR)
                    / Path(ENTITY_VECTORS_FILENAME_FMT.format(**fmt)))
    
    # Skip inferring vectors for this entity if they were computed before
    evs_path_ = Path(evs_path.as_posix() + ".npz")
    if evs_path_.exists() and evs_path_.is_file():
        status_msg = f"Entity {entity_ticker} vectors for {model_name} already exist."
        multiproccess_log(status_msg, lock)
        return

    # Read entity file
    with open(entity_path, "r", encoding="utf-8") as fp:
        entity = json.load(fp)
    
    # Initializing empty arrays for entity vectors
    evs_full = np.empty((vectors_per_entity, model.vector_size))
    evs_summary = np.empty((vectors_per_entity, model.vector_size))
    evs_child = np.empty((len(entity["child_entities"]), vectors_per_entity, 
                          model.vector_size))

    # Displaying a status message
    status_msg = f"Inferring entity {entity_ticker} vectors..."
    multiproccess_log(status_msg, lock)

    t0 = time()

    # Inferring vectors for the parent entity
    content = entity["wiki_page_content"]
    for i in range(vectors_per_entity):
        evs_full[i, :] = model.infer_vector(content["full_text"])
        evs_summary[i, :] = model.infer_vector(content["summary"])
                
    # Inferring vectors for child entities
    for i, child_entity in enumerate(entity["child_entities"]):
        for j in range(vectors_per_entity):
            evs_child[i, j, :] = model.infer_vector(child_entity["summary"])

    t1 = time() - t0

    # Displaying a status message 
    fmt = (entity_ticker, t1, evs_path)
    status_msg = "Inferring vectors for entity {0} finished, took {1:.0f} seconds, saving to {2}..."
    multiproccess_log(status_msg.format(*fmt), lock)

    # Saving vectors to disk
    kwargs = dict(evs_full=evs_full, evs_summary=evs_summary, evs_child=evs_child)
    np.savez(evs_path, **kwargs)


def infer_article_vectors(model, article_paths=str(PATH_TO_ARTICLES), 
                          lock=LOCK, vectors_per_article=100):            
    # Setting up model str representation and paths
    model_name = model_str_repr(model)
    article_paths = sorted(Path(article_paths).glob("*.txt"), 
                           key=lambda x: int(x.name.split(".")[0]))
    
    fmt = dict(vecs_per_article=vectors_per_article, epochs=model.epochs)
    avs_path = (Path(PATH_TO_VECTORS) 
                    / Path(model_name)  
                    / Path(ARTICLE_VECTORS_DIR)
                    / Path(ARTICLE_VECTORS_FILENAME_FMT.format(**fmt)))

    # Skip inferring vectors for this entity if they were computed before
    if avs_path.exists() and avs_path.is_file():
        status_msg = f"Article vectors for {model_name} already exist."
        multiproccess_log(status_msg, lock)
        return

    # Freeze `model` arg of `infer_single_article_vectors` func
    infer_single_article_vectors_ = partial(infer_single_article_vectors, model)

    # Parallelize  partial `infer_single_article_vectors_` func computation
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=40, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    results = parallelized(delayed(infer_single_article_vectors_)
                           (article_path) for article_path in article_paths)
    
    # Validate the order of the results:
    for article_id, (article_name, article_vectors) in enumerate(results):
        assert article_id == int(article_name.split(".")[0])
        print(article_id, article_name, article_vectors.shape)


def infer_entity_vectors(model, entity_paths=str(PATH_TO_ENTITIES), lock=LOCK, 
                         vectors_per_entity=10):
    # Setting up paths
    entity_paths = sorted(Path(entity_paths).glob("*.json"),
                          key=lambda x: x.name.split(".")[0])

    # Freeze `model` arg of `infer_single_entity_vectors` func
    infer_single_entity_vectors_ = partial(infer_single_entity_vectors, model)

    # Parallelize  partial `infer_single_entity_vectors_` func computation
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=40, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    parallelized(delayed(infer_single_entity_vectors_)
                            (entity_path) for entity_path in entity_paths)


def main(model_dir=str(MODEL_DIR), vectors_per_article=100, 
         vectors_per_entity=10, lock=LOCK):

    # Setting up model path
    model_path = Path(model_dir) / Path(model_dir).name
    assert model_path.exists() and model_path.is_file()
    
    # Load model
    model = Doc2Vec.load(model_path.as_posix())

    # Infer article vectors
    # infer_article_vectors(model, vectors_per_article=vectors_per_article)

    # Infer entity vectors
    infer_entity_vectors(model, vectors_per_entity=vectors_per_entity)    


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"gensim FAST_VERSION: {FAST_VERSION}")
    logging.info(f"Available CPU cores: {os.cpu_count()}")
    main()