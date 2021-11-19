import os
import json
import logging
from time import time
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION

from utils import tokenize, multiprocess_log
from utils import get_doc2vec_model_str_repr as model_str_repr


PATH_TO_ARTICLES = "data/processed/articles/validation/texts"
PATH_TO_ENTITIES = "data/processed/entities"
MODEL_DIRECTORIES = ["models/Doc2Vec(dm-c,d100,n20,w2,mc5,s1e-05,t4,ep40)",
                     "models/Doc2Vec(dm-c,d50,n20,w2,mc5,s1e-05,t4,ep20)",
                     "models/Doc2Vec(dm-c,d100,n20,w3,mc5,s1e-05,t4,ep40)",
                     "models/Doc2Vec(dm-c,d100,n20,w4,mc5,s1e-05,t4,ep20)",
                     "models/Doc2Vec(dm-c,d75,n20,w1,mc5,s1e-05,t4,ep20)"]

PATH_TO_VECTORS = "data/test/user_study/vectors"
ARTICLE_VECTORS_DIR = "article_vectors"
ENTITY_VECTORS_DIR = "entity_vectors"

ARTICLE_VECTORS_FILENAME_FMT = "validation.vpa{vecs_per_article}.ep{epochs}.av.vectors.npy"
ENTITY_VECTORS_FILENAME_FMT = "{ticker_sybmol}.vpe{vecs_per_ent}.ep{epochs}.ev.vectors.npy"

LOCK = Lock()


def infer_single_article_vectors(model, vectors_per_article, article_path, lock=LOCK):
    # Read and tokenize article text
    with open(article_path, "r", encoding="utf-8") as fp:
        article_txt = fp.read()

    article_tokenized = tokenize(article_txt)

    # Initializing empty arrays for article vectors
    avs = np.empty((vectors_per_article, model.vector_size))

    # Displaying a status message
    status_msg = f"Inferring article {article_path.name} vectors..."
    multiprocess_log(status_msg, lock)

    t0 = time()

    # Inferring a specified number of article vectors
    for i in range(vectors_per_article):
        avs[i, :] = model.infer_vector(article_tokenized)
    
    t1 = time() - t0
    
    # Displaying a status message 
    fmt = (article_path.name, t1)
    status_msg = "Inferring vectors for article {0} finished, took {1:.0f} seconds"
    multiprocess_log(status_msg.format(*fmt), lock)

    return (article_path.name, avs)


def infer_single_entity_vectors(model, vectors_per_entity, entity_path, lock=LOCK):
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
        multiprocess_log(status_msg, lock)
        return
    elif not evs_path_.parent.exists(): 
        with lock:
            evs_path.parent.mkdir(parents=True, exist_ok=True)
        status_msg = f"Creating a directory at {evs_path.parent}..."
        multiprocess_log(status_msg, lock)

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
    multiprocess_log(status_msg, lock)

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
    multiprocess_log(status_msg.format(*fmt), lock)

    # Saving vectors to disk
    kwargs = dict(evs_full=evs_full, evs_summary=evs_summary, evs_child=evs_child)
    np.savez(evs_path, **kwargs)


def infer_article_vectors(model, vectors_per_article, lock=LOCK, 
                          article_paths=str(PATH_TO_ARTICLES)):           
    # Setting up model str representation and paths
    model_name = model_str_repr(model)
    article_paths = sorted(Path(article_paths).glob("*.txt"), 
                           key=lambda x: int(x.name.split(".")[0][-1]))
    
    fmt = dict(vecs_per_article=vectors_per_article, epochs=model.epochs)
    avs_path = (Path(PATH_TO_VECTORS) 
                    / Path(model_name)  
                    / Path(ARTICLE_VECTORS_DIR)
                    / Path(ARTICLE_VECTORS_FILENAME_FMT.format(**fmt)))

    # Skip inferring vectors for this entity if they were computed before
    if avs_path.exists() and avs_path.is_file():
        status_msg = f"Article vectors for {model_name} already exist."
        multiprocess_log(status_msg, lock)
        return
    else:
        with lock:
            avs_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = (model_name, avs_path.parent)
        status_msg = "Article vectors for {} don't exist. Creating a directory at {}..."
        multiprocess_log(status_msg.format(*fmt), lock)

    # Freeze `model` arg of `infer_single_article_vectors` func
    infer_single_article_vectors_ = partial(infer_single_article_vectors, 
                                            model, vectors_per_article)

    # Parallelize  partial `infer_single_article_vectors_` func computation
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=40, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    results = parallelized(delayed(infer_single_article_vectors_)
                           (article_path) for article_path in article_paths)
    
    # Initializing empty array for article vectors
    avs = np.empty((len(article_paths), vectors_per_article, model.vector_size))

    # Validate the order of the results and assign vectors
    for article_id, (article_name, article_vectors) in enumerate(results):
        assert article_id == int(article_name.split(".")[0][-1])
        avs[article_id, :, :] = article_vectors
    
    np.save(avs_path, avs)


def infer_entity_vectors(model, vectors_per_entity, lock=LOCK, 
                        entity_paths=str(PATH_TO_ENTITIES)):
    # Setting up paths
    entity_paths = sorted(Path(entity_paths).glob("*.json"),
                          key=lambda x: x.name.split(".")[0])

    # Freeze `model` arg of `infer_single_entity_vectors` func
    infer_single_entity_vectors_ = partial(infer_single_entity_vectors, 
                                           model, vectors_per_entity)

    # Parallelize  partial `infer_single_entity_vectors_` func computation
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=40, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    parallelized(delayed(infer_single_entity_vectors_)
                            (entity_path) for entity_path in entity_paths)


def infer_model_vectors(model_path, vectors_per_article=100, 
         vectors_per_entity=10, lock=LOCK):
    assert model_path.exists() and model_path.is_file()
    
    # Load model
    model = Doc2Vec.load(model_path.as_posix())

    # Infer article vectors
    infer_article_vectors(model, vectors_per_article)

    # Infer entity vectors
    # infer_entity_vectors(model, vectors_per_entity)    


def main(model_directories=MODEL_DIRECTORIES, vectors_per_article=100, 
         vectors_per_entity=10, lock=LOCK):
    
    model_directory_paths = [Path(model_dir) for model_dir in model_directories]
    model_paths = [model_dir / model_dir.name for model_dir in model_directory_paths]
    
    for model_path in tqdm(model_paths):
        infer_model_vectors(model_path, vectors_per_article=100, 
                            vectors_per_entity=10)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"gensim FAST_VERSION: {FAST_VERSION}")
    logging.info(f"Available CPU cores: {os.cpu_count()}")
    main()