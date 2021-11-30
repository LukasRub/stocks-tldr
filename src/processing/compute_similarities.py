import re
import json
import logging
import importlib
from itertools import product
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed

# spec = importlib.util.spec_from_file_location("utils", "src/processing/utils.py")
# utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(utils)

MODEL_PATH = "data/test/user_study/vectors/Doc2Vec(dm-c,d100,n20,w4,mc5,s1e-05,t4,ep20)"
ARTICLE_VECTORS_PATH_FMT = "article_vectors/{set_prefix}vpa100.ep{epochs}.av.vectors.npy"
ENTITY_VECTORS_DIR = "entity_vectors"
SIMILARITIES_DIR = MODEL_PATH / Path("similarities/validation")
SIMILARITIES_FILE_NAME = "{ticker_symbol}.validation.similarities.json"

LOCK = Lock()

def cosine_similarity(a, e):
    return (np.einsum('ij,ij->i', a, e) 
            / (np.linalg.norm(a, axis=1) 
               * np.linalg.norm(e, axis=1)))


def get_similarities(arrary, normalized=True, mean=True):
    similarities = cosine_similarity(arrary[:, 0, :], arrary[:, 1, :])
    if normalized:
        similarities = (1 + similarities) / 2
    if mean:
        return similarities.mean()
    else:
        return similarities


def cartesian_product(a, b):
    cartesian_product = np.array([x for x in product(a, b)])
    return cartesian_product


def reshape_to_2d(array):
    return array.reshape(array.shape[0]*array.shape[1], array.shape[2])


def calculate_similarities(avs, evs_path): 
    entity = evs_path.name.split(".")[0]
    similarities_dir = Path(SIMILARITIES_DIR)
    similarities_file = SIMILARITIES_FILE_NAME.format(ticker_symbol=entity)
    similarities_path = similarities_dir / similarities_file 

    if similarities_path.exists() and similarities_path.is_file():
        status_msg = f"similarities for entity {entity} already exist"
        multiprocess_log(status_msg, LOCK)
        return
    else: similarities_dir.mkdir(exist_ok=True)

    # Loading in entity vectors
    npz_file = np.load(evs_path)
    evs_dict = {k:v for k,v in npz_file.items()}
    
    article_similarities = dict()
    for article_index in range(len(avs)):
        article_vectors =  avs[article_index, :, :]

        entity_similarities = dict()
        for kind, entity_vectors in evs_dict.items():
            if kind == "evs_child":
                entity_vectors = reshape_to_2d(entity_vectors)

            product_ = cartesian_product(article_vectors, entity_vectors)
            similarity = get_similarities(product_)

            entity_similarities[kind] = similarity
        
        # Printing out a status message
        sims_str_repr = "\t".join([k + ":" + str(v) 
                                   for k, v in entity_similarities.items()])
        status_msg = f"{article_index}\t{entity}\t{sims_str_repr}"
        multiprocess_log(status_msg, LOCK)

        article_similarities[article_index] = entity_similarities
        
    # Writing results to disk
    with open(similarities_path, "w", encoding="utf-8") as fp:
        json.dump(article_similarities, fp, indent="\t")

    return (entity, article_similarities)
    

def main():
    # Setting up paths
    model_path = Path(MODEL_PATH)
    model_epochs = re.search("ep\d+", model_path.name).group()[2:]
    avs_path = (
        model_path / ARTICLE_VECTORS_PATH_FMT.format(set_prefix="validation.",
                                                     epochs=model_epochs)
    )
    evs_paths = sorted(list((model_path / ENTITY_VECTORS_DIR).glob("*.npz")),
                       key=lambda x: str(x.name).split(".")[0])
    
    # Loading in article vectors
    avs = np.load(avs_path)

    # Freeze `avs` arg of `calculate_similarities` func
    calculate_similarities_ = partial(calculate_similarities, avs)

    # Parallelize  partial `infer_single_entity_vectors_` func computation
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=60, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    
    results = parallelized(delayed(calculate_similarities_)
                          (evs_path) for evs_path in evs_paths)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"Available CPU cores: {cpu_count()}")
    main()