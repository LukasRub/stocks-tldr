import re
import os
import json
import logging
from itertools import product
from pathlib import Path
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed


PATH_TO_VECTORS = "data/test/doc2vec/vectors"
ARTICLE_VECTORS_DIR = "article_vectors"
ENTITY_VECTORS_DIR = "entity_vectors"
SIMILARITIES_FILE_NAME = "similarities.json"
LOCK = Lock()


def multiprocess_log(status_msg, lock=LOCK):
    with lock:
        logging.info("[PID:{}] {}".format(os.getpid(), status_msg))


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


def calculate_similarities(model_path):
    # Setting string representation for easier use across functions
    model = model_path.name

    # Setting path (no need to sort) and loading article vectors
    avs_path = list((model_path / ARTICLE_VECTORS_DIR).glob("*.npy"))[0]
    avs = np.load(avs_path)

    # Setting paths (sorted by entity) to entity vectors
    evs_paths = sorted(list((model_path / ENTITY_VECTORS_DIR).glob("*.npz")),
                       key=lambda x: str(x.name).split(".")[0])
    similarities_file_path = model_path / Path(SIMILARITIES_FILE_NAME)

    if similarities_file_path.exists():
        status_msg = f"Similarities vectors for {model} already exist."
        multiprocess_log(status_msg)
    
    else:
        evs = dict()
        for evs_path in evs_paths:
            entity = evs_path.name.split(".")[0]
            npz_file = np.load(evs_path)
            evs[entity] = {k:v for k,v in npz_file.items()}
        
        model_similarities = dict()
        for article_index in range(len(avs)):
            article_vectors =  avs[article_index, :, :]

            article_similarities = dict()
            for entity, entity_contents in evs.items():
                entity_similarities = dict()
                for kind, entity_vectors in entity_contents.items():
                    if kind == "evs_child":
                        entity_vectors = reshape_to_2d(entity_vectors)

                    product_ = cartesian_product(article_vectors, entity_vectors)
                    similarity = get_similarities(product_)

                    entity_similarities[kind] = similarity
                
                str_repr = "\t".join([k + ":" + str(v) 
                                    for k, v in entity_similarities.items()])
                status_msg = f"{model}\t{article_index}\t{entity}\t{str_repr}"
                multiprocess_log(status_msg)

                article_similarities[entity] = entity_similarities

            model_similarities[article_index] = article_similarities
        
        status_msg = f"Saving {model} similarity scores to {similarities_file_path}..."
        multiprocess_log(status_msg)
        with open(similarities_file_path, "w") as fp:
            json.dump(model_similarities, fp=fp, ensure_ascii=False)


def main():
    # Setting up paths
    model_paths = sorted(list(Path(PATH_TO_VECTORS).glob("*/*")),
                         key=lambda x: int(re.search("w\d{1}", str(x)).group()[1:]))

    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", verbose=40, 
                            batch_size=1, max_nbytes=None, mmap_mode=None)
    parallelized(delayed(calculate_similarities)(model_path) for model_path in model_paths)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"Available CPU cores: {cpu_count()}")
    main()