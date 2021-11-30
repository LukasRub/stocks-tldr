import re
import json
import logging
import importlib
from itertools import product
from pathlib import Path
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed

spec = importlib.util.spec_from_file_location("utils", "src/processing/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("similarity", "src/processing/compute_similarities.py")
similarity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similarity)


PATH_TO_VECTORS = "data/test/trec_eval/vectors"
SIMILARITIES_FILE_NAME = "similarities.json"
LOCK = Lock()


def calculate_similarities(model_path, lock=LOCK):
    # Setting string representation for easier use across functions
    model = model_path.name
    model_epochs = re.search("ep\d+", model_path.name).group()[2:]

    # Setting path (no need to sort) and loading article vectors
    avs_path = model_path  / (similarity.ARTICLE_VECTORS_PATH_FMT
                                .format(set_prefix="", epochs=model_epochs))
    avs = np.load(avs_path)

    # Setting paths (sorted by entity) to entity vectors, sorted by entity ticker symbols
    evs_paths = sorted(list((model_path / similarity.ENTITY_VECTORS_DIR).glob("*.npz")),
                       key=lambda x: str(x.name).split(".")[0])
    similarities_file_path = model_path / Path(SIMILARITIES_FILE_NAME)

    if similarities_file_path.exists():
        status_msg = f"Similarities vectors for {model} already exist."
        utils.multiprocess_log(status_msg, lock)
        return
    
    # Loading in entity vectors
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
                    entity_vectors = similarity.reshape_to_2d(entity_vectors)

                product_ = similarity.cartesian_product(article_vectors, 
                                                        entity_vectors)
                similarity_ = similarity.get_similarities(product_)
                entity_similarities[kind] = similarity_
            
            # Printing out a status message
            str_repr = "\t".join([k + ":" + str(v) 
                                for k, v in entity_similarities.items()])
            status_msg = f"{model}\t{article_index}\t{entity}\t{str_repr}"
            utils.multiprocess_log(status_msg)

            article_similarities[entity] = entity_similarities

        model_similarities[article_index] = article_similarities
    
    # Writing results to disk
    status_msg = f"Saving {model} similarity scores to {similarities_file_path}..."
    utils.multiprocess_log(status_msg, lock=LOCK)
    with open(similarities_file_path, "w") as fp:
        json.dump(model_similarities, fp=fp, ensure_ascii=False, indent="\t")


def main():
    # Setting up paths, sorted by model window sizes
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