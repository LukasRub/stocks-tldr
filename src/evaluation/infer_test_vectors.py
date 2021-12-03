import re
import logging
import importlib
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import numpy as np
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION

spec = importlib.util.spec_from_file_location("utils", "src/processing/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


PATH_TO_ENTITIES = "data/samples/sample_entities/revolut.2021-08-19.top49.entities.jsonl"
PATH_TO_MODELS = "models/"
PATH_TO_ARTICLES = "data/processed/articles/test/texts"
PATH_TO_VECTORS = "data/test/trec_eval/vectors"

ARTICLE_VECTORS_DIR = "article_vectors"
ENTITY_VECTORS_DIR = "entity_vectors"

LOCK = Lock()


def read_and_process_articles(article_paths):
    articles = list()
    for i, path in enumerate(article_paths):
        # Assering that loop index matches with the filename of the article
        assert i == int(path.stem)
        with open(path, "rt", encoding="utf-8") as fp:
            full_txt = fp.read()
        articles.append(utils.tokenize(full_txt))
    return articles


def infer_article_vectors(model, articles, lock=LOCK, vectors_per_article=100):
    
    # Initializing empty array for article vectors
    avs = np.empty((len(articles), vectors_per_article, model.vector_size))
    
    # Inferring vectors for articles
    for i, article in enumerate(articles):
        for j in range(vectors_per_article):
            status_msg = ("Inferring vector #{} for article {}.txt with model {}"
                            .format(j+1, i, utils.get_doc2vec_model_str_repr(model)))
            # Displaying status message
            utils.multiprocess_log(status_msg, lock)
            avs[i, j, :] = model.infer_vector(article)
    return avs


def infer_entity_vectors(model, entity, lock=LOCK, vectors_per_entity=10):
    
    # Initializing empty arrays for entity vectors
    evs_full = np.empty((vectors_per_entity, model.vector_size))
    evs_summary = np.empty((vectors_per_entity, model.vector_size))
    evs_child = np.empty((len(entity["entities"]), vectors_per_entity, 
                          model.vector_size))
    
    # Displaying status message
    status_msg = ("Inferring entity {} vectors with model {}"
                    .format(entity["ticker"], 
                            utils.get_doc2vec_model_str_repr(model)))
    utils.multiprocess_log(status_msg, lock)

    # Inferring vectors for the parent entity
    for i in range(vectors_per_entity):
        evs_full[i, :] = model.infer_vector(entity["content"])
        evs_summary[i, :] = model.infer_vector(entity["summary"])
        
    # Inferring vectors for child entities
    for i, child_entity in enumerate(entity["entities"]):
        for j in range(vectors_per_entity):
            evs_child[i, j, :] = model.infer_vector(child_entity["summary"])
    
    return evs_full, evs_summary, evs_child

            
def infer_vectors(articles, entities, model_path, lock=LOCK, vectors_per_entity=10, 
                  vectors_per_article=100):
    # Load model
    model = Doc2Vec.load(str(model_path))
    model_str_repr = utils.get_doc2vec_model_str_repr(model)
    
    # Set up directories for article vectors
    avs_filename = ("vpa{vpa}.ep{epochs}.av.vectors.npy"
                        .format(vpa=vectors_per_article, epochs=model.epochs))
    avs_path = (Path(PATH_TO_VECTORS) 
                   / Path(model_str_repr)  
                   / Path(ARTICLE_VECTORS_DIR)
                   / avs_filename)
    
    # Infer article vectors and save them to a file if they don't already exist
    if not avs_path.exists():
        avs_path.parent.mkdir(parents=True, exist_ok=True)
        avs = infer_article_vectors(model, articles, lock=lock, 
                                    vectors_per_article=vectors_per_article)
        utils.multiprocess_log(("Saving article vectors for {} to {}..."
                                .format(model_str_repr, avs_path)), lock)
        np.save(avs_path, avs)
    else:
        status_msg = f"Article vectors for {model_str_repr} already exist."
        utils.multiprocess_log(status_msg, lock)
    
    for entity in entities:
        # Set up directories for entity vectors
        entity_ticker = entity["ticker"]
        evs_filename = ("{ticker}.vpe{vpe}.ep{epochs}.ev.vectors.npy"
                            .format(ticker=entity_ticker, 
                                    vpe=vectors_per_entity, 
                                    epochs=model.epochs))
        evs_path = (Path(PATH_TO_VECTORS) 
                       / Path(model_str_repr)  
                       / Path(ENTITY_VECTORS_DIR)
                       / evs_filename)
        
        # Infer entity vectors and save them to a file if they don't already exist

        # NumPy multiple array archives have an additional .npz suffix 
        # that need to be accounted for
        if not Path(str(evs_path) + ".npz").exists(): 
            evs_path.parent.mkdir(parents=True, exist_ok=True)
            evs_full, evs_summary, evs_child = (
                infer_entity_vectors(model, entity, lock=lock, 
                                     vectors_per_entity=vectors_per_entity)
            )
            utils.multiprocess_log(("Saving entity {} vectors for {} to {}..."
                                     .format(entity_ticker, model_str_repr, evs_path)), lock)
            np.savez(evs_path, evs_full=evs_full, evs_summary=evs_summary, 
                     evs_child=evs_child)
        else:
            status_msg = f"Entity {entity_ticker} vectors for {model_str_repr} already exist."
            utils.multiprocess_log(status_msg, lock)
        

def main(vectors_per_article=100, vectors_per_entity=10):
    # Setting up paths to various data resources
    entities_path = Path(PATH_TO_ENTITIES)

    # Sorting out direct model files in order of their window sizes
    model_paths = sorted([path for path in Path(PATH_TO_MODELS).glob("*/*")
                         if path.suffix == ""], 
                         key=lambda x: int(re.search("w\d{1}", str(x)).group()[1:]))

    # Sorting out article text files in order in numerical order
    article_paths = sorted(list(Path(PATH_TO_ARTICLES).glob("*.txt")), 
                           key=lambda x: int(x.stem))

    # Reading in data
    entities = utils.read_jsonl(entities_path)
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