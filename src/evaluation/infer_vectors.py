import logging
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION


QRELS_LINE_FMT = "{article_id} Q0 {ticker} {rank} {score}"
DOC2VEC_MODEL_TEST_DATA_DIR = "data/test/word2vec/models"
DOC2VEC_ARTICLE_TEST_DATA_DIR = "data/test/word2vec/articles"

class TqdmLoggingHandler(logging.StreamHandler):
    """Logging stream-like handler to make sure logging works with tqdm
    https://github.com/tqdm/tqdm/issues/193
    """

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def infer_or_load_entity_vectors(model, vectors_per_entity, stocks, epochs,
                                 test_dir=DOC2VEC_MODEL_TEST_DATA_DIR, 
                                 strategy="summary"):
    model_repr = get_model_str_repr(model)
    model_dir = Path(test_dir) / model_repr

    ev_dir = model_dir / "entity_vectors" / strategy
    ev_dir.mkdir(parents=True, exist_ok=True)

    all_evs = dict()

    total = 0
    for stock in stocks:
        total += vectors_per_entity + vectors_per_entity * len(stock["entities"])
    
    with tqdm(total=total) as pgb:
        for stock in stocks:
            evs_stock = dict()

            # Set up filename for loading or saving if vectors do not yet exist
            ev_filename = "{ticker}.vpe{vpe}.ep{ep}.ev.vectors.npy".format(
                ticker=stock["ticker"], vpe=vectors_per_entity, ep=epochs
            )
            ev_path = ev_dir / ev_filename

            if Path(str(ev_path) + ".npz").exists() and Path(str(ev_path) + ".npz").is_file():
                # If entity vectors are found, load them and store each in a dictionary
                ev_path = Path(str(ev_path) + ".npz")

                logging.info(f"{ev_path.name} found in the appropriate article vectors directory...")
                logging.info(f"Loading {ev_path}...")

                npzfile = np.load(ev_path)

                for kw_name in npzfile.files:
                    evs_stock[kw_name] = npzfile[kw_name]
    
                all_evs[stock["ticker"]] = evs_stock
            else:
                # Inferring entity vectors for main entity
                evs_main = np.empty((vectors_per_entity, model.vector_size))
                for i in range(vectors_per_entity):
                    logging.info("Inferring vector #{i} for entity {name}...".format(
                        i=i+1, name=stock["ticker"]
                    ))
                    if strategy == "summary":
                        evs_main[i, :] = model.infer_vector(stock["summary"], 
                                                           epochs=epochs)
                    else:
                        evs_main[i, :] = model.infer_vector(stock["content"], 
                                                           epochs=epochs)
                    pgb.update(1)
                if strategy == "full" and (model_dir / "entity_vectors" / "summary").exists():
                    target_path = Path(model_dir / "entity_vectors" / "summary" / (ev_filename + ".npz"))
                    logging.info(f"Borrowing child entity vectors from {target_path}...")
                    npzfile = np.load(target_path)
                    evs_child = npzfile["evs_child"]
                    logging.info(f"{target_path} loaded, contains ({evs_child.shape[0]},{evs_child.shape[1]}) vectors")
                    pgb.update(evs_child.shape[0]*evs_child.shape[1])

                else:
                    # Inferring entity vectors for child entities
                    evs_child = np.empty((len(stock["entities"]), 
                                        vectors_per_entity,
                                        model.vector_size))
                    for i, child_entity in enumerate(stock["entities"]):
                        for j in range(vectors_per_entity):
                            logging.info("Inferring vector #{j} for entity {name} (child of {parent}, total {i}/{total})...".format(
                                j=j+1, name=child_entity["name"], parent=stock["ticker"],
                                i=i, total=len(stock["entities"])
                            ))
                            evs_child[i, j, :] = model.infer_vector(
                                child_entity["summary"], 
                                epochs=epochs
                            )
                            pgb.update(1)
                
                logging.info(f"Saving entity vectors to {ev_path}...")
                np.savez(ev_path, evs_main=evs_main, evs_child=evs_child)

                evs_stock["evs_main"] = evs_main
                evs_stock["evs_child"] = evs_child

                all_evs[stock["ticker"]] = evs_stock

    return all_evs


def infer_or_load_article_vectors(model, vectors_per_article, articles, epochs, 
                                  test_dir=DOC2VEC_MODEL_TEST_DATA_DIR):
    model_repr = get_model_str_repr(model)
    model_dir = Path(test_dir) / model_repr

    av_dir = model_dir / "article_vectors"
    av_dir.mkdir(parents=True, exist_ok=True)

    av_filename = "vpa{vpa}.ep{ep}.av.vectors.npy".format(vpa=vectors_per_article, 
                                                          ep=epochs)
    av_path = av_dir / av_filename

    if av_path.exists() and av_path.is_file():
        logging.info(f"{av_path.name} found in the appropriate article vectors directory...")
        logging.info(f"Loading {av_path}...")
        avs = np.load(av_path)
    else:
        avs = np.empty((len(articles), vectors_per_article, model.vector_size))
        with tqdm(total=len(articles)*vectors_per_article) as pbar:
            for i, article in enumerate(articles):
                for j in range(vectors_per_article):
                    logging.info("Inferring vector #{j} for article {name}, (total {i}/{total})".format(
                        j=j+1, name=article["name"], i=i+1, total=len(articles)
                    ))
                    avs[i, j, :] = model.infer_vector(article["tokenized"], epochs=epochs)
                    pbar.update(1)

        logging.info(f"Saving article vectors to {av_path}...")
        np.save(av_path, avs)

        return avs   


def get_model_str_repr(model, epochs_param_str_marker="ep"):
    model_str = str(model).replace("/", "-")
    epochs = model.epochs
    return f"{model_str[:-1]},{epochs_param_str_marker}{epochs})"


def create_model_test_data_directories(model_repr, test_dir=DOC2VEC_MODEL_TEST_DATA_DIR):
    # Create a parent test data subdirectory if it doesn't already exist
    test_data_parent_dir = Path(test_dir)
    test_data_parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test data subdirectory specific to the model if it doesn't already exist
    model_test_data_dir = test_data_parent_dir / model_repr
    model_test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create a test data subdirectory specific to model article and entity vectors
    vectors_dir = model_test_data_dir / "article_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    vectors_dir = model_test_data_dir / "entity_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Create a runfiles subdirectory specific to the model runfiles if it doesn't exist
    runfiles_dir = model_test_data_dir / "runfiles"
    runfiles_dir.mkdir(parents=True, exist_ok=True)

    # Create a runfiles subdirectory specific to the model logs if it doesn't exist
    runfiles_dir = model_test_data_dir / "logs"
    runfiles_dir.mkdir(parents=True, exist_ok=True)

    # run_desc_str = "{model}_{run_params}".format(
    #     model=get_model_str_repr(model),
    #     run_params="_".join([k + ":" + v for k,v in run_params])
    # )

    return model_test_data_dir
    

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
    tokenized = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered


def read_and_process_articles(paths):
    articles = list()
    for path in paths:
        article = dict()

        with open(path, "r", encoding="utf8") as fp:
            full_text = fp.read()

        article["name"] = path.name
        article["full_text"] = full_text
        article["tokenized"] = tokenize(full_text)

        articles.append(article)

    return articles


def main(vectors_per_article=100, vectors_per_entity=50):
    # Read and process articles
    article_paths = sorted(Path("./data/test/word2vec/articles").glob("*.txt"),
                           key=lambda x: int(x.name.split(".")[0]))
    articles = read_and_process_articles(article_paths)

    # Read and process stocks file
    stocks_path = Path("data/temp/revolut.top50.wiki.UPDT.jsonl")
    stocks = read_jsonl_file(stocks_path, processing_func=filter_stocks)

    # Specify which model (directories) to use
    model_paths_str = ["models/Doc2Vec(dm-c,d150,n20,w3,mc5,s1e-05,t4,ep40)"]

    # Convert string paths to PosixPaths and join paths directly to the model files
    model_paths = list()
    for path_str in model_paths_str:
        path = Path(path_str)
        path = path.joinpath(path.name)
        model_paths.append(path)

    # Define ParamGrid
    param_grid = ParameterGrid({
        "strategy":["summary", "full"]
    })

    for path in model_paths:

        # Load model and create appropriate folders in data/test/doc2vec/models
        model = Doc2Vec.load(str(path))
        model_repr = get_model_str_repr(model)
        create_model_test_data_directories(model_repr)
      
        # Load or infer vectors for articles
        av = infer_or_load_article_vectors(model, vectors_per_article, articles, 
                                           epochs=model.epochs)

        for params in list(param_grid):
            # Load or infer vectors for entities
            ev = infer_or_load_entity_vectors(model, vectors_per_entity, stocks,
                                              strategy=params["strategy"], 
                                              epochs=model.epochs)
            for stock, evs in ev.items():
                print(f"Stock {stock}: evs_main:{evs['evs_main'].shape}, evs_child:{evs['evs_child'].shape}")

            # Setting up file logger to file
            # file_handler = logging.FileHandler(model_test_data_dir / "logs" / f"{model_repr}.log")
            # logging_fmt = logging.getLogger().handlers[0].formatter
            # file_handler.setFormatter(logging_fmt)
            # logging.getLogger().addHandler(file_handler)

            # Setting up Tqdm logger
            # tqdm_handler = TqdmLoggingHandler()
            # logging.getLogger().addHandler(tqdm_handler)

            

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO, handlers=[TqdmLoggingHandler()])
    logging.info(f"gensim FAST_VERSION: {FAST_VERSION}")
    main(vectors_per_article=100, vectors_per_entity=10)