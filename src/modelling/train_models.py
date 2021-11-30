import json
import multiprocessing
import logging
import pickle
from pathlib import Path
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import ParameterGrid
from gensim import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, TaggedLineDocument, FAST_VERSION


class TaggedWaPoLineCorpus(TaggedLineDocument):
    def __init__(self, source):
        self.source = source

        if Path(self.source).is_file():
            logging.debug('single file given as source, rather than a directory of files')
            logging.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # Force code compatibility with list of files
        elif Path(self.source).is_dir():
            logging.info('reading directory %s', self.source)
            self.source = Path(self.source)
            self.input_files = [x for x in self.source.iterdir() if x.is_file()]
            self.input_files.sort()  # Makes sure it happens in filename order
        else:  # Not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logging.info('files read into TaggedWaPoLineCorpus:%s', 
                        '\n'.join([str(x) for x in self.input_files]))

    def __iter__(self):
        """Iterate through the files and the lines in the source."""
        for file_name in self.input_files:
            logging.info('reading file %s', file_name)
            with utils.open(file_name) as fp:
                for line in fp:
                    document = json.loads(line)
                    text = utils.to_unicode(document['plaintext'])
                    id = document['id']
                    yield TaggedDocument(tokenize(text), [id])


def get_model_str_repr(model, epochs_param_str_marker="ep"):
    model_str = str(model).replace("/", "-")
    epochs = model.epochs
    return f"{model_str[:-1]},{epochs_param_str_marker}{epochs})"


def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokenized = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered


def create_or_load_tagged_corpus(source, force_create=False, source_serialized="data/serialized/corpus"):
    serialized_corpus_exists = (Path(source_serialized).exists() and 
                                len([child for child in Path(source_serialized).glob("*.pkl")]) > 0)

    if force_create or not serialized_corpus_exists:
        # Streaming and tagging (as per gensim model training requirements) document corpus from disk 
        # requires a significantly long time - about 65 minutes on an Intel i5-8265U machine - 
        # - uncomment the follwing line if can afford to just put it in memory (~30GB)
        # tagged_corpus = list(TaggedWaPoLineCorpus(source))

        # Otherwise:
        tagged_corpus = TaggedWaPoLineCorpus(source)

        if not serialized_corpus_exists:
            # Serialize tagged corpus so that we could skip the tagging step in future uses
            Path(source_serialized).mkdir(exist_ok=True)
            timestamp = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
            filename = "WaPo_Tagged_corpus_{0}.pkl".format(timestamp)
            output_path = Path(source_serialized).joinpath(filename)
            logging.info("Serializing tagged corpus at {0}...".format(output_path))
            pickle.dump(tagged_corpus, open(output_path, "wb"), pickle.HIGHEST_PROTOCOL)

    elif serialized_corpus_exists:
        files = sorted([child for child in Path(source_serialized).glob("*.pkl")])
        if len(files) > 1:
            warn_msg = (("More than one file exists in {0} directory;" + 
                         "only the most recent one will be attempted to de-serialized")
                            .format(source_serialized))
            logging.warn(warn_msg)

        logging.info("De-serializing file at {0}...".format(files[-1]))
        tagged_corpus = pickle.load(open(files[-1], "rb"))

    return tagged_corpus


def main():
    documents = create_or_load_tagged_corpus("data/processed/articles/", force_create=True)

    common_params = dict(dm=1, dm_concat=1, hs=0, min_count=5,  
                         sample=1e-5, workers=multiprocessing.cpu_count())
  
    param_grid = [
        # PV-DM model with negative sampling with concatenation of context word vectors. 
        {
            "vector_size": [100],   # "vector_size": [100], 
            "epochs": [40],         # "epochs": [40],
            "window": [2],          # "window": [1,2], 
            "negative": [30]        # "negative": [30]
        },
        {
            "vector_size": [100, 150, 200],
            "epochs": [20, 30, 40],
            "window": [1, 2], 
            "negative": [20]
        },

    ]
   
    for params in ParameterGrid(param_grid):

        model = Doc2Vec(**params, **common_params)

        # Creating a directory for model and log files
        filename = get_model_str_repr(model)
        output_path = Path(f"models/{filename}")
        output_path.mkdir(exist_ok=True, parents=True)

        # Setting up logging to file
        logging_fmt = logging.getLogger().handlers[0].formatter
        file_handler = logging.FileHandler(output_path.joinpath(f"{filename}.log"))
        file_handler.setFormatter(logging_fmt)
        logging.getLogger().addHandler(file_handler)

        # Model set-up and training
        logging.info(f"{str(model)} scanning vocabulary & initializing state...")
        model.build_vocab(documents)
        
        logging.info(f"Training {str(model)}...")
        model.train([documents], total_examples=len(documents), epochs=model.epochs)

        # Save model to file
        logging.info(f"Saving {str(model)} to file {str(output_path)}...")
        model.save(str(output_path.joinpath(filename)))

        # Remove file handler from the list of handlers in the root logger
        logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(f"gensim FAST_VERSION: {FAST_VERSION}")
    logging.info(f'Available cores: {multiprocessing.cpu_count()}')
    main()