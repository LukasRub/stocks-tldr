from CompactJSONEncoder import CompactJSONEncoder

import os
import json
import logging
import warnings
from datetime import timedelta  
from pathlib import Path
from functools import partial
from multiprocessing import Lock, cpu_count

import wikipedia
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import  thread_map
from tqdm import tqdm  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wikipedia.exceptions import WikipediaException


PATH_TO_STOCKS = "data/processed/stocks/revolut.2021-07-05.complete.stocks.jsonl"
PATH_TO_EXISTING_ENTINTIES = "data/samples/sample_entities/revolut.2021-08-19.top49.entities.jsonl"
PATH_TO_ENTITIES_DIR = "data/processed/entities"
ENTITY_FILENAME_FMT = "{entity}.ce{n_child}.entity.json"
LOCK = Lock()



def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokenized = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    filtered = [token for token in tokenized if not token in stop_words]
    return filtered


def valid_paragraph(paragraph):
    return (len(paragraph) > 0 and not paragraph.isspace() 
                               and not paragraph.startswith('=='))


def get_plaintext(page_content):
    return ' '.join([paragraph for paragraph in page_content.split('\n') 
                     if valid_paragraph(paragraph)])

def extract_and_process_page_content(page, entity="parent"):
    wiki_content = dict()
    if entity == "parent":
        # Extracting wiki page content for the parent entity
        wiki_content["full_text"] = tokenize(get_plaintext(page.content))
        wiki_content["summary"] = tokenize(page.summary)
    elif entity != "parent":
        # Extracting wiki page content for the child entity
        wiki_content["title"] = entity
        wiki_content["summary"] = tokenize(page.summary)
    return wiki_content


def extract_and_save_entity_data(existing_entities, entity):

    # TODO regex matching to see if entity in question already exist
    # in `PATH_TO_ENTITIES_DIR`

    # Make sure to set wikipedia rate limiting to true
    wikipedia.set_rate_limiting(True, min_wait=timedelta(milliseconds=100))
  
    if entity["ticker_symbol"] not in existing_entities:

        # Extracting wiki page's contents for the parent entity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            page = wikipedia.WikipediaPage(pageid=entity["wiki_page_id"])
        entity["wiki_page_content"] = extract_and_process_page_content(page)

        # Extracting wiki pages' contents for the child entities
        entity["child_entities"] = list()
        
       
        for child_entity in page.links:
            try:
                page = wikipedia.WikipediaPage(title=child_entity)
            except wikipedia.DisambiguationError:
                # TODO: logging stderr to file
                pass
            except wikipedia.PageError:
                # TODO: logging stderr to file
                pass
            except Exception:
                # TODO: logging stderr to file
                pass
            else:
                child_entity = extract_and_process_page_content(page, child_entity)
                entity["child_entities"].append(child_entity)
    else:
        # Parent and child entities' wiki page contents already exist in
        # `existing_entities` dict

        # Extracting wiki page's contents from the existing parent entity 
        existing_entity = existing_entities[entity["ticker_symbol"]]
        entity["wiki_page_content"] = dict(full_text=existing_entity["content"],
                                           summary=existing_entity["summary"])
        
        # Extracting wiki page's contents from existing child entities
        entity["child_entities"] = list()
        
        for existing_child_entity in existing_entity["entities"]:                                       
            child_entity = dict(title=existing_child_entity["name"], 
                                summary=existing_child_entity["summary"])
            entity["child_entities"].append(child_entity)

    # Save entity data to it's own destination
    fmt = dict(entity=entity["ticker_symbol"], n_child=len(entity["child_entities"]))
    entity_filename = ENTITY_FILENAME_FMT.format(**fmt)
    path_to_entity_file = Path(PATH_TO_ENTITIES_DIR) / entity_filename
    with open(path_to_entity_file, "w", encoding="utf8") as fp:
        json.dump(entity, fp, ensure_ascii=False, indent="\t")


def read_jsonl_file(filename):
    objects = list()
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            obj = json.loads(line)
            objects.append(obj)
    return objects


def main(stocks_path=str(PATH_TO_STOCKS),
         existing_entities_path=str(PATH_TO_EXISTING_ENTINTIES),
         path_to_entities_dir=str(PATH_TO_ENTITIES_DIR)):

    # Prepare paths
    stocks_path = Path(stocks_path)
    assert stocks_path.exists() and stocks_path.is_file()

    existing_entities_path = Path(existing_entities_path)
    assert existing_entities_path.exists() and existing_entities_path.is_file()

    path_to_entities_dir = Path(path_to_entities_dir) 
    path_to_entities_dir.mkdir(parents=True, exist_ok=True)
    assert path_to_entities_dir.exists() and path_to_entities_dir.is_dir()

    # Read stocks and existing entities JSONL files
    stocks = read_jsonl_file(stocks_path)
    existing_entities = read_jsonl_file(existing_entities_path)

    # Re-organize `existing_entities` list to a dict
    existing_entities = {entity["name"]:entity for entity in existing_entities}

    # Freeze `existing_entities` arg of `extract_and_save_entity_data` func
    extract_and_save_entity_data_ = partial(extract_and_save_entity_data,
                                            existing_entities) 
    
    parallelized = Parallel(n_jobs=cpu_count()*8, backend="threading", verbose=60, 
                            batch_size=1, mmap_mode=None, require=None)

    # with tqdm(total=len(stocks)) as pbar:
    parallelized(delayed(extract_and_save_entity_data_)(stock) for stock in stocks)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"Available CPU cores: {cpu_count()}")
    main()
