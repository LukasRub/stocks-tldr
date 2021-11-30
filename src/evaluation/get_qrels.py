import json
import logging
import argparse
import importlib
from pathlib import Path

spec = importlib.util.spec_from_file_location("utils", "src/processing/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

QRELS_LINE_FMT = "{article_id} 0 {ticker} {relevance}"


def write_qrels_to_file(qrels, output_path, filename="qrels.txt"):
    output_file = Path(output_path).joinpath(filename)
    logging.info(f"Writing qrels to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as fp:
        for line in qrels:
            fp.write(line)
            fp.write("\n")


def get_qrels(labels, stocks):
    logging.info("Generating qrels...")
    qrels = list()
    for label in labels:
        for stock in stocks:
            relevance = 0
            if stock["ticker"] in label["ticker_top"]:
                relevance = 2
            elif stock["ticker"] in label["ticker_rel"]:
                relevance = 1
            qrel_line = QRELS_LINE_FMT.format(article_id=label["doc_id"], 
                                              ticker=stock["ticker"], 
                                              relevance=relevance)
            logging.info(qrel_line)
            qrels.append(qrel_line)
    return qrels


def filter_stocks(stocks_list, attribs=["name", "ticker", "sector", "industry", "wiki"]):
    """
    Filters out attributes that are not `name`, `ticker`, `sector`, `industry` 
    or `wiki` and removes stocks whose companies do not have a wiki page (or 
    equivalent)
    """
    filtered_list = list()
    for stock in stocks_list:
        filtered_stock = dict()
        for attr in attribs:
            filtered_stock[attr] = stock[attr]
        filtered_list.append(filtered_stock)
    return filtered_list


def read_jsonl_file(filename, processing_func=None):
    objects = list()
    with open(filename, "r", encoding="utf8") as fp:
        for line in fp:
            obj = json.loads(line)
            objects.append(obj)
    if processing_func:
        objects = processing_func(objects)
    return objects


def main(labels_file, stocks_file, output_path):
    labels = utils.read_jsonl(labels_file)
    stocks = utils.read_jsonl(stocks_file)
    qrels = get_qrels(labels, stocks)
    write_qrels_to_file(qrels, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Script for generating qrels for annotated articles"""
    )
    parser.add_argument("labels_file", help="Article labels (JSON lines file)")
    parser.add_argument("stocks_file", help="Stocks present in the labels (JSON lines file)")
    parser.add_argument("output_path", help="Path where qrels should be saved")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)

    main(args.labels_file, args.stocks_file, args.output_path)