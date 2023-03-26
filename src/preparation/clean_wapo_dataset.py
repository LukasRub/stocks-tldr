import re
import json
import argparse
import logging
from pathlib import Path

import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging stream-like handler to make sure logging works with tqdm
    https://github.com/tqdm/tqdm/issues/193
    """

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)


def batch_size_reached(i, split_files_at):
    return i % split_files_at == 0


def write_processed_data(articles_count, processed_articles, output_path):
    filename = "WaPo_articles_({0}-{1}).cor".format(
        articles_count - len(processed_articles) + 1, articles_count
    )

    output_file = output_path.joinpath(filename)
    logging.info(f"Writing to file: {output_file} ...")

    with output_file.open('w', encoding="utf8") as fp:
        for item in processed_articles:
            json.dump(item, fp, ensure_ascii=False)
            fp.write('\n')


def is_rel_type_subtype(contents_obj):
    relevant_subtypes = ['paragraph', 'correction', 'blockquote', 'p1']
    return (contents_obj is not None and
            contents_obj["type"] == "sanitized_html" and 
            contents_obj["subtype"] in relevant_subtypes)


def extract_plaintext(contents):
    raw_html_pieces = [content_piece["content"] for content_piece in contents 
                       if is_rel_type_subtype(content_piece)]
    document = '\n'.join(raw_html_pieces)
    raw_text = re.sub(r'<.*?>', '', document)
    return raw_text


def process_single_line(json_string):
    article = {}
    try:
        wapo_article_obj = json.loads(json_string)
    except json.JSONDecodeError:
        logging.warning("Invalid JSON string:")
        logging.warning(json_string)
    else:
        article['id'] = wapo_article_obj['id']           
        article['plaintext'] = extract_plaintext(wapo_article_obj['contents'])
    return article


def main(input_file, output_path, batch_size):
    if not Path(output_path).exists():
        Path(output_path).mkdir(exist_ok=True)
    output_path = Path(output_path)
    
    with open(input_file, 'r') as fp:
        linecount = 0
        for line in fp:
            linecount += 1

    processed_lines = []
    with open(input_file, 'r') as fp:
        for i, line in enumerate(tqdm.tqdm(fp, total=linecount), start=1):
            processed_lines.append(process_single_line(line))

            # Write the batch of processed lines to file if the threshold is reached
            if batch_size_reached(i, batch_size):
                logging.info(f'Processed {i} lines ...')
                write_processed_data(i, processed_lines, output_path)
                processed_lines = []

    # Write to file the left-over lines
    if len(processed_lines) > 0:
        logging.info(f'Processed {i} lines ... Finished')
        write_processed_data(i, processed_lines, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Preprocess WaPo articles and convert into a batches of 
                       TaggedLineDocuments for gensim Doc2Vec model training"""
        )
    parser.add_argument("input_file", help="Bundle (JSON lines file)")
    parser.add_argument("output_path", help="Path where processed TaggedLineDocuments should be saved")
    parser.add_argument("batch_size", help="Lines per each outputted TaggedLineDocument file", type=int)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO, handlers=[TqdmLoggingHandler()])

    main(args.input_file, args.output_path, args.batch_size)