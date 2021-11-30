import os
import json
import logging
from pathlib import Path
from multiprocessing import Lock, cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid


RUN_FILE_LINE_FMT = "{article_id} Q0 {entity} {rank} {score} {run_id}"
PATH_TO_VECTORS = "data/test/trec_eval/vectors"
RUN_FILES_DIR = "run_files"
SIMILARITIES_FILE = "similarities.json"
LOCK = Lock()


def multiprocess_log(status_msg, lock=LOCK):
    with lock:
        logging.info("[PID:{}] {}".format(os.getpid(), status_msg))


def get_similarity_score(similarities_dict, p, strategy):
    key = "_".join(["evs", strategy])
    return (p * similarities_dict[key]) + (1-p) * similarities_dict["evs_child"]


def score_model_performance(model_path):
    model = model_path.name

    # Read in similarities generated using this model
    with open(model_path / SIMILARITIES_FILE, "r") as fp:
        similarities = json.loads(fp.read())
    
    param_grid = {
        "strategy": ["full", "summary"],
        "ratio": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 
                  0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    }

    for run_params in ParameterGrid(param_grid):
        strategy = run_params["strategy"]
        ratio = run_params["ratio"]

        # Create a path to the target file
        run_id = f"run.{model}.r({ratio}).{strategy}"
        run_file = model_path / RUN_FILES_DIR / (run_id + ".txt")

        if not run_file.exists():
            run_file.parent.mkdir(exist_ok=True)
            
            # Lines to be written to the run file
            run_file_lines = []
            for article_id, entities in similarities.items():
                scores = (pd.Series(
                            {entity: get_similarity_score(similarities, ratio, strategy)
                            for entity, similarities in entities.items()}))
                rankings = pd.Series(np.arange(len(scores)), 
                                     index=scores.index[scores.argsort()[::-1]])

                # Generate lines that will be written to the run file regarding
                # the current article
                article_run_file_lines = []
                for entity in entities.keys():
                    line = (RUN_FILE_LINE_FMT
                                .format(article_id=article_id,
                                        entity=entity,
                                        rank=rankings[entity],
                                        score=scores[entity],
                                        run_id=run_id))
                    article_run_file_lines.append(line)
                    multiprocess_log(line)
                
                # Append run file lines to the global list
                run_file_lines += article_run_file_lines 
            
            # Write run file lines to a target run file
            with open(run_file, "w") as fp:
                fp.write("\n".join(run_file_lines))

        else:
            status_msg = f"Run file {run_file} already extists"
            multiprocess_log(status_msg)


def main():
    model_paths = list(Path(PATH_TO_VECTORS).glob("*/"))
    parallelized = Parallel(n_jobs=cpu_count(), backend="multiprocessing", 
                            verbose=40, batch_size=1, max_nbytes=None, 
                            mmap_mode=None)
    parallelized(delayed(score_model_performance)(path) for path in model_paths)
    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    logging.info(f"Available CPU cores: {cpu_count()}")
    main()