import argparse
from os.path import dirname, abspath, join, exists
from operator import itemgetter
import os
import logging
import sys

from onmt.utils.misc import jarWrapper, call_subprocess
from onmt.utils.scores import make_ext_evaluators
import multiprocessing
import math
import concurrent.futures
import timeit
import urllib.request
from multiprocessing.pool import ThreadPool
import numpy as np
from functools import reduce
import collections


import sys
import re
import os


BASE_DIR = dirname(dirname(abspath(__file__)))


class JavaDelimiter:
    @property
    def varargs(self):
        return "..."

    @property
    def rightBrackets(self):
        return "]"

    @property
    def leftBrackets(self):
        return "["

    @property
    def rightCurlyBrackets(self):
        return "}"

    @property
    def leftCurlyBrackets(self):
        return "{"

    @property
    def biggerThan(self):
        return ">"

    @property
    def semicolon(self):
        return ";"

    @property
    def comma(self):
        return ","

    @property
    def dot(self):
        return "."

    @property
    def assign(self):
        return "="

    @property
    def left(self):
        return "."


def get_logger(run_name="logs", save_log=None, isDebug=False):
    log_filename = f'{run_name}.log'
    if save_log is None:
        log_dir = join(BASE_DIR, 'logs')
        if not exists(log_dir):
            os.makedirs(log_dir)
        log_filepath = join(log_dir, log_filename)
    else:
        log_filepath = save_log

    logger = logging.getLogger(run_name)

    debug_level = logging.DEBUG if isDebug else logging.INFO
    logger.setLevel(debug_level)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def get_total_nums_line(path):
    return len(open(path).readlines())


def chunks(lst, chunk_size):
    """Yield n chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


class PredictionSplit(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logging = logger
        self.args_checkup()

        self.score = make_ext_evaluators(self.opt.measure)[0]

        self.src_buggy = [line.strip() for line in open(self.opt.src_buggy, 'r')]
        self.src_fixed = [line.strip() for line in open(self.opt.src_fixed, 'r')]
        self.pred_fixed = [line.strip() for line in open(self.opt.pred_fixed, 'r')]
        self.n_best = self.opt.n_best
        self.nums_buggy = len(self.src_buggy)
        self.nums_thread = self.opt.nums_thread

        self.jar_exe_args = ['scala', f'-Dlog4j.configuration=file://{self.opt.log4j_config}', self.opt.jar, 'astdiff']

        self.delimiter = JavaDelimiter()

        logging.debug(f"The number of n_best is [{args.n_best}], "
                      f"the threshold of best ratio is {args.best_ratio}, "
                      f"The nums thread {self.nums_thread}")

    def get_score(self, src, tgt):
        if self.opt.measure == 'bleu':
            return self.score([src], [tgt]) / 100
        elif self.opt.measure == 'ast':
            args = self.jar_exe_args + [src] + [tgt]
            nums = self.score(args)
            total = nums / ((len(src.split()) + len(src.split())) / 2)
            return 1.0 - total
        else:
            return self.score(src, tgt)

    def args_checkup(self):
        nums_src_buggy = get_total_nums_line(self.opt.src_buggy)
        nums_src_fixed = get_total_nums_line(self.opt.src_fixed)
        nums_pred_fixed = get_total_nums_line(self.opt.pred_fixed)
        nums_best = self.opt.n_best
        if nums_src_buggy != nums_src_fixed:
            self.logging.error(f"The nums of buggy[{nums_src_buggy}] and fixed[{nums_src_fixed}] does not match")
            exit(-1)

        if nums_src_buggy != (nums_pred_fixed / nums_best):
            self.logging.error(f"The nums of buggy[{nums_src_buggy}] does not match predict[{nums_src_fixed}] "
                               f"with nums of best[{nums_best}]")
            exit(-1)

    def debug_accuarcy(self):
        result = self.thread_run(0, self.src_buggy, self.src_fixed, self.pred_fixed)

        count_perfect, count_changed, count_bad = result["statistics"]
        self.logging.info(
            f"Count Perfect[{count_perfect}], changed {count_changed}, "
            f"bad {count_bad}, performance {(count_perfect * 1.0 / self.nums_buggy): 0.4f}")


        pred_best = result["pred_best"]
        with open(self.opt.output, 'w') as output:
            output.writelines("%s\n" % place for place in pred_best)



        return count_perfect, count_changed, count_bad

    def thread_run(self, thread_id, src_buggy, src_fixed, pred_fixed):
        nums_buggy = len(src_buggy)
        count_perfect = 0
        count_changed = 0
        count_bad = 0
        pred_best = []

        for i in range(nums_buggy):
            buggy = src_buggy[i]
            fixed = src_fixed[i]
            preds = pred_fixed[i * self.n_best:(i + 1) * self.n_best]

            # Reture the best score of similarity with a tuple of (index, similarity_score).
            fixed_preds_score = [(index, self.get_score(fixed, tgt)) for index, tgt in enumerate(preds)]
            fixed_max_match = max(fixed_preds_score, key=itemgetter(1))
            pred_best.append(preds[fixed_max_match[0]])

            buggy_preds_scores = [(index, self.get_score(buggy, tgt)) for index, tgt in enumerate(preds)]
            buggy_max_match = max(buggy_preds_scores, key=itemgetter(1))

            if fixed_max_match[1] >= self.opt.best_ratio:
                count_perfect += 1
            elif buggy_max_match[1] >= self.opt.best_ratio:
                count_bad += 1
            else:
                count_changed += 1

        self.logging.debug(
            f"Thread[{thread_id}]: Count Perfect {count_perfect}, changed {count_changed}, "
            f"bad {count_bad}, performance {(count_perfect * 1.0 / nums_buggy): 0.4f}")

        return {thread_id: {"statistics": np.array([count_perfect, count_changed, count_bad]), "pred_best": pred_best}}

    def retrieve_java_code(self, path):
        predictions = open(path, "r").readlines()
        predictions_asCodeLines = []

        for prediction in predictions:
            tmp = self.retrieve_single_java_source(prediction)
            if tmp != "":
                predictions_asCodeLines.append(tmp)

        if len(predictions_asCodeLines) == 0:
            sys.stderr.write("All predictions contains <unk> token")
            sys.exit(1)

        predictions_asCodeLines_file = open(os.path.join(self.opt.output, "predictions_JavaSource.txt"), "w")
        for predictions_asCodeLine in predictions_asCodeLines:
            predictions_asCodeLines_file.write(predictions_asCodeLine + "\n")
        predictions_asCodeLines_file.close()
        sys.exit(0)

    def retrieve_single_java_source(self, prediction):
        tokens = prediction.strip().split(" ")
        codeLine = ""

        for i in range(len(tokens)):
            # if tokens[i] == "<unk>":
            #     return ""
            if i + 1 < len(tokens):
                # DEL = delimiters
                # ... = method_referece
                # STR = token with alphabet in it

                if not self.isDelimiter(tokens[i]):
                    if not self.isDelimiter(tokens[i + 1]):  # STR (i) + STR (i+1)
                        codeLine = codeLine + tokens[i] + " "
                    else:  # STR(i) + DEL(i+1)
                        codeLine = codeLine + tokens[i]
                else:
                    if tokens[i] == self.delimiter.varargs:  # ... (i) + ANY (i+1)
                        codeLine = codeLine + tokens[i] + " "
                    elif tokens[i] == self.delimiter.biggerThan:  # > (i) + ANY(i+1)
                        codeLine = codeLine + tokens[i] + " "
                    elif tokens[i] == self.delimiter.semicolon:
                        codeLine = codeLine + tokens[i] + " "
                    elif tokens[i] == self.delimiter.comma:
                        codeLine = codeLine + tokens[i] + " "
                    elif tokens[i] == self.delimiter.dot:
                        codeLine = codeLine + tokens[i]
                    elif tokens[i] == self.delimiter.assign:
                        codeLine = codeLine + " " + tokens[i] + " "
                    elif tokens[i] == self.delimiter.leftCurlyBrackets:
                        codeLine = codeLine + tokens[i] + " "
                    elif tokens[i] == self.delimiter.rightBrackets and i > 0:
                        if tokens[i - 1] == self.delimiter.leftBrackets:  # [ (i-1) + ] (i)
                            codeLine = codeLine + tokens[i] + " "
                        else:  # DEL not([) (i-1) + ] (i)
                            codeLine = codeLine + tokens[i]
                    else:  # DEL not(... or ]) (i) + ANY
                        codeLine = codeLine + tokens[i]
            else:
                codeLine = codeLine + tokens[i]
        return codeLine

    @staticmethod
    def isDelimiter(token):
        return not token.upper().isupper()

    def run(self):
        return_value = dict()
        statistics = []
        pred_best = []
        batch_size = math.ceil(self.nums_buggy / self.nums_thread)

        buggy_chunks = chunks(self.src_buggy, batch_size)
        fixed_chunks = chunks(self.src_fixed, batch_size)
        pred_chunks = chunks(self.pred_fixed, batch_size * self.n_best)

        # Using ThreadPoolExecutor will lead to a GIL issue and cannot run in parallel
        # https://gist.github.com/mangecoeur/9540178
        # https://stackoverflow.com/questions/61149803/threads-is-not-executing-in-parallel-python-with-threadpoolexecutor

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nums_thread) as executor:
            futures = {executor.submit(self.thread_run,
                                       thread_id,
                                       buggy,
                                       fixed,
                                       pred): thread_id
                       for thread_id, (buggy, fixed, pred) in
                       enumerate(zip(buggy_chunks, fixed_chunks, pred_chunks))}

            for future in concurrent.futures.as_completed(futures, 60):
                thread_id = futures[future]
                try:
                    return_value.update(future.result())
                except Exception as exc:
                    self.logging.error(f'Thread {thread_id} generated an exception: {exc}')
        return_value = collections.OrderedDict(sorted(return_value.items()))

        for key, value in return_value.items():
            statistics.append(value["statistics"])
            pred_best += value["pred_best"]

        with open(self.opt.output, 'w') as output:
            output.writelines("%s\n" % place for place in pred_best)

        return reduce(lambda a, b: a + b, statistics)


if __name__ == "__main__":
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-output', '--output', required=True, type=str, default='')
    parser.add_argument('-src_buggy', '--src_buggy', type=str, required=True)
    parser.add_argument('-src_fixed', '--src_fixed', type=str, required=True)
    parser.add_argument('-pred_fixed', '--pred_fixed', type=str, required=True)
    parser.add_argument('-n_best', '--n_best', type=int, default=1)
    parser.add_argument('-best_ratio', '--best_ratio', type=float, default=1.0)
    parser.add_argument('-measure', '--measure', type=str, default='bleu', choices=['similarity', 'ast', 'bleu'])
    parser.add_argument('-nums_thread', '--nums_thread', type=int, default=16)
    parser.add_argument('-project_log', '--project_log', type=str, default='log.txt')

    parser.add_argument('-log4j_config', '--log4j_config', type=str, required=False,
                        default='/home/bing/project/OpenNMT-py/examples/learning_fix/config/log4j.properties')

    parser.add_argument('-jar', '--jar', type=str, required=False,
                        default='/home/bing/project/OpenNMT-py/examples/learning_fix/bin/java_abstract-1.0-jar-with'
                                '-dependencies.jar')
    parser.add_argument('-debug', '--debug', type=bool, default=False)
    args = parser.parse_args()

    logging = get_logger(save_log=args.project_log, isDebug=args.debug)
    predictor = PredictionSplit(args, logging)

    # Debug code
    # predictor.debug_accuarcy()

    count_perfect, count_changed, count_bad = predictor.run()

    # predictor.retrieve_java_code(args.src_buggy)

    logging.debug(f'Executing Time: {timeit.default_timer() - start}')
    sys.exit((str(count_perfect) + " " + str(count_changed) + " " + str(count_bad)))
