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


class SequencRData(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logging = logger
        self.args_checkup()

        self.score = make_ext_evaluators(self.opt.measure)[0]

        self.src_buggy = [line.strip() for line in open(self.opt.src_buggy, 'r')]
        self.src_fixed = [line.strip() for line in open(self.opt.src_fixed, 'r')]
        self.abstract_buggy = [line.strip() for line in open(self.opt.abstract_buggy, 'r')]
        self.abstract_fixed = [line.strip() for line in open(self.opt.abstract_fixed, 'r')]

        self.nums_buggy = len(self.src_buggy)

    def get_score(self, src, tgt):
        if self.opt.measure == 'bleu':
            return self.score([src], [tgt]) / 100
        else:
            return self.score(src, tgt)

    def args_checkup(self):
        nums_src_buggy = get_total_nums_line(self.opt.src_buggy)
        nums_src_fixed = get_total_nums_line(self.opt.src_fixed)
        nums_abstract_buggy = get_total_nums_line(self.opt.abstract_buggy)
        nums_abstract_fixed = get_total_nums_line(self.opt.abstract_fixed)

        if nums_src_buggy != nums_src_fixed:
            self.logging.error(f"[src] The nums of buggy[{nums_src_buggy}] and fixed[{nums_src_fixed}] does not match")
            exit(-1)

        if nums_abstract_buggy != nums_abstract_fixed:
            self.logging.error(f"[abstract] The nums of buggy[{nums_abstract_buggy}] and fixed[{nums_abstract_fixed}] does not match")
            exit(-1)

        if nums_abstract_buggy != nums_src_buggy:
            self.logging.error(f"[abstract-src] The nums of buggy[{nums_abstract_buggy}] and fixed[{nums_src_buggy}] does not match")
            exit(-1)

    def run(self):
        buggy_abstract_list = []
        fixed_abstract_list = []
        buggy_src_list = []
        fixed_src_list = []
        for i in range(self.nums_buggy):
            buggy_src = self.src_buggy[i]
            fixed_src = self.src_fixed[i]

            buggy_abstract = self.abstract_buggy[i]
            fixed_abstract = self.abstract_fixed[i]

            score_src = self.get_score(buggy_src, fixed_src)
            score_abstract = self.get_score(buggy_abstract, fixed_abstract)


            if score_src < 1.0:
                buggy_src_list.append(buggy_src)
                fixed_src_list.append(fixed_src)
            else:
                logging.debug(f"[src-buggy]-[{i}]-[{score_src}]: {buggy_src}")
                logging.debug(f"[src-fixed]-[{i}]-[{score_src}]: {fixed_src}")
                logging.debug(f"[abstract-buggy]-[{i}]-[{score_abstract}]: {buggy_abstract}")
                logging.debug(f"[abstract-fixed]-[{i}]-[{score_abstract}]: {fixed_abstract}\n\n")

            if score_abstract < 1.0:
                buggy_abstract_list.append(buggy_abstract)
                fixed_abstract_list.append(fixed_abstract)
            else:
                logging.debug(f"[src-buggy]-[{i}]-[{score_src}]: {buggy_src}")
                logging.debug(f"[src-fixed]-[{i}]-[{score_src}]: {fixed_src}")
                logging.debug(f"[abstract-buggy]-[{i}]-[{score_abstract}]: {buggy_abstract}")
                logging.debug(f"[abstract-fixed]-[{i}]-[{score_abstract}]: {fixed_abstract}\n\n")


        if len(buggy_abstract_list) != len(fixed_abstract_list):
            logging.info(f"[Abstract] The buggy{len(buggy_abstract_list)} and fixed {len(fixed_abstract_list)} does not match ...")
        else:
            logging.info(f"[Abstract] The data [{self.nums_buggy}-{len(fixed_abstract_list)}] will be output {self.opt.output}")


        if len(buggy_src_list) != len(fixed_src_list):
            logging.info(f"[src] The buggy{len(buggy_src_list)} and fixed {len(fixed_src_list)} does not match ...")
        else:
            logging.info(f"[src] The data [{self.nums_buggy}-{len(buggy_src_list)}] will be output {self.opt.output}")



        _nums_buggy = len(buggy_abstract_list)
        nums_train = math.ceil(_nums_buggy * 0.8)
        nums_test = round(_nums_buggy * 0.1)

        train_buggy = buggy_abstract_list[:nums_train]
        train_fixed = fixed_abstract_list[:nums_train]

        with open(join(self.opt.output, "train-buggy.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in train_buggy)

        with open(join(self.opt.output, "train-fixed.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in train_fixed)

        logging.info(f"Generate train file: {len(train_buggy)}, {len(train_fixed)}")

        test_buggy = buggy_abstract_list[nums_train:][:nums_test]
        test_fixed = fixed_abstract_list[nums_train:][:nums_test]

        with open(join(self.opt.output, "test-buggy.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in test_buggy)

        with open(join(self.opt.output, "test-fixed.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in test_fixed)

        logging.info(f"Generate test file: {len(test_buggy)}, {len(test_fixed)}")


        valid_buggy = buggy_abstract_list[nums_train:][nums_test:]
        valid_fixed = fixed_abstract_list[nums_train:][nums_test:]


        with open(join(self.opt.output, "valid-buggy.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in valid_buggy)

        with open(join(self.opt.output, "valid-fixed.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in valid_fixed)
        logging.info(f"Generate valid file: {len(valid_buggy)}, {len(valid_fixed)}")


        with open(join(self.opt.output, "buggy-src.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in buggy_src_list)

        with open(join(self.opt.output, "fixed-src.txt"), 'w') as output:
            output.writelines("%s\n" % place for place in fixed_src_list)
        logging.info(f"Generate source file: {len(buggy_src_list)}, {len(fixed_src_list)}")


if __name__ == "__main__":
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-output', '--output', required=True, type=str, default='')
    parser.add_argument('-src_buggy', '--src_buggy', type=str, required=True)
    parser.add_argument('-src_fixed', '--src_fixed', type=str, required=True)
    parser.add_argument('-abstract_buggy', '--abstract_buggy', type=str, required=True)
    parser.add_argument('-abstract_fixed', '--abstract_fixed', type=str, required=True)
    parser.add_argument('-measure', '--measure', type=str, default='bleu', choices=['similarity', 'bleu'])
    parser.add_argument('-project_log', '--project_log', type=str, default='log.txt')

    parser.add_argument('-debug', '--debug', type=bool, default=False)
    args = parser.parse_args()

    logging = get_logger(save_log=args.project_log, isDebug=args.debug)
    sequence = SequencRData(args, logging)
    sequence.run()



    logging.debug(f'Executing Time: {timeit.default_timer() - start}')
