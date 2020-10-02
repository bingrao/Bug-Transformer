import argparse
from os.path import dirname, abspath, join, exists
from operator import itemgetter
import os
import logging
import sys
from onmt.utils.scores import make_ext_evaluators

BASE_DIR = dirname(dirname(abspath(__file__)))


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


class PredictionSplit(object):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logging = logger
        self.args_checkup()

        self.score = make_ext_evaluators(opt.measure)[0]
        logging.info(f"The number of n_best is [{args.n_best}], and the threshold of best ratio is {args.best_ratio}")

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

    def accuarcy(self):
        count_perfect = 0
        count_changed = 0
        count_bad = 0

        src_buggy = [line.strip() for line in open(self.opt.src_buggy, 'r')]
        src_fixed = [line.strip() for line in open(self.opt.src_fixed, 'r')]
        pred_fixed = [line.strip() for line in open(self.opt.pred_fixed, 'r')]
        n_best = self.opt.n_best
        nums_buggy = len(src_buggy)

        with open(self.opt.output, 'w') as output:
            for i in range(nums_buggy):
                buggy = src_buggy[i]
                fixed = src_fixed[i]
                preds = pred_fixed[i * n_best:(i + 1) * n_best]

                # Reture the best score of similarity with a tuple of (index, similarity_score).
                fixed_max_match = max([(index, self.score(fixed, tgt)) for index, tgt in enumerate(preds)],
                                      key=itemgetter(1))

                # Write the best match into the file
                if i != nums_buggy - 1:
                    output.write(preds[fixed_max_match[0]] + "\n")
                else:
                    output.write(preds[fixed_max_match[0]])

                buggy_max_match = max([(index, self.score(buggy, tgt)) for index, tgt in enumerate(preds)],
                                      key=itemgetter(1))

                if fixed_max_match[1] >= self.opt.best_ratio:
                    count_perfect += 1
                elif buggy_max_match[1] >= self.opt.best_ratio:
                    count_bad += 1
                else:
                    count_changed += 1

        return count_perfect, count_changed, count_bad

    def run(self):
        return self.accuarcy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-output', '--output', required=True, type=str, default='')
    parser.add_argument('-src_buggy', '--src_buggy', type=str, required=True)
    parser.add_argument('-src_fixed', '--src_fixed', type=str, required=True)
    parser.add_argument('-pred_fixed', '--pred_fixed', type=str, required=True)
    parser.add_argument('-n_best', '--n_best', type=int, default=1)
    parser.add_argument('-best_ratio', '--best_ratio', type=float, default=1.0)
    parser.add_argument('-measure', '--measure', type=str, default='similarity', choices=['similarity', 'ast', 'bleu'])
    parser.add_argument('-project_log', '--project_log', type=str, default='log.txt')
    parser.add_argument('-debug', '--debug', type=bool, default=False)
    args = parser.parse_args()
    logging = get_logger(save_log=args.project_log, isDebug=args.debug)
    predictor = PredictionSplit(args, logging)
    count_perfect, count_changed, count_bad = predictor.run()

    sys.exit((str(count_perfect) + " " + str(count_changed) + " " + str(count_bad)))
