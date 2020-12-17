""" Statistics calculation utility """
from __future__ import division
import time
import sys

from onmt.utils.logging import logger

import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from statistics import mean

def top_most(field, path, nums=20):
    name, base_field = field.fields[0]
    keys, values = zip(*base_field.vocab.freqs.most_common(nums))
    figure(num=None, figsize=(16, 14), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(keys, values, align='center')
    plt.savefig(path)
    plt.close()


def histogram(data, path):
    bins = np.arange(0, 200, 10) # fixed bin size
    figure(num=None, figsize=(16, 14), dpi=80, facecolor='w', edgecolor='k')
    plt.xlim([0, max(data) + 5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('The distribution of total nums of each examples', fontdict = {'fontsize': 26})
    plt.xlabel(f"The length of each input example", fontdict = {'fontsize': 26})
    plt.ylabel('count', fontdict = {'fontsize': 26})
    plt.savefig(path)
    plt.close()


class VocabularyStats:
    def __init__(self, fields):
        self.fields = fields
        self.src_field = self.fields["src"]
        self.tgt_field = self.fields["tgt"]

    def top_src(self, path):
        top_most(field=self.src_field, path=path, nums=30)

    def top_tgt(self, path):
        top_most(field=self.tgt_field, path=path, nums=30)



class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)


class ScoreMetrics(object):
    def __init__(self, ext_scores):
        self.ext_scores = ext_scores if isinstance(ext_scores, list) else [ext_scores]
        self.metrics = self.build_metrics()

    def build_metrics(self):
        metrics = dict()
        for metric in self.ext_scores:
            metrics[metric] = []
        return metrics

    def update(self, results):
        assert isinstance(results, dict)
        for key, value in results.items():
            self.metrics[key] += [value]

    def get_statistics(self):
        reg = dict()
        for key, value in self.metrics.items():
            reg[key] = round(mean(value), 4) if len(value) > 0 else 0.0
        return reg
