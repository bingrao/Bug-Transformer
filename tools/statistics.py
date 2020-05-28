import math
from matplotlib import pyplot as plt
import numpy as np
import random
from matplotlib.pyplot import figure


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

