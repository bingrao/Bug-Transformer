#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict
from onmt.utils.statistics import histogram

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab, _load_vocab, old_style_vocab, load_old_vocab

from functools import partial
from multiprocessing import Pool
import time
from onmt.utils.statistics import VocabularyStats
from onmt.inputters import PositionDataReader, PathDataReader


def check_existing_pt_files(opt, corpus_type, ids, existing_fields):
    """ Check if there are existing .pt files to avoid overwriting them """
    existing_shards = []
    for maybe_id in ids:
        if maybe_id:
            shard_base = corpus_type + "_" + maybe_id
        else:
            shard_base = corpus_type
        pattern = opt.save_data + '.{}.*.pt'.format(shard_base)
        if glob.glob(pattern):
            if opt.overwrite:
                maybe_overwrite = ("will be overwritten because "
                                   "`-overwrite` option is set.")
            else:
                maybe_overwrite = ("won't be overwritten, pass the "
                                   "`-overwrite` option if you want to.")
            logger.warning("Shards for corpus {} already exist, {}"
                           .format(shard_base, maybe_overwrite))
            existing_shards += [maybe_id]
    return existing_shards


def process_one_shard(corpus_params, params):
    """
    1.  Iteration loading data shards and fill them into corresponding items in the fileds
    2.  During above process, it also generates a src/tgt vocabularies.
    :param corpus_params:
    :param params:
    :return:
    """
    corpus_type, fields, src_reader, tgt_reader, align_reader, opt, \
    existing_fields, src_vocab, tgt_vocab, _src_reader, _tgt_reader = corpus_params

    if _src_reader and _tgt_reader:
        if isinstance(_src_reader, PathDataReader):
            shard_type = "path"
        elif isinstance(_src_reader, PositionDataReader):
            shard_type = "position"
        else:
            shard_type = "code"
    else:
        shard_type = "code"

    i, (src_shard, tgt_shard, align_shard, maybe_id, filter_pred, _src_shard, _tgt_shard) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(tgt_shard)
    logger.info("Building shard %d." % i)

    src_data = {"reader": src_reader, "data": src_shard, "dir": opt.src_dir}
    _src_data = {"reader": _src_reader, "data": _src_shard, "dir": None} \
        if _src_reader is not None and _src_shard is not None else None

    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "dir": None}
    _tgt_data = {"reader": _tgt_reader, "data": _tgt_shard, "dir": None} \
        if _tgt_reader is not None and _tgt_shard is not None else None

    align_data = {"reader": align_reader, "data": align_shard, "dir": None}

    dataset_config = [('src', src_data), ('tgt', tgt_data), ('align', align_data)]

    if _src_data is not None:
        dataset_config.append((f'src_{shard_type}', _src_data))

    if _tgt_data is not None:
        dataset_config.append((f'tgt_{shard_type}', _tgt_data))

    _readers, _data, _dir = inputters.Dataset.config(dataset_config)

    # Reading (_readers) data from (_data) and save them as corresponding (fields)
    dataset = inputters.Dataset(
        fields, readers=_readers, data=_data, dirs=_dir,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred,
        corpus_id=maybe_id
    )

    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            sub_sub_counter['corpus_id'].update(["train" if maybe_id is None else maybe_id])
            for name, field in fields.items():
                if ((opt.data_type == "audio") and (name == "src")) or name == "src_pos" or name == "tgt_pos":
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(f_iter, all_data):
                    has_vocab = (sub_n in ['src', 'src_path'] and src_vocab is not None) or \
                                (sub_n in ['tgt', 'tgt_path'] and tgt_vocab is not None)
                    if hasattr(sub_f, 'sequential') and sub_f.sequential and not has_vocab:
                        if sub_n in ['src_path', 'tgt_path']:
                            import functools
                            import operator
                            val = functools.reduce(operator.iconcat, fd, [])
                            val = functools.reduce(operator.iconcat, val, [])
                        else:
                            val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, shard_base, i)

    if False:
        src_data_path = "{:s}.{:s}.{:s}_{:d}.png".format(opt.save_data, shard_base, "src_hist", i)
        tgt_data_path = "{:s}.{:s}.{:s}_{:d}.png".format(opt.save_data, shard_base, "tgt_hist", i)
        src_lens = list(map(lambda x: len(x.src[0]), dataset.examples))
        tgt_lens = list(map(lambda x: len(x.tgt[0]), dataset.examples))
        histogram(data=src_lens, path=src_data_path)
        histogram(data=tgt_lens, path=tgt_data_path)
        logger.info(
            f"Phase [{shard_base}] average src token {sum(src_lens) / len(src_lens)}, tgt token {sum(tgt_lens) / len(tgt_lens)}")
    logger.info(" * saving %sth %s data shard to %s." % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def maybe_load_vocab(corpus_type, counters, opt):
    src_vocab = None
    tgt_vocab = None
    existing_fields = None
    if corpus_type == "train":
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = _load_vocab(
                opt.tgt_vocab, "tgt", counters,
                opt.tgt_words_min_frequency)
    return src_vocab, tgt_vocab, existing_fields


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, align_reader, opt,
                       src_pos_reader=None, tgt_pos_reader=None,
                       src_path_reader=None, tgt_path_reader=None):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        srcs_pos = opt.train_src_pos
        srcs_path = opt.train_src_path

        tgts = opt.train_tgt
        tgts_pos = opt.train_tgt_pos
        tgts_path = opt.train_tgt_path
        ids = opt.train_ids
        aligns = opt.train_align
    else:  # valid
        counters = None
        srcs = [opt.valid_src]
        srcs_pos = opt.valid_src_pos
        srcs_path = [opt.valid_src_path]

        tgts = [opt.valid_tgt]
        tgts_pos = opt.valid_tgt_pos
        tgts_path = [opt.valid_tgt_path]
        ids = [None]
        aligns = [opt.valid_align]

    # Check if exist src_vocab, tgt_vocab and fields provided by users
    src_vocab, tgt_vocab, existing_fields = maybe_load_vocab(
        corpus_type, counters, opt)

    existing_shards = check_existing_pt_files(
        opt, corpus_type, ids, existing_fields)

    # every corpus has shards, no new one
    if existing_shards == ids and not opt.overwrite:
        return

    def shard_iterator(srcs, tgts, ids, aligns, existing_shards,
                       existing_fields, corpus_type, opt):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, tgt, maybe_id, maybe_align in zip(srcs, tgts, ids, aligns):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}".format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None, \
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "text" or opt.data_type == "code",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)

            for i, (ss, ts, a_s) in enumerate(zip(src_shards, tgt_shards, align_shards)):
                yield i, (ss, ts, a_s, maybe_id, filter_pred, None, None)

    def shard_iterator_with_position(srcs, tgts, ids, aligns, existing_shards,
                                     existing_fields, corpus_type, opt, srcs_pos, tgts_pos):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, tgt, maybe_id, maybe_align, src_pos, tgt_pos in zip(srcs, tgts, ids, aligns, srcs_pos, tgts_pos):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}".format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None, \
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            src_pos_shards = split_corpus(src_pos, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)

            tgt_pos_shards = split_corpus(tgt_pos, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)

            for i, (ss, ts, a_s, sp, tp) in enumerate(
                    zip(src_shards, tgt_shards, align_shards, src_pos_shards, tgt_pos_shards)):
                yield i, (ss, ts, a_s, maybe_id, filter_pred, sp, tp)

    def shard_iterator_with_path(srcs, tgts, ids, aligns, existing_shards,
                                 existing_fields, corpus_type, opt, srcs_path, tgts_path):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, tgt, maybe_id, maybe_align, src_path, tgt_path in zip(srcs, tgts, ids, aligns, srcs_path, tgts_path):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}".format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None, \
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            src_path_shards = split_corpus(src_path, opt.shard_size)

            tgt_shards = split_corpus(tgt, opt.shard_size)
            tgt_path_shards = split_corpus(tgt_path, opt.shard_size)

            align_shards = split_corpus(maybe_align, opt.shard_size)

            for i, (ss, ts, a_s, sp, tp) in enumerate(
                    zip(src_shards, tgt_shards, align_shards, src_path_shards, tgt_path_shards)):
                yield i, (ss, ts, a_s, maybe_id, filter_pred, sp, tp)

    # Loading data from disk and save them as a shard of lines
    if srcs_pos is not None and tgts_pos is not None:
        shard_iter = shard_iterator_with_position(srcs, tgts, ids, aligns, existing_shards,
                                                  existing_fields, corpus_type, opt, srcs_pos, tgts_pos)
    elif srcs_path is not None and tgts_path is not None:
        shard_iter = shard_iterator_with_path(srcs, tgts, ids, aligns, existing_shards,
                                              existing_fields, corpus_type, opt, srcs_path, tgts_path)
    else:
        shard_iter = shard_iterator(srcs, tgts, ids, aligns, existing_shards, existing_fields, corpus_type, opt)

    with Pool(opt.num_threads) as p:
        if srcs_pos is not None and tgts_pos is not None:
            dataset_params = (corpus_type, fields, src_reader, tgt_reader,
                              align_reader, opt, existing_fields,
                              src_vocab, tgt_vocab, src_pos_reader, tgt_pos_reader)
        elif srcs_path is not None and tgts_path is not None:
            dataset_params = (corpus_type, fields, src_reader, tgt_reader,
                              align_reader, opt, existing_fields,
                              src_vocab, tgt_vocab, src_path_reader, tgt_path_reader)
        else:
            dataset_params = (corpus_type, fields, src_reader, tgt_reader,
                              align_reader, opt, existing_fields,
                              src_vocab, tgt_vocab, None, None)

        func = partial(process_one_shard, dataset_params)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        new_fields = _build_fields_vocab(
            fields, counters, opt.data_type,
            opt.share_vocab, opt.vocab_size_multiple,
            opt.src_vocab_size, opt.src_words_min_frequency,
            opt.tgt_vocab_size, opt.tgt_words_min_frequency,
            subword_prefix=opt.subword_prefix,
            subword_prefix_is_joiner=opt.subword_prefix_is_joiner)
        if existing_fields is None:
            fields = new_fields
        else:
            fields = existing_fields

        if old_style_vocab(fields):
            fields = load_old_vocab(
                fields, opt.data_type, dynamic_dict=opt.dynamic_dict)

        # patch corpus_id
        if fields.get("corpus_id", False):
            fields["corpus_id"].vocab = new_fields["corpus_id"].vocab_cls(
                counters["corpus_id"])
        if True:
            _vocab = VocabularyStats(fields)
            _vocab.top_src(path=opt.save_data + '.vocab.src.png')
            _vocab.top_tgt(path=opt.save_data + '.vocab.tgt.png')

        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def preprocess(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)

    logger = init_logger(opt.log_file)

    logger.info("The Input Parameters:")
    for key, val in vars(opt).items():
        logger.info(f"[Config]: {key} => {val}")

    start_time = time.time()
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src[0]) if opt.data_type == 'text' or opt.data_type == 'code' else 0
    tgt_nfeats = count_features(opt.train_tgt[0])  # tgt always text so far
    if len(opt.train_src) > 1 and (opt.data_type == 'text' or opt.data_type == 'code'):
        for src, tgt in zip(opt.train_src[1:], opt.train_tgt[1:]):
            assert src_nfeats == count_features(src), \
                "%s seems to mismatch features of " \
                "the other source datasets" % src
            assert tgt_nfeats == count_features(tgt), \
                "%s seems to mismatch features of " \
                "the other target datasets" % tgt
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        with_align=opt.train_align[0] is not None,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc,
        opt=opt)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    src_pos_reader = inputters.str2reader['position'].from_opt(opt) \
        if opt.data_type == "code" and (opt.train_src_pos is not None or opt.valid_src_pos is not None) else None
    src_path_reader = inputters.str2reader['path'].from_opt(opt) \
        if opt.data_type == "code" and (opt.train_src_path is not None or opt.valid_src_path is not None) else None

    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    tgt_pos_reader = inputters.str2reader['position'].from_opt(opt) \
        if opt.data_type == "code" and (opt.train_tgt_pos is not None or opt.valid_tgt_pos is not None) else None
    tgt_path_reader = inputters.str2reader['path'].from_opt(opt) \
        if opt.data_type == "code" and (opt.train_tgt_path is not None or opt.valid_tgt_path is not None) else None

    align_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset('train', fields, src_reader, tgt_reader, align_reader, opt,
                       src_pos_reader=src_pos_reader, tgt_pos_reader=tgt_pos_reader,
                       src_path_reader=src_path_reader, tgt_path_reader=tgt_path_reader)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, align_reader, opt,
                           src_pos_reader=src_pos_reader, tgt_pos_reader=tgt_pos_reader,
                           src_path_reader=src_path_reader, tgt_path_reader=tgt_path_reader)

    logger.info("--- %s seconds ---" % (time.time() - start_time))


# current_target = "None"


def _get_parser():
    parser = ArgumentParser(model="preprocess", description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    preprocess(opt)


if __name__ == "__main__":
    main()
