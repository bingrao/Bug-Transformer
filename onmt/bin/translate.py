#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    logger.debug("The Input Parameters:")
    for key, val in vars(opt).items():
        logger.debug(f"[Config]: {key} => {val}")

    translator = build_translator(opt, report_score=True)
    # A list of line in opt.src
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)

    if translator.path_encoding:
        src_path_shards = split_corpus(opt.src_path, opt.shard_size) if opt.src_path is not None else None
        tgt_path_shards = split_corpus(opt.tgt_path, opt.shard_size) if opt.tgt_path is not None else None

        if src_path_shards is None:
            raise ValueError("Not src path is provided")
        elif tgt_path_shards is None:
            shard_pairs = zip(src_shards, tgt_shards, src_path_shards)
        else:
            shard_pairs = zip(src_shards, tgt_shards, src_path_shards, tgt_path_shards)
    else:
        shard_pairs = zip(src_shards, tgt_shards)

    for i, shard_data in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        src_path_shard = None
        tgt_path_shard = None

        if len(shard_data) == 4:
            src_shard, tgt_shard, src_path_shard, tgt_path_shard = shard_data
        elif len(shard_data) == 3:
            src_shard, tgt_shard, src_path_shard = shard_data
        else:
            src_shard, tgt_shard = shard_data

        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug,
            src_path=src_path_shard,
            tgt_path=tgt_path_shard)


def _get_parser():
    parser = ArgumentParser(model="translate", description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
