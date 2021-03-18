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

    src_path_shards = split_corpus(opt.src_path, opt.shard_size) if opt.src_path is not None else None
    tgt_path_shards = split_corpus(opt.tgt_path, opt.shard_size) if opt.tgt_path is not None else None

    if src_path_shards is not None and tgt_path_shards is not None:
        shard_pairs = zip(src_shards, src_path_shards, tgt_shards, tgt_path_shards)

        for i, (src_shard, src_path_shard, tgt_shard, tgt_path_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
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
    else:
        shard_pairs = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug)


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
