#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare text datasets and vocabulary for OpenNMT training."""

import argparse
import glob
import sys
import gc
import os
import codecs
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def _build_experiment_suffix(split_type=None, split_seed=None, split_repetition=None, split_tag=None):
    """Build a stable suffix for split/seed/repetition-specific preprocessing artifacts."""
    if split_tag:
        return split_tag
    parts = []
    if split_type:
        parts.append(str(split_type))
    if split_seed is not None:
        parts.append(f"seed{split_seed}")
    if split_repetition is not None:
        parts.append(f"rep{split_repetition}")
    return "_".join(parts)


def _apply_split_path_overrides(opt):
    """Optionally auto-fill train/valid paths and save_data from split-specific settings."""
    if not getattr(opt, "prepared_dir", None):
        return

    suffix = _build_experiment_suffix(
        split_type=getattr(opt, "split_type", None),
        split_seed=getattr(opt, "split_seed", None),
        split_repetition=getattr(opt, "split_repetition", None),
        split_tag=getattr(opt, "split_tag", None),
    )

    prepared_dir = opt.prepared_dir
    if suffix:
        candidate = os.path.join(prepared_dir, suffix)
        if os.path.isdir(candidate):
            prepared_dir = candidate

    opt.train_src = getattr(opt, "train_src", None) or os.path.join(prepared_dir, "train_src.txt")
    opt.train_tgt = getattr(opt, "train_tgt", None) or os.path.join(prepared_dir, "train_tgt.txt")
    opt.valid_src = getattr(opt, "valid_src", None) or os.path.join(prepared_dir, "valid_src.txt")
    opt.valid_tgt = getattr(opt, "valid_tgt", None) or os.path.join(prepared_dir, "valid_tgt.txt")

    if getattr(opt, "save_data_root", None):
        base_prefix = getattr(opt, "save_data_prefix", "data")
        save_stem = f"{base_prefix}_{suffix}" if suffix else base_prefix
        opt.save_data = os.path.join(opt.save_data_root, save_stem)
    elif not getattr(opt, "save_data", None):
        base_prefix = getattr(opt, "save_data_prefix", "data")
        save_stem = f"{base_prefix}_{suffix}" if suffix else base_prefix
        opt.save_data = os.path.join(prepared_dir, "preprocessed", save_stem)

    for path_name in ["train_src", "train_tgt", "valid_src", "valid_tgt"]:
        path_value = getattr(opt, path_name)
        if not path_value or not os.path.exists(path_value):
            raise FileNotFoundError(f"Missing required file for preprocessing: {path_name}={path_value}")

    if getattr(opt, "save_data", None):
        save_data_dir = os.path.dirname(opt.save_data)
        if save_data_dir:
            os.makedirs(save_data_dir, exist_ok=True)


def check_existing_pt_files(opt):
    """Check for existing .pt files before writing preprocessing outputs."""
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    """ Parsing arguments """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--prepared-dir", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--prepared-dir",
        default=None,
        help="Directory containing train_src.txt/train_tgt.txt/valid_src.txt/valid_tgt.txt. If a split suffix subfolder exists, it is used.",
    )
    parser.add_argument(
        "--split-type",
        choices=["group", "random", "extreme_ood"],
        default=None,
        help="Optional split type used to derive prepared subfolder and save_data suffix.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional split seed identifier used in suffix naming.",
    )
    parser.add_argument(
        "--split-repetition",
        type=int,
        default=None,
        help="Optional split repetition identifier used in suffix naming.",
    )
    parser.add_argument(
        "--split-tag",
        default=None,
        help="Optional explicit suffix (overrides split-type/seed/repetition composition).",
    )
    parser.add_argument(
        "--save-data-root",
        default=None,
        help="Optional directory root for save_data. When set, save_data becomes <save-data-root>/<save-data-prefix>_<suffix>.",
    )
    parser.add_argument(
        "--save-data-prefix",
        default="data",
        help="Base filename prefix used with --save-data-root.",
    )

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    if pre_args.prepared_dir:
        # In auto mode, allow these to be inferred from --prepared-dir.
        for action in parser._actions:
            if action.dest in {"train_src", "train_tgt", "valid_src", "valid_tgt", "save_data"}:
                action.required = False

    opt = parser.parse_args()
    _apply_split_path_overrides(opt)
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, fields,
                                           corpus_type, opt):
    """Read source and target text in chunks to limit memory use.

    Save at most opt.shard_size examples in each chunk. The last chunk may
    contain fewer examples.
    """

    with codecs.open(src_corpus, "r", encoding="utf-8") as fsrc:
        with codecs.open(tgt_corpus, "r", encoding="utf-8") as ftgt:
            logger.info("Reading source and target files: %s %s."
                        % (src_corpus, tgt_corpus))
            src_data = fsrc.readlines()
            tgt_data = ftgt.readlines()

            num_shards = int(len(src_data) / opt.shard_size)
            for x in range(num_shards):
                logger.info("Splitting shard %d." % x)
                f = codecs.open(src_corpus + ".{0}.txt".format(x), "w",
                                encoding="utf-8")
                f.writelines(
                        src_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                f.close()
                f = codecs.open(tgt_corpus + ".{0}.txt".format(x), "w",
                                encoding="utf-8")
                f.writelines(
                        tgt_data[x * opt.shard_size: (x + 1) * opt.shard_size])
                f.close()
            num_written = num_shards * opt.shard_size
            if len(src_data) > num_written:
                logger.info("Splitting shard %d." % num_shards)
                f = codecs.open(src_corpus + ".{0}.txt".format(num_shards),
                                'w', encoding="utf-8")
                f.writelines(
                        src_data[num_shards * opt.shard_size:])
                f.close()
                f = codecs.open(tgt_corpus + ".{0}.txt".format(num_shards),
                                'w', encoding="utf-8")
                f.writelines(
                        tgt_data[num_shards * opt.shard_size:])
                f.close()

    src_list = sorted(glob.glob(src_corpus + '.*.txt'))
    tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))

    ret_list = []

    for index, src in enumerate(src_list):
        logger.info("Building shard %d." % index)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src_path=src,
            tgt_path=tgt_list[index],
            src_dir=opt.src_dir,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size
        )

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        logger.info(" * saving %sth %s data shard to %s."
                    % (index, corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)
        os.remove(src)
        os.remove(tgt_list[index])
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    if (opt.shard_size > 0):
        return build_save_in_shards_using_shards_size(src_corpus,
                                                      tgt_corpus,
                                                      fields,
                                                      corpus_type,
                                                      opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src_corpus,
        tgt_path=tgt_corpus,
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window,
        image_channel_size=opt.image_channel_size)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)

    # assert that arrow tokens are in vocab
    assert '>' in fields['tgt'].vocab.stoi, "Arrow token '>' not found in target vocabulary"


def main():
    opt = parse_args()

    if (opt.max_shard_size > 0):
        raise AssertionError("-max_shard_size is deprecated, please use \
                             -shard_size (number of examples) instead.")

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


if __name__ == "__main__":
    main()
