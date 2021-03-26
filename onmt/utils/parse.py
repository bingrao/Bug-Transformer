import configargparse as cfargparse
from configargparse import ConfigFileParser, ConfigFileParserException
import torch
import onmt.opts as opts
from onmt.utils.logging import logger
import os
from collections import OrderedDict
from onmt.constants import CorpusName, ModelTask
from onmt.transforms import AVAILABLE_TRANSFORMS


class CustomizedYAMLConfigFileParser(ConfigFileParser):
    """Parses YAML config files. Depends on the PyYAML module.
    https://pypi.python.org/pypi/PyYAML
    """
    def __init__(self):
        self.work_model = "preprocess"

    def get_syntax_description(self):
        msg = ("The config file uses YAML syntax and must represent a YAML "
            "'mapping' (for details, see http://learn.getgrav.org/advanced/yaml).")
        return msg

    def set_work_model(self, value):
        self.work_model = value

    def _load_yaml(self):
        """lazy-import PyYAML so that configargparse doesn't have to dependend
        on it unless this parser is used."""
        try:
            import yaml
        except ImportError:
            raise ConfigFileParserException("Could not import yaml. "
                "It can be installed by running 'pip install PyYAML'")

        return yaml

    def parse(self, stream):
        """Parses the keys and values from a config file."""
        yaml = self._load_yaml()

        logger.info(f"Loading Config File from {stream} ...")

        try:
            parsed_obj = yaml.safe_load(stream)

        except Exception as e:
            raise ConfigFileParserException("Couldn't parse config file: %s" % e)

        if not isinstance(parsed_obj, dict):
            raise ConfigFileParserException("The config file doesn't appear to "
                "contain 'key: value' pairs (aka. a YAML mapping). "
                "yaml.load('%s') returned type '%s' instead of 'dict'." % (
                getattr(stream, 'name', 'stream'),  type(parsed_obj).__name__))

        result = OrderedDict()
        for key, value in parsed_obj[self.work_model].items():
            if isinstance(value, list):
                result[key] = value
            else:
                result[key] = str(value)

        return result


    def serialize(self, items, default_flow_style=False):
        """Does the inverse of config parsing by taking parsed values and
        converting them back to a string representing config file contents.

        Args:
            default_flow_style: defines serialization format (see PyYAML docs)
        """

        # lazy-import so there's no dependency on yaml unless this class is used
        yaml = self._load_yaml()

        # it looks like ordering can't be preserved: http://pyyaml.org/ticket/29
        items = dict(items)
        return yaml.dump(items, default_flow_style=default_flow_style)


class DataOptsCheckerMixin(object):
    """Checker with methods for validate data related options."""

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file!")

    @classmethod
    def _validate_data(cls, opt):
        """Parse corpora specified in data field of YAML file."""
        import yaml
        default_transforms = opt.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml.safe_load(opt.data)

        for cname, corpus in corpora.items():
            # Check Transforms
            _transforms = corpus.get('transforms', None)
            if _transforms is None:
                logger.info(f"Missing transforms field for {cname} data, "
                            f"set to default: {default_transforms}.")
                corpus['transforms'] = default_transforms
            # Check path
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None:
                raise ValueError(f'Corpus {cname} src path is required.'
                                 'tgt path is also required for non language'
                                 ' modeling tasks.')
            else:
                opt.data_task = ModelTask.SEQ2SEQ
                if path_tgt is None:
                    logger.warning(
                        "path_tgt is None, it should be set unless the task"
                        " is language modeling"
                    )
                    opt.data_task = ModelTask.LANGUAGE_MODEL
                    # tgt is src for LM task
                    corpus["path_tgt"] = path_src
                    corpora[cname] = corpus
                    path_tgt = path_src
                cls._validate_file(path_src, info=f'{cname}/path_src')
                cls._validate_file(path_tgt, info=f'{cname}/path_tgt')
            path_align = corpus.get('path_align', None)
            if path_align is None:
                if hasattr(opt, 'lambda_align') and opt.lambda_align > 0.0:
                    raise ValueError(f'Corpus {cname} alignment file path are '
                                     'required when lambda_align > 0.0')
                corpus['path_align'] = None
            else:
                cls._validate_file(path_align, info=f'{cname}/path_align')
            # Check prefix: will be used when use prefix transform
            src_prefix = corpus.get('src_prefix', None)
            tgt_prefix = corpus.get('tgt_prefix', None)
            if src_prefix is None or tgt_prefix is None:
                if 'prefix' in corpus['transforms']:
                    raise ValueError(f'Corpus {cname} prefix are required.')
            # Check weight
            weight = corpus.get('weight', None)
            if weight is None:
                if cname != CorpusName.VALID:
                    logger.warning(f"Corpus {cname}'s weight should be given."
                                   " We default it to 1 for you.")
                corpus['weight'] = 1
        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms_opts(cls, opt):
        """Check options used by transforms."""
        for name, transform_cls in AVAILABLE_TRANSFORMS.items():
            if name in opt._all_transform:
                transform_cls._validate_options(opt)

    @classmethod
    def _get_all_transform(cls, opt):
        """Should only called after `_validate_data`."""
        all_transforms = set(opt.transforms)
        for cname, corpus in opt.data.items():
            _transforms = set(corpus['transforms'])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)
        if hasattr(opt, 'lambda_align') and opt.lambda_align > 0.0:
            if not all_transforms.isdisjoint(
                    {'sentencepiece', 'bpe', 'onmt_tokenize'}):
                raise ValueError('lambda_align is not compatible with'
                                 ' on-the-fly tokenization.')
            if not all_transforms.isdisjoint(
                    {'tokendrop', 'prefix', 'bart'}):
                raise ValueError('lambda_align is not compatible yet with'
                                 ' potentiel token deletion/addition.')
        opt._all_transform = all_transforms

    @classmethod
    def _validate_fields_opts(cls, opt, build_vocab_only=False):
        """Check options relate to vocab and fields."""
        if build_vocab_only:
            if not opt.share_vocab:
                assert opt.tgt_vocab, \
                    "-tgt_vocab is required if not -share_vocab."
            return
        # validation when train:
        cls._validate_file(opt.src_vocab, info='src vocab')
        if not opt.share_vocab:
            cls._validate_file(opt.tgt_vocab, info='tgt vocab')

        if opt.dump_fields or opt.dump_transforms:
            assert opt.save_data, "-save_data should be set if set \
                -dump_fields or -dump_transforms."
        # Check embeddings stuff
        if opt.both_embeddings is not None:
            assert (opt.src_embeddings is None
                    and opt.tgt_embeddings is None), \
                "You don't need -src_embeddings or -tgt_embeddings \
                if -both_embeddings is set."

        if any([opt.both_embeddings is not None,
                opt.src_embeddings is not None,
                opt.tgt_embeddings is not None]):
            assert opt.embeddings_type is not None, \
                "You need to specify an -embedding_type!"
            assert opt.save_data, "-save_data should be set if use \
                pretrained embeddings."

    @classmethod
    def _validate_language_model_compatibilities_opts(cls, opt):
        if opt.model_task != ModelTask.LANGUAGE_MODEL:
            return

        logger.info("encoder is not used for LM task")

        assert opt.share_vocab and (
            opt.tgt_vocab is None
        ), "vocab must be shared for LM task"

        assert (
            opt.decoder_type == "transformer"
        ), "Only transformer decoder is supported for LM task"

    @classmethod
    def validate_prepare_opts(cls, opt, build_vocab_only=False):
        """Validate all options relate to prepare (data/transform/vocab)."""
        if opt.n_sample != 0:
            assert opt.save_data, "-save_data should be set if \
                want save samples."
        cls._validate_data(opt)
        cls._get_all_transform(opt)
        cls._validate_transforms_opts(opt)
        cls._validate_fields_opts(opt, build_vocab_only=build_vocab_only)

    @classmethod
    def validate_model_opts(cls, opt):
        cls._validate_language_model_compatibilities_opts(opt)


class ArgumentParser(cfargparse.ArgumentParser, DataOptsCheckerMixin):
    """OpenNMT option parser powered with option check methods."""

    def __init__(
            self,
            model,
            config_file_parser_class=CustomizedYAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

        assert model in ["abstract", "preprocess", "train", "translate"], "Unsupported model type %s" % model
        self._config_file_parser.set_work_model(model)

    @classmethod
    def defaults(cls, model, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls(model)
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        if model_opt.word_vec_size > 0:
            model_opt.src_word_vec_size = model_opt.word_vec_size
            model_opt.tgt_word_vec_size = model_opt.word_vec_size

        # Backward compatibility with "fix_word_vecs_*" opts
        if hasattr(model_opt, 'fix_word_vecs_enc'):
            model_opt.freeze_word_vecs_enc = model_opt.fix_word_vecs_enc
        if hasattr(model_opt, 'fix_word_vecs_dec'):
            model_opt.freeze_word_vecs_dec = model_opt.fix_word_vecs_dec

        if model_opt.layers > 0:
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.rnn_size > 0:
            model_opt.enc_rnn_size = model_opt.rnn_size
            model_opt.dec_rnn_size = model_opt.rnn_size

        model_opt.brnn = model_opt.encoder_type == "brnn"

        if model_opt.copy_attn_type is None:
            model_opt.copy_attn_type = model_opt.global_attention

        if model_opt.alignment_layer is None:
            model_opt.alignment_layer = -2
            model_opt.lambda_align = 0.0
            model_opt.full_context_alignment = False

    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.model_type in ["text", "img", "audio", "vec"], \
            "Unsupported model type %s" % model_opt.model_type

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = model_opt.enc_rnn_size == model_opt.dec_rnn_size
        assert model_opt.model_type == 'audio' or same_size, \
            "The encoder and decoder rnns must be the same size for now"

        assert model_opt.rnn_type != "SRU" or model_opt.gpu_ranks, \
            "Using SRU requires -gpu_ranks set."
        if model_opt.share_embeddings:
            if model_opt.model_type != "text":
                raise AssertionError(
                    "--share_embeddings requires --model_type text.")
        if model_opt.lambda_align > 0.0:
            assert model_opt.decoder_type == 'transformer', \
                "Only transformer is supported to joint learn alignment."
            assert model_opt.alignment_layer < model_opt.dec_layers and \
                model_opt.alignment_layer >= -model_opt.dec_layers, \
                "N° alignment_layer should be smaller than number of layers."
            logger.info("Joint learn alignment at layer [{}] "
                        "with {} heads in full_context '{}'.".format(
                            model_opt.alignment_layer,
                            model_opt.alignment_heads,
                            model_opt.full_context_alignment))

    @classmethod
    def ckpt_model_opts(cls, model, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(model, opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if opt.epochs:
            raise AssertionError(
                  "-epochs is deprecated please use -train_steps.")
        if opt.truncated_decoder > 0 and max(opt.accum_count) > 1:
            raise AssertionError("BPTT is not compatible with -accum > 1")

        if opt.gpuid:
            raise AssertionError(
                  "gpuid is deprecated see world_size and gpu_ranks")
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if opt.world_size < len(opt.gpu_ranks):
            raise AssertionError(
                  "parameter counts of -gpu_ranks must be less or equal "
                  "than -world_size.")
        if opt.world_size == len(opt.gpu_ranks) and \
                min(opt.gpu_ranks) > 0:
            raise AssertionError(
                  "-gpu_ranks should have master(=0) rank "
                  "unless -world_size is greater than len(gpu_ranks).")
        assert len(opt.data_ids) == len(opt.data_weights), \
            "Please check -data_ids and -data_weights options!"

        assert len(opt.dropout) == len(opt.dropout_steps), \
            "Number of dropout values must match accum_steps values"

        assert len(opt.attention_dropout) == len(opt.dropout_steps), \
            "Number of attention_dropout values must match accum_steps values"

    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

    @classmethod
    def validate_preprocess_args(cls, opt):
        assert opt.max_shard_size == 0, \
            "-max_shard_size is deprecated. Please use \
            -shard_size (number of examples) instead."
        assert opt.shuffle == 0, \
            "-shuffle is not implemented. Please shuffle \
            your data before pre-processing."

        assert len(opt.train_src) == len(opt.train_tgt), \
            "Please provide same number of src and tgt train files!"

        assert len(opt.train_src) == len(opt.train_ids), \
            "Please provide proper -train_ids for your data!"

        for file in opt.train_src + opt.train_tgt:
            assert os.path.isfile(file), "Please check path of %s" % file

        if len(opt.train_align) == 1 and opt.train_align[0] is None:
            opt.train_align = [None] * len(opt.train_src)
        else:
            assert len(opt.train_align) == len(opt.train_src), \
                "Please provide same number of word alignment train \
                files as src/tgt!"
            for file in opt.train_align:
                assert os.path.isfile(file), "Please check path of %s" % file

        assert not opt.valid_align or os.path.isfile(opt.valid_align), \
            "Please check path of your valid alignment file!"

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check path of your valid src file!"
        assert not opt.valid_tgt or os.path.isfile(opt.valid_tgt), \
            "Please check path of your valid tgt file!"

        assert not opt.src_vocab or os.path.isfile(opt.src_vocab), \
            "Please check path of your src vocab!"
        assert not opt.tgt_vocab or os.path.isfile(opt.tgt_vocab), \
            "Please check path of your tgt vocab!"


