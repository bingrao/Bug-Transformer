# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase
from onmt.constants import DefaultTokens


def expand_path(position, d_model):
    import numpy as np
    if type(position) is list:
        index = list(filter(lambda x: x < d_model, position[1:]))
        embedding = np.zeros(d_model, dtype=int)
        embedding[index] = 1
        return embedding
    else:
        position


# https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
def postprocessing(data, arg=None):
    d_model = 512
    arr = []
    for obj in data:
        arr.append([expand_path(pos, d_model) for pos in obj[0]])
    return arr


def preprocess(arr):
    """
    Args:
        data is a list of token positions wher a position is a form of
            list [d_model, idx1, idx2, idx3 ,...] in which d_model means the length
            of the position list, "idx1", "idx2", "idx3" are the the indexes that
            correpsonding value is 1.

    Returns: a 2D dimention (seq_len, d_model=215) to represent positions
    """
    results = []
    for token_paths in arr:
        _token_paths = []
        for token_path in token_paths:
            token, path = token_path.split('@')
            path = path.split(u'\u2191')
            _token_paths.append(path)

        results.append(_token_paths)

    return results


class PathDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src_path"`` or ``"tgt_path"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with PathDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def path_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt-pos"):
        return len(ex.src_pos[0]), len(ex.tgt_pos[0])
    return len(ex.src_pos[0])


# mix this with partial
def _feature_tokenize(string, truncate=None, tok_delim=None, path_delim=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens_path = string.replace(" ", "").split(tok_delim)
    # Sample from tokens
    if truncate is not None:
        tokens_path = tokens_path[:truncate]
    if path_delim is not None:
        tokens_path = [t.split(path_delim) for t in tokens_path]
    return tokens_path


class PathMultiField(Field):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields, path_vec_size=256):
        super(PathMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        self.path_vec_size = path_vec_size
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def pad_seq(self, seq, max_length):
        # pad tail of sequence to extend sequence length up to max_length
        pad = self.base_field.vocab.stoi[self.base_field.pad_token]
        res = seq + [pad for i in range(max_length - len(seq))]
        return res

    def pad(self, minibatch, device=None):
        """Pad a batch of examples to the length of the longest example.
        Args:

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape
                ``(batch_size, max_len, n_feats, feat_dim)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        minibatch = list(minibatch)
        lengths_path = []
        # Batch_Size (b) * nums_path_within_seq (k) * nums_node_within_path (l)
        batch_path = []
        max_len = 0
        for seq in minibatch:
            seq = seq[0]
            seq_path = []
            lengths_path.append(len(seq))
            for tok_paths in seq:
                for path in tok_paths:
                    padded_path = ([self.base_field.init_token] if self.base_field.init_token is not None else [])\
                                  + path \
                                  + ([self.base_field.eos_token] if self.base_field.eos_token is not None else [])
                    arr = [self.base_field.vocab.stoi[x] for x in padded_path]
                    max_len = max_len if max_len > len(arr) else len(arr)
                    seq_path.append(arr)
            batch_path.append(seq_path)

        # flattening (b, k, l) to (b * k, l)
        # this is useful to make torch.tensor
        batch_flatten = [symbol for k in batch_path for symbol in k]
        batch_path_lengths = [len(p) for p in batch_flatten]
        padded_batch = [self.pad_seq(s, max_len) for s in batch_flatten]

        # padded = torch.tensor(padded_batch, dtype=torch.long, device=device)\
        #     .reshape(len(minibatch), -1, max_len)

        padded = torch.tensor(padded_batch, dtype=torch.long, device=device)
        """
        Return:
          padded: [b*l, padded_dim], torch.Size([3927, 18]), The nums of padded AST nodes for each path
          lengths_path: [b, l] sum(lengths_path) ==  b*l. The nums of path for each example
          batch_path_lengths: [b*l, l_token] Then nums of AST node (including EOS) in each path
        """
        return padded, lengths_path, batch_path_lengths

    def postprocess(self, x):
        return [f.postprocessing(x) for _, f in self.fields]

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.
        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is True
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            padded, lengths_path, batch_path_lengths = arr
            lengths_path = torch.tensor(lengths_path, dtype=torch.int, device=device)
            batch_path_lengths = torch.tensor(batch_path_lengths, dtype=torch.int, device=device)

        return padded, lengths_path, batch_path_lengths

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[Int]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """
        return [f.preprocess(x) for _, f in self.fields]

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            device:
            batch (list(object)): A list of object from a batch of examples.
            An object is a list of token positions wher a position is a form of
            list [d_model, idx1, idx2, idx3 ,...] in which d_model means the length
            of the position list, "idx1", "idx2", "idx3" are the the indexes that
            correpsonding value is 1.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        padded = self.pad(batch, device=device)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def __getitem__(self, item):
        return self.fields[item]


def path_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        PathMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", DefaultTokens.PAD)
    bos = kwargs.get("bos", DefaultTokens.BOS)
    eos = kwargs.get("eos", DefaultTokens.EOS)
    path_vec_size = kwargs.get("path_vec_size", 256)
    truncate = kwargs.get("truncate", None)
    fields_ = []
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        use_len = i == 0 and include_lengths
        tokenize = partial(
            _feature_tokenize,
            truncate=truncate,
            tok_delim=u'\u21D2',
            path_delim='\u21C8')
        feat = Field(
            init_token=bos, eos_token=eos, pad_token=pad, dtype=torch.int,
            include_lengths=use_len, use_vocab=False, tokenize=tokenize,
            preprocessing=partial(preprocess), postprocessing=partial(postprocessing))
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = PathMultiField(fields_[0][0], fields_[0][1], fields_[1:], path_vec_size)
    return field
