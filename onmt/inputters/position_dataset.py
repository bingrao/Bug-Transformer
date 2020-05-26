# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase


def preprocess(data):
    return list(map(lambda x: list(eval(x)), data.split(" ")))


def expand_pos(position, d_model):
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
        arr.append([expand_pos(pos, d_model) for pos in obj[0]])
    return arr


class PositionDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src_pos"`` or ``"tgt_pos"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with PositionDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def position_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt-pos"):
        return len(ex.src_pos[0]), len(ex.tgt_pos[0])
    return len(ex.src_pos[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
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

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class PositionMultiField(Field):
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

    def __init__(self, base_name, base_field, feats_fields):
        super(PositionMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.
        Args:

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape
                ``(batch_size, max_len, n_feats, feat_dim)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential

        minibatch = list(minibatch)
        lengths = [len(x[0]) for x in minibatch]
        max_len = max(lengths)

        # nfeats = minibatch[0].size(1)
        # feat_dim = minibatch[0].size(2)
        # feats = torch.full((len(minibatch), max_len, nfeats, feat_dim), self.pad_token)
        # for i, (feat, len_) in enumerate(zip(minibatch, lengths)):
        #     feats[i, 0:len_, :, :] = feat

        padded = []
        for x in minibatch:
            x[0].extend([[0, 0] for i in range(max_len - len(x[0]))])
            padded.append(x)
        if self.include_lengths:
            return padded, lengths
        return padded

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
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)
        # arr = postprocessing(arr)

        arr = self.postprocess(arr)

        arr = torch.tensor(arr, dtype=self.dtype, device=device)

        # if self.sequential and not self.batch_first:
        #     arr = arr.permute(1, 0, 2, 3)

        # if self.sequential:
        #     arr = arr.contiguous()

        if self.include_lengths:
            return arr, lengths
        return arr

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
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def __getitem__(self, item):
        return self.fields[item]


def position_fields(**kwargs):
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
        PositionMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", [0])
    fields_ = []
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        use_len = i == 0 and include_lengths
        feat = Field(pad_token=pad, include_lengths=use_len, sequential=False,
                     use_vocab=False, preprocessing=partial(preprocess), postprocessing=partial(postprocessing))
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = PositionMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
