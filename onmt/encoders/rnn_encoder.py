"""Define RNN-based encoders."""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
import torch

class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, src, lengths=None, **kwargs):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        # [seq_len, batch_size, 1] --> [seq_len, batch_size, dim]
        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        position = kwargs.get("position", None)
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
        # encoder_final (hidden, cell) -> ([nums_layer*directions, batch_size, dim],
        # [nums_layer*directions, batch_size, dim]) torch.Size([6, 83, 512])
        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            # [seq_len, batch_size, dim] torch.Size([47, 83, 512])
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout


class PathRNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0, embeddings=None, use_bridge=False):
        super(PathRNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = rnn_factory(rnn_type,
                                                        input_size=input_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        dropout=dropout,
                                                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            rnn_type='LSTM',
            bidirectional=False,
            num_layers=1,
            input_size=opt.enc_rnn_size,
            hidden_size=opt.enc_rnn_size,
            dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings=embeddings,
            use_bridge=opt.bridge)

    def forward(self, src, lengths=None, **kwargs):
        # x_path [b*l, p], torch.Size([3901, 16]), The padded of each AST path
        # x_example_len [b], torch.Size([83]), The number of paths (l) for each example in a batch
        # x_path_len [b*l], The number of AST nodes (k) for each path

        x_path, x_example_len, x_path_len = src

        x_path_len, perm_idx = x_path_len.sort(0, descending=True)
        x_path = x_path[perm_idx]
        emb = self.embeddings(x_path.unsqueeze(-1)).transpose(0, 1)     # torch.Size([1886, 21]) -> torch.Size([21, 1886, 512])
        if lengths is None:
            lengths = x_path_len

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        # memory_bank, [p, b*l, dim], torch.Size([21, 1886, 512])
        # state -> [hidden, cell], [1, b*l, dim], torch.Size([1, 1886, 512])
        memory_bank, (final_hidden, final_cell) = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank, _ = unpack(memory_bank)

        sorted_idx, reversed_perm_idx = perm_idx.sort(0)
        final_hidden = final_hidden[:, reversed_perm_idx, :]
        final_cell = final_cell[:, reversed_perm_idx, :]
        memory_bank = memory_bank[:, reversed_perm_idx, :]

        if self.use_bridge:
            final_hidden, final_cell = self._bridge((final_hidden, final_cell))

        # [batch_size, (l, dim)]
        output_bag = torch.split(final_hidden.squeeze(0),
                                 x_example_len.cpu().detach().tolist(), dim=0)

        src_len = kwargs.get('src_len', 100)
        # src_path_vec, [b, p_l, dim], torch.Size([41, 46, 512])
        src_path_vec = torch.stack([torch.nn.functional.pad(x, pad=[0, 0, 0, src_len - x.size(0)],
                                                            mode='constant', value=0) for x in output_bag])

        return src_path_vec, (final_hidden, final_cell)

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
