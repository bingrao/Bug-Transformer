import torch
import torch.nn as nn
from onmt.decoders.decoder import RNNDecoderBase
from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention
from onmt.utils.rnn_factory import rnn_factory
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.utils.misc import aeq


class PathRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.

    Faster implementation, uses CuDNN for implementation.
    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank=None, memory_lengths=None, **kwargs):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                ``(len, batch, nfeats)``.
            memory_bank (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(src_len, batch, hidden_size)``.
            memory_lengths (LongTensor): the source memory_bank lengths.

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """

        assert self.copy_attn is None  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        attns = {}
        x_path, x_example_len, x_path_len = tgt
        x_path_len, perm_idx = x_path_len.sort(0, descending=True)
        x_path = x_path[perm_idx]
        emb = self.embeddings(x_path.unsqueeze(-1)).transpose(0, 1)

        lengths = x_path_len
        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        # hidden, cell = self.state["hidden"]
        # hidden = self.padded_init_state(emb.size(1), hidden.size(1), hidden)
        # cell = self.padded_init_state(emb.size(1), cell.size(1), cell)

        if isinstance(self.rnn, nn.GRU):
            rnn_output, (final_hidden, final_cell) = self.rnn(packed_emb)
        else:
            rnn_output, (final_hidden, final_cell) = self.rnn(packed_emb)

        if lengths is not None:
            rnn_output, _ = unpack(rnn_output)

        sorted_idx, reversed_perm_idx = perm_idx.sort(0)
        final_hidden = final_hidden[:, reversed_perm_idx, :]
        final_cell = final_cell[:, reversed_perm_idx, :]
        rnn_output = rnn_output[:, reversed_perm_idx, :]
        dec_state = (final_hidden, final_cell)

        # Check
        tgt_batch, tgt_len = x_path.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        output_bag = torch.split(final_hidden.squeeze(0),
                                 x_example_len.cpu().detach().tolist(), dim=0)

        tgt_path_vec = torch.stack([torch.nn.functional.pad(x, pad=[0, 0, 0, kwargs.get("tgt_len", 100) - x.size(0)],
                                                            mode='constant', value=0) for x in output_bag])

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_output
        else:
            dec_outs, p_attn = self.attn(tgt_path_vec, memory_bank,
                                         memory_lengths=memory_lengths)
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(emb.view(-1, emb.size(2)),
                                         rnn_output.view(-1, rnn_output.size(2)),
                                         dec_outs.view(-1, dec_outs.size(2)))
            dec_outs = dec_outs.view(tgt_len, tgt_batch, self.hidden_size)

        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns, rnn_output

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    def padded_init_state(self, len_src, len_init, state):
        if len_src > len_init:
            last = state[:, -1, :].unsqueeze(1)
            padded_last = torch.cat((len_src - len_init) * [last], dim=1)
            state = torch.cat([state, padded_last], dim=1)
        else:
            state = state[:, :len_src, :]
        return state

    def forward(self, tgt, memory_bank=None, memory_lengths=None,
                step=None, position=None, **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
              :param position:
              :param step:
        """

        dec_state, dec_outs, attns, rnn_output = self._run_forward_pass(tgt, memory_bank,
                                                                        memory_lengths=memory_lengths, **kwargs)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns, rnn_output

    def predict(self, tgt, memory_bank=None, memory_lengths=None, **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
              :param position:
              :param step:
        """

        assert self.copy_attn is None  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        attns = {}
        x_path, x_example_len, x_path_len = tgt
        if x_path.dim() == 2:
            emb = self.embeddings(x_path.unsqueeze(-1))
            emb = emb.transpose(0, 1)
            tgt_batch, tgt_len = x_path.size()
        else:
            tgt_len, tgt_batch, _ = x_path.size()
            emb = x_path

        packed_emb = emb

        hidden, cell = self.state["hidden"]
        hidden = self.padded_init_state(emb.size(1), hidden.size(1), hidden)
        cell = self.padded_init_state(emb.size(1), cell.size(1), cell)

        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(packed_emb, hidden)
        else:
            rnn_output, dec_state = self.rnn(packed_emb, (hidden, cell))

        # Check
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        output_bag = torch.split(dec_state[0].squeeze(0),
                                 x_example_len.cpu().detach().tolist(), dim=0)

        tgt_path_vec = torch.stack([torch.nn.functional.pad(x, pad=[0, 0, 0, kwargs.get("tgt_len", 100) - x.size(0)],
                                                            mode='constant', value=0) for x in output_bag])

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_output
        else:
            dec_outs, p_attn = self.attn(tgt_path_vec, memory_bank,
                                         memory_lengths=memory_lengths)
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(emb.view(-1, emb.size(2)),
                                         rnn_output.view(-1, rnn_output.size(2)),
                                         dec_outs.view(-1, dec_outs.size(2)))
            dec_outs = dec_outs.view(tgt_len, tgt_batch, self.hidden_size)

        dec_outs = self.dropout(dec_outs)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns, rnn_output

    @property
    def _input_size(self):
        return self.embeddings.embedding_size

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        # embeddings = nn.Embedding(512, opt.dec_rnn_size)
        # embeddings.embedding_size = opt.dec_rnn_size
        return cls(
            opt.rnn_type,
            opt.brnn,
            1,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim. torch.Size([6, 83, 512])
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        src_path, src_example_len, src_path_len = src
        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)  # torch.Size([1, batch_size, dim])
        self.state["coverage"] = None
