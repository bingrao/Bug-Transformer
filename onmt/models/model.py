""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False,
                src_pos=None, tgt_pos=None, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
            :param src:
            :param tgt_pos:
            :param src_pos:
        """
        dec_in = tgt[:-1]  # exclude last target from inputs
        tgt_pos = tgt_pos[:-1] if tgt_pos is not None else tgt_pos

        src_path = kwargs.get('src_path', None)
        if src_path is not None and hasattr(self, 'path_encoder'):
            src_path_vec, src_path_state = self.path_encoder(src_path, src_len=src.size(0))
        else:
            src_path_vec = None
            src_path_state = None

        enc_state, memory_bank, lengths = self.encoder(src, lengths, position=src_pos, src_path_vec=src_path_vec)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
            if src_path is not None and hasattr(self, "path_decoder"):
                self.path_decoder.init_state(src_path, src_path_vec, src_path_state)

        tgt_path = kwargs.get('tgt_path', None)
        if src_path is not None and tgt_path is not None and hasattr(self, 'path_decoder'):
            tgt_path_vec, tgt_path_attns, tgt_path_output = self.path_decoder(tgt_path,
                                                                              memory_bank=src_path_vec,
                                                                              memory_lengths=lengths,
                                                                              tgt_len=dec_in.size(0))
        else:
            tgt_path_vec = None
            tgt_path_attns = None
            tgt_path_output = None

        dec_out, attns = self.decoder(dec_in, memory_bank, position=tgt_pos,
                                      memory_lengths=lengths, with_align=with_align,
                                      tgt_path_vec=tgt_path_vec)

        return dec_out, attns, tgt_path_output, tgt_path_attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
