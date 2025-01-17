"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from torch.autograd import Variable
from onmt.constants import ModelTask


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
        )
    elif opt.focal_loss > 0 and train:
        criterion = FocalLoss(class_num=len(tgt_field.vocab.itos),
                              ignore_index=padding_idx,
                              alpha=opt.focal_loss,
                              gamma=opt.focal_loss_gamma,
                              isLearnable=opt.focal_loss_learnable,
                              reduction='sum')
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)

    loss_gen = model.generator[0] if use_raw_logits else model.generator

    if opt.copy_attn:
        if opt.model_task == ModelTask.SEQ2SEQ:
            compute = onmt.modules.CopyGeneratorLossCompute(
                criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
                lambda_coverage=opt.lambda_coverage
            )
        elif opt.model_task == ModelTask.LANGUAGE_MODEL:
            compute = onmt.modules.CopyGeneratorLMLossCompute(
                criterion, loss_gen, tgt_field.vocab,
                opt.copy_loss_by_seqlength,
                lambda_coverage=opt.lambda_coverage
            )
        else:
            raise ValueError(
                f"No copy generator loss defined for task {opt.model_task}"
            )
    else:
        if opt.model_task == ModelTask.SEQ2SEQ:
            compute = NMTLossCompute(
                criterion, loss_gen, lambda_coverage=opt.lambda_coverage,
                loss_uncertainty=opt.loss_uncertainty,
                lambda_align=opt.lambda_align)
        elif opt.model_task == ModelTask.LANGUAGE_MODEL:
            assert (
                    opt.lambda_align == 0.0
            ), "lamdba_align not supported in LM loss"
            compute = LMLossCompute(
                criterion,
                loss_gen,
                lambda_coverage=opt.lambda_coverage,
                lambda_align=opt.lambda_align
            )
        else:
            raise ValueError(
                f"No compute loss defined for task {opt.model_task}"
            )
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator, **kwargs):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None, attns_path=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None, **kwargs):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        attns_path = kwargs.get("attns_path", None)
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns, attns_path)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state, **kwargs)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        # shard['output'].shape, torch.Size([2, 61, 512]), shard['target'].shape: torch.Size([2, 61])
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard, **kwargs)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target, align_loss=None, num_align=0):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct,
                                     align_loss=align_loss.item(), num_align=num_align)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class CommonLossCompute(LossComputeBase):
    """
    Loss Computation parent for NMTLossCompute and LMLossCompute

    Implement loss compatible with coverage and alignement shards
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0, tgt_shift_index=1, **kwargs):
        super(CommonLossCompute, self).__init__(criterion, generator, **kwargs)
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align
        self.tgt_shift_index = tgt_shift_index
        self.loss_uncertainty = kwargs.get("loss_uncertainty", False)
        if self.loss_uncertainty:
            # Weight definition for each loss
            self.tgt_log_var = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            self.align_log_var = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def _make_shard_state(self, batch, output, range_, attns=None, attns_path=None):
        """

        :param batch:
        :param output: output.shape: [tgt_len, batch_size, dim] e.g. torch.Size([67, 61, 512])
        :param range_:
        :param attns:
        :return:
        """
        range_start = range_[0] + self.tgt_shift_index
        range_end = range_[1]
        shard_state = {
            # output.shape: [tgt_len, batch_size, dim] e.g.torch.Size([67, 61, 512])
            # target.shape: [tgt_len, batch_size] e.g.torch.Size([67, 61])
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0].contiguous(),
        }

        if self.lambda_coverage != 0.0:
            self._add_coverage_shard_state(shard_state, attns, attns_path)
        if self.lambda_align != 0.0:
            self._add_align_shard_state(
                shard_state, batch, range_start, range_end, attns, attns_path
            )
        return shard_state


    def _add_coverage_shard_state(self, shard_state, attns, attns_path=None):
        coverage = attns.get("coverage", None)
        std = attns.get("std", None)
        assert attns is not None
        assert coverage is not None, (
            "lambda_coverage != 0.0 requires coverage attention"
            " that could not be found in the model."
            " Transformer decoders do not implement coverage"
        )
        assert std is not None, (
            "lambda_coverage != 0.0 requires attention mechanism"
            " that could not be found in the model."
        )
        shard_state.update({"std_attn": attns.get("std"),
                            "coverage_attn": coverage})

    def _compute_loss(self, batch, output, target, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None, **kwargs):
        """

        :param batch:
        :param output: shard['output'].shape [shard_size, bath_size, dim] e.g. torch.Size([2, 61, 512])
        :param target: shard['target'].shape,[shard_size, batch_size] e.g. torch.Size([2, 61])
        :param std_attn:
        :param coverage_attn:
        :param align_head:
        :param ref_align:
        :return:
        """
        num_align = kwargs.get('num_align', 0)

        # bottled_output.shape: [shard_size*batch_size, dim] e.g. torch.Size([122, 512])
        bottled_output = self._bottle(output)

        # score.shape: [shard_size*batch_size, vocab_size] e.g. torch.Size([122, 503])
        scores = self.generator(bottled_output)

        # gtruth.shape: [shard_size*batch_size] e.g. torch.Size([122])
        gtruth = target.view(-1)

        # match_loss = get_max_match_greedy_decode(scores, gtruth).div(float(gtruth.shape[0]))

        # scores [shard_size*batch_size, vocab_size], torch.Size([166, 558]),
        # gtruth: [shard_size*batch_size], torch.Size([166])
        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss

        _align_loss = torch.tensor(0)
        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            _align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)

            if self.loss_uncertainty:
                loss = torch.exp(-self.tgt_log_var) * loss + 0.5 * self.tgt_log_var
                align_loss = 0.5 * torch.exp(-self.align_log_var) * _align_loss + self.align_log_var
            else:
                align_loss = _align_loss * self.lambda_align
            loss += align_loss
        stats = self._stats(loss.clone(), scores, gtruth, align_loss=_align_loss, num_align=num_align)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

    def _add_align_shard_state(self, shard_state, batch, range_start,
                               range_end, attns, attns_path=None):
        # attn_align should be in (batch_size, pad_tgt_size, pad_src_size)
        attn_align = attns.get("align", None)
        if attns_path:
            attn_path_align = attns_path.get("align", None)
            if attn_path_align is not None:
                # TODO use dot product for future
                attn_align = attn_align + attn_path_align

        # align_idx should be a Tensor in size([N, 3]), N is total number
        # of align src-tgt pair in current batch, each as
        # ['sent_N°_in_batch', 'tgt_id+1', 'src_id'] (check AlignField)
        align_idx = batch.align
        assert attns is not None
        assert attn_align is not None, (
            "lambda_align != 0.0 requires " "alignement attention head"
        )
        assert align_idx is not None, (
            "lambda_align != 0.0 requires " "provide guided alignement"
        )
        pad_tgt_size, batch_size, _ = batch.tgt.size()
        pad_src_size = batch.src[0].size(0)
        align_matrix_size = [batch_size, pad_tgt_size, pad_src_size]
        ref_align = onmt.utils.make_batch_align_matrix(
            align_idx, align_matrix_size, normalize=True
        )  # torch.Size([40, 32, 48])
        # NOTE: tgt-src ref alignement that in range_ of shard
        # (coherent with batch.tgt)
        shard_state.update(
            {
                "align_head": attn_align,
                "ref_align": ref_align[:, range_start:range_end, :],
                "num_align": align_idx.size(0),
            }
        )

    def _compute_alignement_loss(self, align_head, ref_align):
        """Compute loss between 2 partial alignment matrix."""
        # align_head contains value in [0, 1) presenting attn prob, torch.Size([2, 31, 48])
        # 0 was resulted by the context attention src_pad_mask, torch.Size([2, 31, 48])
        # So, the correspand position in ref_align should also be 0
        # Therefore, clip align_head to > 1e-18 should be bias free.
        align_loss = -align_head.clamp(min=1e-18).log().mul(ref_align).sum()
        return align_loss


class NMTLossCompute(CommonLossCompute):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0, **kwargs):
        super(NMTLossCompute, self).__init__(criterion, generator,
                                             normalization=normalization,
                                             lambda_coverage=lambda_coverage,
                                             lambda_align=lambda_align,
                                             tgt_shift_index=1, **kwargs)


class LMLossCompute(CommonLossCompute):
    """
    Standard LM Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0):
        super(LMLossCompute, self).__init__(criterion, generator,
                                            normalization=normalization,
                                            lambda_coverage=lambda_coverage,
                                            lambda_align=lambda_align,
                                            tgt_shift_index=0)


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


def get_max_match_greedy_decode(probability, target):
    _, next_word = torch.max(probability, dim=1)
    return torch.sum(torch.eq(next_word, target) == True)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, ignore_index, alpha=None, gamma=2, isLearnable=True, reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        super(FocalLoss, self).__init__()
        if isLearnable:
            self.alpha = Variable(torch.ones(class_num, 1), requires_grad=True)
        else:
            self.alpha = Variable(torch.ones(class_num, 1).fill_(alpha), requires_grad=False)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.class_num = class_num
        self.reduction = reduction

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'sum':
            loss = batch_loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = batch_loss.mean()
        return loss
