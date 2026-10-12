import pdb

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class Attention(dm.WordComparator):
    r"""__init__(heads=1, hidden_size=None, raw_alignment=False, input_dropout=0, alignment_network='decomposable', scale=False, score_dropout=0, value_transform_network=None, input_transform_network=None, value_merge='concat', transform_dropout=0, comparison_merge='concat', comparison_network='2-layer-highway', input_size=None)

    Attention based Word Comparator with `multi-head <https://arxiv.org/abs/1706.03762>`__
    support. This module does the following:

    1. Computes an alignment matrix between the primary input sequence and the context
       input sequence.
    2. For each vector in the primary input sequence, takes a weighted average over all
       vectors in the context input sequence, where weights are given by the alignment
       matrix. Intuitively, for each word / phrase vector in the primary input sequence,
       this represents the aligning word / phrase vector in the context input sequence.
    3. Compares the vectors in the primary input sequence with its aligning vectors.

    Args:
        heads (int):
            Number of attention heads to use. Defaults to 1.
        hidden_size (int):
            The default hidden size of the `alignment_network`, transform networks (if
            applicable), and comparison network.
        raw_alignment (bool):
            If True, uses the contextualized version (transformed by the Word
            Contextualizer module) of the input and context sequences for computing
            alignment in Step 1 described above. If False, uses the raw
            (non-contextualized) word embedding sequences for computing alignment in Step
            1 described above. For step 2, the Word Contextualizer's version of the
            context sequence is used for computing the weighted averages in both cases.
            Raw alignment has been shown to `perform better and speed up convergence
            <http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf>`__,
            especially in cases of limited training data.
        input_dropout (float):
            If non-zero, applies dropout to the input to this module. Dropout probability
            must be between 0 and 1.
        alignment_network (string or :class:`deepmatcher.modules.AlignmentNetwork` or callable):
            The neural network that takes the primary input sequence, aligns the word /
            phrase vectors in this sequence with word / phrase vector in the context
            sequence, and returns the corresponding alignment score matrix. Argument must
            specify a :ref:`align-op` operation.
        scale (bool):
            Whether to scale the alignment scores by the square root of the
            `hidden_size` parameter. Based on `scaled dot-product attention
            <https://arxiv.org/abs/1706.03762>`__
        score_dropout (float):
            If non-zero, applies dropout to the alignment score matrix. Dropout
            probability must be between 0 and 1.
        value_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to transform the context input sequence before taking the
            weighted averages in Step 2 described above. Argument must be None or specify
            a :ref:`transform-op` operation. If the argument is a string, the hidden size
            of the transform operation is computed as :code:`hidden_size // heads`. If
            argument is None, and `heads` is 1, then the values are not transformed. If
            argument is None and `heads` is > 1, then a 1 layer highway network without
            any non-linearity is used. The hidden size for this is computed as mentioned
            above.
        input_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to transform the primary input sequence before it is
            compared with the aligning vectors in the context sequence in Step 3 described
            above. Argument must be None or specify a :ref:`transform-op` operation. If
            None, uses the same neural network as the value transform network (sharing not
            just the structure but also weight parameters).
        value_merge (string or :class:`~deepmatcher.modules.Merge` or callable):
            Specifies how to merge the outputs of all attention heads for each vector
            in the primary input sequence. Concatenates the outputs of all heads by
            default. Argument must specify a :ref:`merge-op` operation.
        transform_dropout (float):
            If non-zero, applies dropout to the outputs of the `value_transform_network`
            and `input_transform_network`, if applicable. Dropout probability must be
            between 0 and 1.
        comparison_merge (string or :class:`~deepmatcher.modules.Merge` or callable):
            For each vector in the primary input sequence, specifies how to merge it with
            its aligning vector in the context input sequence, to obtain a single vector.
            The resulting sequence vectors forms the input to the `comparison_network`.
            Concatenates each primary input vector with its aligning vector by default.
            Argument must specify a :ref:`merge-op` operation.
        comparison_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            The neural network to compare the vectors in the primary input sequence and
            their aligning vectors in the context input. Input to this module is produced
            by the `comparison_merge` operation. Argument must specify a
            :ref:`transform-op` operation.
        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.
    """

    def _init(self,
              heads=1,
              hidden_size=None,
              raw_alignment=False,
              input_dropout=0,
              alignment_network='decomposable',
              scale=False,
              score_dropout=0,
              value_transform_network=None,
              input_transform_network=None,
              value_merge='concat',
              transform_dropout=0,
              comparison_merge='concat',
              comparison_network='2-layer-highway',
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size[0]

        self.alignment_networks = nn.ModuleList()
        for head in range(heads):
            self.alignment_networks.append(
                dm.modules._alignment_module(alignment_network, hidden_size))

        if value_transform_network is None and heads > 1:
            value_transform_network = dm.modules.Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size // heads)
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size // heads)

        if input_transform_network is None:
            self.input_transform_network = self.value_transform_network
        else:
            self.input_transform_network = dm.modules._transform_module(
                input_transform_network, hidden_size // heads)

        self.value_merge = dm.modules._merge_module(value_merge)
        self.comparison_merge = dm.modules._merge_module(comparison_merge)
        self.comparison_network = dm.modules._transform_module(comparison_network,
                                                               hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(score_dropout)
        self.softmax = nn.Softmax(dim=2)

        self.raw_alignment = raw_alignment
        self.heads = heads
        self.scale = scale

    def _forward(self,
                 input_with_meta,
                 context_with_meta,
                 raw_input_with_meta=None,
                 raw_context_with_meta=None):
        input = self.input_dropout(input_with_meta.data)
        context = self.input_dropout(context_with_meta.data)
        raw_input = self.input_dropout(raw_input_with_meta.data)
        raw_context = self.input_dropout(raw_context_with_meta.data)

        queries = input
        keys = context
        values = context
        if self.raw_alignment:
            queries = raw_input
            keys = raw_context

        inputs_transformed = []
        values_aligned = []
        for head in range(self.heads):
            # Dims: batch x len1 x len2
            alignment_scores = self.score_dropout(self.alignment_networks[head](queries,
                                                                                keys))
            if self.scale:
                alignment_scores = alignment_scores / torch.sqrt(hidden_size)

            if context_with_meta.lengths is not None:
                mask = _utils.sequence_mask(context_with_meta.lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                alignment_scores.data.masked_fill_(~mask, -float('inf'))

            # Make values along dim 2 sum to 1.
            normalized_scores = self.softmax(alignment_scores)

            if self.input_transform_network is not None:
                inputs_transformed.append(
                    self.transform_dropout(self.input_transform_network(input)))
            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(values))
            else:
                values_transformed = values

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        inputs_merged = input
        if inputs_transformed:
            inputs_merged = self.value_merge(*inputs_transformed)
        values_merged = self.value_merge(*values_aligned)

        comparison_merged = self.comparison_merge(inputs_merged, values_merged)
        comparison = self.comparison_network(comparison_merged)

        return AttrTensor.from_old_metadata(comparison, input_with_meta)


class MultiHeadCrossAttention(dm.WordComparator):
    """__init__(heads=4, hidden_size=None, raw_alignment=True, dropout=0.1, comparison_network='2-layer-highway', input_size=None)

    Multi-head cross-attention Word Comparator (alias ``'multi-head-attention'``).

    Like :class:`Attention`, each word of the primary sequence is aligned with the
    context sequence and compared with what it aligns to; unlike it, alignment is
    scaled dot-product attention with learned query/key/value projections and several
    heads, so different heads can align on different cues (token identity, character
    shape, position). The comparison network sees ``[word, aligned, word * aligned]``.

    Args:
        heads (int): number of heads; must divide `hidden_size`.
        hidden_size (int): width of the attention space and of the output. Defaults to
            the width of the (contextualized) input.
        raw_alignment (bool): as in :class:`Attention`, compute queries and keys from
            the raw word embeddings instead of the contextualized ones (values always
            come from the contextualized context). Default True, like the Hybrid model.
        dropout (float): dropout on attention weights, training only.
        comparison_network (string or callable): transform applied to the comparison
            vector, as in :class:`Attention`.
        input_size (list): set automatically by :class:`LazyModule`.
    """

    def _init(self, heads=4, hidden_size=None, raw_alignment=True, dropout=0.1,
              comparison_network='2-layer-highway', input_size=None):
        dim = hidden_size if hidden_size is not None else input_size[0]
        if dim % heads != 0:
            raise ValueError('MultiHeadCrossAttention: hidden size {} is not divisible by '
                             '{} heads'.format(dim, heads))
        query_in = input_size[2] if raw_alignment else input_size[0]
        key_in = input_size[3] if raw_alignment else input_size[1]
        self.query = nn.Linear(query_in, dim)
        self.key = nn.Linear(key_in, dim)
        self.value = nn.Linear(input_size[1], dim)
        self.out = nn.Linear(dim, dim)
        self.input_proj = nn.Linear(input_size[0], dim) if dim != input_size[0] else None
        self.comparison_network = dm.modules._transform_module(comparison_network, dim)
        self.heads = heads
        self.dropout = dropout
        self.raw_alignment = raw_alignment
        self.dim = dim

    def _split(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.dim // self.heads).transpose(1, 2)

    def _forward(self,
                 input_with_meta,
                 context_with_meta,
                 raw_input_with_meta=None,
                 raw_context_with_meta=None):
        input, context = input_with_meta.data, context_with_meta.data
        queries, keys = input, context
        if self.raw_alignment:
            queries, keys = raw_input_with_meta.data, raw_context_with_meta.data

        mask = None
        if context_with_meta.lengths is not None:
            from .word_contextualizers import key_padding_mask
            mask = key_padding_mask(context_with_meta.lengths, context.size(1),
                                    context.device)[:, None, None, :]
        aligned = F.scaled_dot_product_attention(
            self._split(self.query(queries)), self._split(self.key(keys)),
            self._split(self.value(context)), attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        batch, _, length, _ = aligned.shape
        aligned = self.out(aligned.transpose(1, 2).reshape(batch, length, self.dim))

        own = self.input_proj(input) if self.input_proj is not None else input
        comparison = self.comparison_network(torch.cat([own, aligned, own * aligned], dim=2))
        return AttrTensor.from_old_metadata(comparison, input_with_meta)
