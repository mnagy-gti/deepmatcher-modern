import math

import six

import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import _utils
from ..batch import AttrTensor


class RNN(dm.modules.RNN, dm.WordContextualizer):
    """Multi layered RNN based Word Contextualizer.

    Supports dropout and residual / highway connections. Takes the same parameters as the
    :class:`~deepmatcher.modules.RNN` module.
    """
    pass


# class CNN(dm.WordContextualizer):
#     pass


class SelfAttention(dm.WordContextualizer):
    """__init__(heads=1, hidden_size=None, input_dropout=0, alignment_network='decomposable', scale=False, score_dropout=0, value_transform_network=None, value_merge='concat', transform_dropout=0, output_transform_network=None, output_dropout=0, bypass_network='highway', input_size=None)

    Self Attention based Word Contextualizer.

    Supports `vanilla self attention <https://arxiv.org/abs/1606.01933>`__ and `multi-head
    self attention <https://arxiv.org/abs/1706.03762>`__.

    Args:
        heads (int):
            Number of attention heads to use. Defaults to 1.
        hidden_size (int):
            The default hidden size of the `alignment_network` and transform networks, if
            they are not disabled.
        input_dropout (float):
            If non-zero, applies dropout to the input to this module. Dropout probability
            must be between 0 and 1.
        alignment_network (string or :class:`deepmatcher.modules.AlignmentNetwork` or callable):
            The neural network takes the input sequence, aligns the words in the sequence
            with other words in the sequence, and returns the corresponding alignment
            score matrix. Argument must specify a :ref:`align-op` operation.
        scale (bool):
            Whether to scale the alignment scores by the square root of the
            `hidden_size` parameter. Based on `scaled dot-product attention
            <https://arxiv.org/abs/1706.03762>`__
        score_dropout (float):
            If non-zero, applies dropout to the alignment score matrix. Dropout
            probability must be between 0 and 1.
        value_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            For each word embedding in the input sequence, SelfAttention takes a weighted
            average of the aligning values, i.e., the aligning word embeddings based on
            the alignment scores. This parameter specifies the neural network to transform
            the values (word embeddings) before taking the weighted average. Argument must
            be None or specify a :ref:`transform-op` operation. If the argument is a
            string, the hidden size of the transform operation is computed as
            :code:`hidden_size // heads`. If argument is None, and `heads` is 1, then the
            values are not transformed. If argument is None and `heads` is > 1, then a 1
            layer highway network without any non-linearity is used. The hidden size for
            this is computed as mentioned above.
        value_merge (string or :class:`~deepmatcher.modules.Merge` or callable):
            For each word embedding in the input sequence, each SelfAttention head
            produces one corresponding vector as output. This parameter specifies how
            to merge the outputs of all attention heads for each word embedding.
            Concatenates the outputs of all heads by default. Argument must specify a
            :ref:`merge-op` operation.
        transform_dropout (float):
            If non-zero, applies dropout to the output of the `value_transform_network`,
            if applicable. Dropout probability must be between 0 and 1.
        output_transform_network (string or :class:`~deepmatcher.modules.Transform` or callable):
            For each word embedding in the input sequence, SelfAttention produces one
            corresponding vector as output. This neural network specifies how to transform
            each of these output vectors to obtain a hidden representation of size
            `hidden_size`. Argument must be None or specify a :ref:`transform-op`
            operation. If argument is None, and `heads` is 1, then the output vectors are
            not transformed. If argument is None and `heads` is > 1, then a 1 layer
            highway network without any non-linearity is used.
        output_dropout (float):
            If non-zero, applies dropout to the output of the `output_transform_network`,
            if applicable. Dropout probability must be between 0 and 1.
        bypass_network (string or :class:`Bypass` or callable):
            The bypass network (e.g. residual or highway network) to use. The input word
            embedding sequence to this module is considered as the raw input to the bypass
            network and the final output vector sequence (output of `value_merge` or
            `output_transform_network` if applicable) is considered as the transformed
            input. Argument must specify a :ref:`bypass-op` operation. If None, does not
            use a bypass network.
        input_size (int):
            The number of features in the input to the module. This parameter will be
            automatically specified by :class:`LazyModule`.
    """

    def _init(self,
              heads=1,
              hidden_size=None,
              input_dropout=0,
              alignment_network='decomposable',
              scale=False,
              score_dropout=0,
              value_transform_network=None,
              value_merge='concat',
              transform_dropout=0,
              output_transform_network=None,
              output_dropout=0,
              bypass_network='highway',
              input_size=None):
        hidden_size = hidden_size if hidden_size is not None else input_size

        self.alignment_networks = nn.ModuleList()
        for head in range(heads):
            self.alignment_networks.append(
                dm.modules._alignment_module(alignment_network, hidden_size))

        if value_transform_network is None and heads > 1:
            value_transform_network = dm.modules.Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size // heads)
        self.value_transform_network = dm.modules._transform_module(
            value_transform_network, hidden_size // heads)

        self.value_merge = dm.modules._merge_module(value_merge)

        self.softmax = nn.Softmax(dim=2)

        if output_transform_network is None and heads > 1:
            output_transform_network = dm.modules.Transform(
                '1-layer-highway', non_linearity=None, hidden_size=hidden_size)
        self.output_transform_network = dm.modules._transform_module(
            output_transform_network, hidden_size)

        self.input_dropout = nn.Dropout(input_dropout)
        self.transform_dropout = nn.Dropout(transform_dropout)
        self.score_dropout = nn.Dropout(output_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.bypass_network = dm.modules._bypass_module(bypass_network)

        self.heads = heads
        self.scale = scale
        self.hidden_size = hidden_size

    def _forward(self, input_with_meta):
        input = self.input_dropout(input_with_meta.data)

        values_aligned = []
        for head in range(self.heads):
            # Dims: batch x len1 x len2
            alignment_scores = self.score_dropout(self.alignment_networks[head](input,
                                                                                input))

            if self.scale:
                alignment_scores = alignment_scores / torch.sqrt(self.hidden_size)

            if input_with_meta.lengths is not None:
                mask = _utils.sequence_mask(input_with_meta.lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                alignment_scores.data.masked_fill_(~mask, -float('inf'))

            normalized_scores = self.softmax(alignment_scores)

            if self.value_transform_network is not None:
                values_transformed = self.transform_dropout(
                    self.value_transform_network(input))
            else:
                values_transformed = input

            # Dims: batch x len1 x channels
            values_aligned.append(torch.bmm(normalized_scores, values_transformed))

        values_merged = self.value_merge(*values_aligned)

        output = values_merged
        if self.output_transform_network:
            output = self.output_transform_network(output)
        output = self.output_dropout(output)

        final_output = self.bypass_network(output, input)

        return AttrTensor.from_old_metadata(final_output, input_with_meta)


def sinusoidal_positions(length, dim, device, dtype):
    """Fixed sin/cos position encodings (Vaswani et al., 2017), shape (length, dim).

    Computed on the fly rather than stored as a buffer so checkpoints carry no extra
    state and any sequence length works."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) *
                     (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)[:, :dim // 2]
    return pe.to(dtype)


def key_padding_mask(lengths, length, device):
    """Boolean (batch, length) mask, True where a token is real (not padding).

    A row with no real token (empty value) is unmasked entirely so softmax cannot
    produce NaN; its output is meaningless anyway and is never selected by pooling."""
    if lengths is None:
        return None
    mask = torch.arange(length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask | ~mask.any(dim=1, keepdim=True)


class _PreNormBlock(nn.Module):
    """One pre-norm transformer encoder layer: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))``.

    Hand-rolled on :func:`torch.nn.functional.scaled_dot_product_attention` (explicit
    boolean key mask, no fused nested-tensor fast path) so eager CPU/MPS/CUDA behave
    the same."""

    def __init__(self, dim, heads, ffn_dim, dropout):
        super(_PreNormBlock, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: batch x heads x length x head_dim
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.drop(self.out(attn))
        return x + self.drop(self.ffn(self.norm2(x)))


class TransformerSelfAttention(dm.WordContextualizer):
    """__init__(heads=4, layers=1, hidden_size=None, ffn_mult=2, dropout=0.1, positional=True, input_size=None)

    Transformer-encoder Word Contextualizer: pre-norm multi-head self-attention layers
    with a feed-forward block, sinusoidal position encodings and dropout.

    This is a different module from :class:`SelfAttention` (the original DeepMatcher
    attention, which has no feed-forward block, no normalization and no positions).
    Alias strings: ``'transformer'`` and ``'transformer-encoder'``. (The string
    ``'self-attention'`` keeps meaning the original module so existing code and saved
    model definitions are unchanged.)

    Padding positions are masked out of the keys using the batch's sequence lengths.

    Args:
        heads (int): number of attention heads; must divide `hidden_size`.
        layers (int): number of stacked encoder layers.
        hidden_size (int): model width. If None, the input size (a linear projection is
            added when it differs from the input size).
        ffn_mult (int): feed-forward inner width as a multiple of `hidden_size`.
        dropout (float): dropout on attention weights, sub-layer outputs and the
            feed-forward activations; only active in training.
        positional (bool): add sinusoidal position encodings. Word order matters for
            names ("Bank of America" vs "America Bank") but is noisy for address
            fragments, so it can be switched off.
        input_size (int): set automatically by :class:`LazyModule`.
    """

    def _init(self, heads=4, layers=1, hidden_size=None, ffn_mult=2, dropout=0.1,
              positional=True, input_size=None):
        dim = hidden_size if hidden_size is not None else input_size
        if dim % heads != 0:
            raise ValueError('TransformerSelfAttention: hidden size {} is not divisible by '
                             '{} heads'.format(dim, heads))
        self.in_proj = nn.Linear(input_size, dim) if dim != input_size else None
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [_PreNormBlock(dim, heads, ffn_mult * dim, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.positional = positional
        self.dim = dim

    def _forward(self, input_with_meta):
        x = input_with_meta.data
        if self.in_proj is not None:
            x = self.in_proj(x)
        # Normalize first so token vectors (fastText values are small) and the unit
        # amplitude position encodings are on a comparable scale.
        x = self.input_norm(x)
        if self.positional:
            x = x + sinusoidal_positions(x.size(1), self.dim, x.device, x.dtype)
        x = self.drop(x)
        mask = key_padding_mask(input_with_meta.lengths, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return AttrTensor.from_old_metadata(self.final_norm(x), input_with_meta)
