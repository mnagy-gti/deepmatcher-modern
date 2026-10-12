r"""Registry of the model and training methods selectable in this package.

User interfaces are expected to render :data:`METHODS` as is, so the texts are written
to be shown to people: what the method is, why it *might* help entity matching on short,
noisy, multilingual company records, where it comes from and what it costs. None of the
"new" entries has been shown to beat the original models on a particular dataset;
treat each as a hypothesis to compare on a validation set, not as an upgrade.

Each :class:`Method` names where the option is used:

==============  ========================================================  ================
kind            selected through                                          example
==============  ========================================================  ================
summarizer      ``MatchingModel(attr_summarizer=...)``                    ``'hybrid'``
contextualizer  ``word_contextualizer=`` of an attribute summarizer       ``'gru'``
comparator      ``word_comparator=`` of an attribute summarizer           ``'multi-head-attention'``
loss            ``TrainConfig(loss=...)``                                 ``'focal'``
schedule        ``TrainConfig(lr_schedule=...)``                          ``'cosine'``
regularizer     ``TrainConfig(token_dropout=..., ema_decay=...)``         ``'ema'``
==============  ========================================================  ================

``hyperparameters`` maps a name to ``{'default', 'range', 'help'}`` where ``range`` is
``[low, high]`` for numbers or the list of allowed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEEPMATCHER = 'Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration", SIGMOD 2018'


@dataclass(frozen=True)
class Method:
    name: str
    kind: str          # summarizer | contextualizer | comparator | loss | schedule | regularizer
    description: str
    rationale: str     # why it may help on short, noisy, multilingual company records
    reference: str
    hyperparameters: dict = field(default_factory=dict)
    cost_note: str = ''
    status: str = 'new'  # legacy | new


def _hp(default, range_, help_):
    return {'default': default, 'range': range_, 'help': help_}


_ALL = [
    # ------------------------------------------------------------ summarizers
    Method(
        'sif', 'summarizer',
        'Each attribute value becomes one vector: an average of its word vectors weighted '
        'by inverse word frequency, with the first principal component of all values '
        'removed. No word order, no cross-record alignment.',
        'The cheapest model and a strong baseline when identity is carried by a few rare '
        'tokens (distinctive company names). It cannot see that two records share a word '
        'in a different position or that a token matches its counterpart, so it tends to '
        'trail the attention models on dirtier data.',
        'Arora et al., "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", ICLR '
        '2017; used in ' + DEEPMATCHER,
        {}, 'Cheapest of the four: no recurrent or attention layers.', 'legacy'),
    Method(
        'rnn', 'summarizer',
        'A bidirectional GRU runs over each attribute value; its last hidden states '
        'summarize the value. Order-aware, no comparison between the two records until '
        'the summaries are merged.',
        'Order matters for names and addresses ("Bank of America" vs "America Bank"), and '
        'an RNN can learn local patterns such as legal-form suffixes. It has no mechanism '
        'to align words across the two records.',
        DEEPMATCHER, {'hidden_size': _hp(300, [16, 1024], 'Width of the recurrent state.')},
        'Sequential over the tokens; slower than SIF, fast on short values.', 'legacy'),
    Method(
        'attention', 'summarizer',
        'Decomposable attention: every word of one record is aligned with the words of '
        'the other record and the pairs are compared, then the comparisons are summed.',
        'Alignment lets the model decide record by record which words correspond, which '
        'helps with reordered tokens, dropped words and abbreviations. It ignores word '
        'order within a value.',
        'Parikh et al., "A Decomposable Attention Model for Natural Language Inference", '
        'EMNLP 2016; used in ' + DEEPMATCHER, {}, 'Quadratic in value length; values here '
        'are short, so this is cheap.', 'legacy'),
    Method(
        'hybrid', 'summarizer',
        'The default model: GRU contextualization, decomposable attention over the raw '
        'word vectors for alignment, then an attention-weighted aggregation that uses the '
        'other record as context.',
        'Combines order awareness with cross-record alignment, among the best performers on '
        'dirty text in the original study, and the model the production checkpoints use.',
        DEEPMATCHER, {'hidden_size': _hp(300, [16, 1024], 'Width of the GRU and of the '
                                                         'comparison network.')},
        'Most expensive of the original four, still small next to the embedding tables.',
        'legacy'),
    Method(
        'hybrid-transformer', 'summarizer',
        'The hybrid model with a transformer encoder (pre-norm self-attention, '
        'feed-forward block, sinusoidal positions) instead of the GRU and multi-head '
        'cross-attention instead of the single-head decomposable attention.',
        'Several heads can align on different cues at once (identity, position, '
        'character shape) and self-attention sees the whole value at once instead of '
        'through a recurrent state. Whether that beats a GRU on values of 3 to 10 tokens '
        'is an empirical question: the sequences are short and the training sets are '
        'small, which favor the smaller model. Compare against `hybrid` on validation '
        'data before adopting it.',
        'Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (architecture); the '
        'combination is specific to this package, not a published model.',
        {'heads': _hp(4, [1, 12], 'Attention heads of both modules; must divide the '
                                  'hidden size (300 by default).'),
         'dropout': _hp(0.1, [0.0, 0.5], 'Dropout inside the attention modules.')},
        'More parameters than the GRU hybrid; no recurrent loop, so it parallelizes '
        'better on GPU/MPS. Measure speed on your hardware before assuming a trade-off.', 'new'),
    # -------------------------------------------------------- contextualizers
    Method(
        'gru', 'contextualizer',
        'Bidirectional gated recurrent unit over the word vectors of one value.',
        'Cheap order-aware context for short sequences; the original default.',
        'Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", EMNLP '
        '2014; used in ' + DEEPMATCHER,
        {'layers': _hp(1, [1, 4], 'Number of stacked layers.')}, 'Sequential.', 'legacy'),
    Method(
        'lstm', 'contextualizer',
        'Bidirectional long short-term memory over the word vectors of one value.',
        'An alternative recurrent unit; differences from the GRU are usually small.',
        'Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation 1997; '
        'used in ' + DEEPMATCHER,
        {'layers': _hp(1, [1, 4], 'Number of stacked layers.')}, 'Four gates instead of '
        'three: about a third more recurrent parameters than the GRU.', 'legacy'),
    Method(
        'self-attention', 'contextualizer',
        'The original DeepMatcher self-attention module (alignment network, optional '
        'multiple heads, highway bypass). Has no normalization, feed-forward block or '
        'position information.',
        'Kept for compatibility. `transformer` is the modern variant.',
        DEEPMATCHER, {'heads': _hp(1, [1, 8], 'Attention heads.')}, 'Quadratic in '
        'value length.', 'legacy'),
    Method(
        'transformer', 'contextualizer',
        'Transformer encoder: input projection and layer norm, sinusoidal positions, N '
        'pre-norm layers of multi-head self-attention plus a feed-forward block, key '
        'padding mask from the value lengths. String aliases `transformer` and '
        '`transformer-encoder`; class '
        '`word_contextualizers.TransformerSelfAttention`.',
        'Lets every token of a value attend to every other one, so a token can be '
        'interpreted in light of the rest ("Holding" after a name vs. alone). Positions '
        'keep order information. Short values give it little to model, and it needs more '
        'data than a GRU to train well, so expect gains, if any, mainly on larger '
        'training sets.',
        'Vaswani et al., NeurIPS 2017; pre-norm placement from Xiong et al., "On Layer '
        'Normalization in the Transformer Architecture", ICML 2020.',
        {'heads': _hp(4, [1, 12], 'Attention heads; must divide the hidden size.'),
         'layers': _hp(1, [1, 6], 'Encoder layers.'),
         'ffn_mult': _hp(2, [1, 8], 'Feed-forward width as a multiple of the hidden size.'),
         'dropout': _hp(0.1, [0.0, 0.5], 'Dropout rate (training only).'),
         'positional': _hp(True, [True, False], 'Add sinusoidal position encodings.')},
        'Parameters grow with layers and ffn_mult; quadratic attention is irrelevant at '
        'these lengths.', 'new'),
    # ------------------------------------------------------------ comparators
    Method(
        'decomposable-attention', 'comparator',
        'Single-head alignment between the words of the two records (a learned '
        'decomposable alignment network), followed by a comparison network over each '
        'word and its aligned counterpart.',
        'The original comparator: robust, few parameters, good for token-level matching '
        'of names and addresses.',
        'Parikh et al., EMNLP 2016; ' + DEEPMATCHER,
        {'raw_alignment': _hp(True, [True, False], 'Align on the raw word vectors '
                                                  '(helps with little data).')},
        'Quadratic in value length.', 'legacy'),
    Method(
        'multi-head-attention', 'comparator',
        'Multi-head scaled dot-product cross-attention with learned query/key/value '
        'projections; the comparison network sees [word, aligned, word * aligned]. '
        'Alias `multi-head-attention`; class `word_comparators.MultiHeadCrossAttention`. '
        '(The original `Attention` comparator also has a `heads` option; this one is a '
        'different, standard formulation.)',
        'Different heads can specialize (exact token overlap, similar spelling, '
        'position), which may separate near-duplicates from true matches. More '
        'parameters to fit than the decomposable comparator, which can overfit small '
        'training sets.',
        'Vaswani et al., NeurIPS 2017.',
        {'heads': _hp(4, [1, 12], 'Attention heads; must divide the hidden size.'),
         'dropout': _hp(0.1, [0.0, 0.5], 'Dropout on attention weights.'),
         'raw_alignment': _hp(True, [True, False], 'Queries and keys from raw word '
                                                  'vectors, as in the Hybrid model.')},
        'Four linear projections per attribute; negligible at these lengths.', 'new'),
    # ----------------------------------------------------------------- losses
    Method(
        'soft_nll', 'loss',
        'Negative log-likelihood with label smoothing and a class weight for the match '
        'class (`pos_neg_ratio`). The original loss.',
        'Label smoothing keeps the network from becoming overconfident on labels that '
        'are themselves noisy (matching labels from heuristics always are); the class '
        'weight compensates for the match/non-match imbalance of candidate sets.',
        'Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016 (label '
        'smoothing); ' + DEEPMATCHER,
        {'label_smoothing': _hp(0.05, [0.0, 0.3], 'Probability mass moved to the wrong '
                                                 'class.'),
         'pos_neg_ratio': _hp(1.0, [0.1, 20.0], 'Weight of matches relative to '
                                               'non-matches.')},
        'None.', 'legacy'),
    Method(
        'focal', 'loss',
        'Focal loss: cross-entropy scaled by (1 - p)^gamma, which shrinks the loss of '
        'examples the model already gets right. Label smoothing and the class weight are '
        'applied as in soft_nll.',
        'Candidate sets are dominated by easy non-matches; focal loss shifts the '
        'gradient to the hard near-duplicates that decide precision. It also makes '
        'probabilities less calibrated, so re-tune the decision threshold (see '
        '`evaluate().threshold_for_precision`).',
        'Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.',
        {'focal_gamma': _hp(2.0, [0.0, 5.0], 'Focusing strength; 0 equals soft_nll.'),
         'label_smoothing': _hp(0.05, [0.0, 0.3], 'As in soft_nll.')},
        'None.', 'new'),
    Method(
        'weighted', 'loss',
        'soft_nll where each training pair contributes in proportion to its `weight` '
        'column (weighted mean per batch).',
        'Lets you encode label confidence (reviewed pairs above heuristic ones), '
        'recency, or cost asymmetries per example. Only as good as the weights supplied.',
        'Standard importance weighting; no single reference.',
        {'sample_weights': _hp(True, [True, False], 'Also usable with the focal loss '
                                                   'via TrainConfig.sample_weights.')},
        'Needs a numeric `weight` column in all CSV splits.', 'new'),
    # --------------------------------------------------------------- schedules
    Method(
        'plateau', 'schedule',
        'Multiply the learning rate by 0.8 after every epoch whose validation score did '
        'not beat the best so far.',
        'Safe default that only reduces the rate when progress stalls. Note the original '
        '`run_train` actually decays after every epoch (set lr_schedule="legacy" to '
        'reproduce that).',
        DEEPMATCHER + ' (decay factor); ReduceLROnPlateau in common frameworks.',
        {'plateau_decay': _hp(0.8, [0.1, 1.0], 'Multiplier on stall.')}, 'None.', 'legacy'),
    Method(
        'cosine', 'schedule',
        'Cosine annealing from the base rate to zero over the whole run, with optional '
        'linear warm-up.',
        'A smooth decay needs no validation signal, which is useful with the tiny '
        'validation sets of low-resource countries. Needs the epoch count up front.',
        'Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", '
        'ICLR 2017 (without restarts).',
        {'warmup_epochs': _hp(0.0, [0.0, 5.0], 'Linear warm-up length in epochs.')},
        'None.', 'new'),
    Method(
        'onecycle', 'schedule',
        'One-cycle policy: rise from lr/25 to lr over the first 30 % of the steps, then '
        'anneal to lr/10000.',
        'Often reaches a good solution in fewer epochs, which matters when training on '
        'a laptop. Less forgiving when lr is mis-set.',
        'Smith and Topin, "Super-Convergence", 2018.', {}, 'None.', 'new'),
    Method(
        'constant', 'schedule',
        'Fixed learning rate (after optional warm-up).',
        'Baseline for comparing schedules and for short fine-tuning runs.',
        'Fixed rate; nothing to cite.', {'warmup_epochs': _hp(0.0, [0.0, 5.0], 'Linear warm-up length.')},
        'None.', 'new'),
    # ----------------------------------------------------------- regularizers
    Method(
        'token-dropout', 'regularizer',
        'During training, each input token is deleted with probability p (the sequence '
        'is compacted, the start/end markers are always kept). Applied independently to '
        'both records.',
        'Company records lose and gain words all the time (legal forms, "the", city '
        'names). Training on randomly shortened values teaches the model not to hinge on '
        'a single token. Subword robustness against misspellings already comes from '
        'fastText vectors, so a character n-gram fallback was deliberately not added.',
        'Related to word dropout in Iyyer et al., "Deep Unordered Composition Rivals '
        'Syntactic Methods for Text Classification", ACL 2015.',
        {'token_dropout': _hp(0.0, [0.0, 0.5], 'Per-token deletion probability; try '
                                              '0.05 to 0.2.')},
        'Slightly noisier epochs; no cost at inference.', 'new'),
    Method(
        'ema', 'regularizer',
        'Keep an exponential moving average of the trainable weights; validate and save '
        '`best.pth` with the averaged weights.',
        'Averaging smooths the last noisy steps, which tends to matter with small '
        'batches and tiny validation sets where the single best epoch is partly luck.',
        'Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging", '
        'SIAM J. Control Optim. 1992.',
        {'ema_decay': _hp(0.99, [0.5, 0.9999], 'Per-step decay; with few steps per '
                                              'epoch use values below 0.99.')},
        'One extra copy of the trainable weights in memory.', 'new'),
]

METHODS = {m.name: m for m in _ALL}

# Considered and deliberately not implemented.
NOT_IMPLEMENTED = {
    'char-ngram-fallback': 'fastText vectors already provide subword information, so a '
                           'separate character n-gram encoder was left out.',
}


def by_kind(kind):
    """Methods of one kind, in registry order."""
    return [m for m in METHODS.values() if m.kind == kind]
