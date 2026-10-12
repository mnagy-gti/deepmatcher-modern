import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger('deepmatcher.optim')


class SoftNLLLoss(nn.NLLLoss):
    """A soft version of negative log likelihood loss with support for label smoothing.

    Effectively equivalent to PyTorch's :class:`torch.nn.NLLLoss`, if `label_smoothing`
    set to zero. While the numerical loss values will be different compared to
    :class:`torch.nn.NLLLoss`, this loss results in the same gradients. This is because
    the implementation uses :class:`torch.nn.KLDivLoss` to support multi-class label
    smoothing.

    Args:
        label_smoothing (float):
            The smoothing parameter :math:`epsilon` for label smoothing. For details on
            label smoothing refer `this paper <https://arxiv.org/abs/1512.00567v1>`__.
        weight (:class:`torch.Tensor`):
            A 1D tensor of size equal to the number of classes. Specifies the manual
            weight rescaling applied to each class. Useful in cases when there is severe
            class imbalance in the training set.
        num_classes (int):
            The number of classes.
        size_average (bool):
            By default, the losses are averaged for each minibatch over observations **as
            well as** over dimensions. However, if ``False`` the losses are instead
            summed. This is a keyword only parameter.
    """

    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', Variable(weight))

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        # torch will change KLDivLoss(reduction='mean') to mean 'batchmean'; keep the
        # original element-wise mean the models were trained with explicitly.
        reduction = kwargs.pop('reduction', 'mean')
        self._elementwise_mean = reduction == 'mean'
        self.criterion = nn.KLDivLoss(
            reduction='sum' if self._elementwise_mean else reduction, **kwargs)

    def forward(self, input, target, sample_weight=None):
        """Args:
            input: log-probabilities, shape (N, C).
            target: class indices, shape (N,).
            sample_weight: optional (N,) per-example weights. The loss is then the
                weighted mean ``sum(w_i * l_i) / (sum(w_i) * C)``, which equals the
                unweighted loss when every weight is 1 and ignores the batch's total
                weight scale (so a batch of heavy examples does not get a bigger step).
        """
        one_hot = _smoothed_targets(input, target, self.label_smoothing, self.confidence,
                                    self.num_classes, self.weight)

        if sample_weight is None:
            loss = self.criterion(input, one_hot)
            return loss / input.numel() if self._elementwise_mean else loss

        per_example = torch.nn.functional.kl_div(
            input, one_hot, reduction='none').sum(dim=1)
        return _weighted_mean(per_example, sample_weight, self.num_classes)


def _smoothed_targets(input, target, label_smoothing, confidence, num_classes, weight):
    one_hot = torch.zeros_like(input)
    one_hot.fill_(label_smoothing / (num_classes - 1))
    one_hot.scatter_(1, target.unsqueeze(1).long(), confidence)
    if weight is not None:
        one_hot.mul_(weight)
    return one_hot


def _weighted_mean(per_example, sample_weight, num_classes):
    w = sample_weight.to(per_example.dtype)
    return (w * per_example).sum() / (w.sum().clamp_min(1e-12) * num_classes)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) on log-probabilities, with label smoothing.

    ``-sum_c q_c (1 - p_c)^gamma log p_c`` where ``q`` is the (smoothed, optionally
    class-weighted) target distribution built exactly like :class:`SoftNLLLoss`'s.
    ``gamma=0`` reduces to cross entropy with smoothed labels. The reduction uses the
    same ``/ num_classes`` scale as :class:`SoftNLLLoss` so learning rates stay
    comparable.

    Args:
        gamma (float): focusing parameter; larger values down-weight easy examples.
        label_smoothing (float): as in :class:`SoftNLLLoss`.
        weight (:class:`torch.Tensor`): optional per-class weights, shape (C,).
        num_classes (int): number of classes.
    """

    def __init__(self, gamma=2.0, label_smoothing=0, weight=None, num_classes=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0 and 0.0 <= label_smoothing <= 1.0
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.confidence = 1 - label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', weight)

    def forward(self, input, target, sample_weight=None):
        q = _smoothed_targets(input, target, self.label_smoothing, self.confidence,
                              self.num_classes, self.weight)
        modulating = (1 - input.exp()).clamp_min(0) ** self.gamma
        per_example = -(q * modulating * input).sum(dim=1)
        if sample_weight is None:
            return per_example.mean() / self.num_classes
        return _weighted_mean(per_example, sample_weight, self.num_classes)


# This class is based on the Optimizer class in the ONMT-py project.
class Optimizer(object):
    """Controller class for optimization.

    Mostly a thin wrapper for `optim`, but also useful for implementing learning rate
    scheduling beyond what is currently available. Also implements necessary methods for
    training RNNs such as grad manipulations.

    Args:
        method (string):
            One of [sgd, adagrad, adadelta, adam].
        lr (float):
            Learning rate.
        lr_decay (float):
            Learning rate decay multiplier.
        start_decay_at (int):
            Epoch to start learning rate decay. If None, starts decay when the validation
            accuracy stops improving. Defaults to 1.
        beta1, beta2 (float):
            Hyperarameters for adam.
        adagrad_accum (float, optional):
            Initialization hyperparameter for adagrad.
    """

    def __init__(self,
                 method='adam',
                 lr=0.001,
                 max_grad_norm=5,
                 start_decay_at=1,
                 beta1=0.9,
                 beta2=0.999,
                 adagrad_accum=0.0,
                 lr_decay=0.8):
        self.last_acc = None
        self.lr = lr
        self.original_lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.params = None
        logger.info('Initial learning rate: {:0.3e}'.format(self.lr))

    def set_parameters(self, params):
        """Sets the model parameters and initializes the base optimizer.

        Args:
            params:
                Dictionary of named model parameters. Parameters that do not require
                gradients will be filtered out for optimization.
        """
        self.params = []
        for k, p in params:
            if p.requires_grad:
                self.params.append(p)
        if self.method == 'sgd':
            self.base_optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.base_optimizer = optim.Adagrad(self.params, lr=self.lr)
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    self.base_optimizer.state[p]['sum'] = self.base_optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'adadelta':
            self.base_optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.base_optimizer = optim.Adam(
                self.params, lr=self.lr, betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _set_rate(self, lr):
        for param_group in self.base_optimizer.param_groups:
            param_group['lr'] = self.lr

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient clipping.
        """
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.base_optimizer.step()

    def update_learning_rate(self, acc, epoch):
        """Decay learning rate.

        Decays lerning rate if val perf does not improve or we hit the `start_decay_at`
        limit.

        Args:
            acc:
                The accuracy score on the validation set.
            epoch:
                The current epoch number.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_acc is not None and acc < self.last_acc:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            logger.info('Setting learning rate to {:0.3e} for next epoch'.format(self.lr))

        self.last_acc = acc
        self._set_rate(self.lr)
