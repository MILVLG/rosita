# coding=utf-8
# Copyright 2019 project LXRT
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required
import logging
logger = logging.getLogger(__name__)

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

SCHEDULES = {
    'warmup_cosine':   warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear':   warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # print(state)
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        warned_for_t_total = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)
                # print(state['step'])
                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # LXRT: grad is clipped outside.
                # Add grad clipping
                # if group['max_grad_norm'] > 0:
                #     clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step']/group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])
                    # warning for exceeding t_total (only active with warmup_linear
                    if group['schedule'] == "warmup_linear" and progress > 1. and not warned_for_t_total:
                        logger.warning(
                            "Training beyond specified 't_total' steps with schedule '{}'. Learning rate set to {}. "
                            "Please set 't_total' of {} correctly.".format(group['schedule'], lr_scheduled, self.__class__.__name__))
                        warned_for_t_total = True
                    # end warning
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

    def set_start_step(self, step):
        pass
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         state = self.state[p]
        #         state['step'] = step

    def decay(self, decay_r):
        for group in self.param_groups:
            group['lr'] *= decay_r


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2).
            Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias
            in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, warmup=-1, t_total=-1, schedule='warmup_linear', betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - "
                             "should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - "
                             "should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - "
                             "should be >= 0.0".format(eps))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        warned_for_t_total = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse '
                        'gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    step_size = group['lr'] * schedule_fct(progress, group['warmup'])
                    # warning for exceeding t_total (only active with warmup_linear
                    if group['schedule'] == "warmup_linear" and progress > 1. and not warned_for_t_total:
                        logger.warning(
                            "Training beyond specified 't_total' steps with schedule '{}'. Learning rate set to {}. "
                            "Please set 't_total' of {} correctly.".format(
                                group['schedule'], step_size, self.__class__.__name__))
                        warned_for_t_total = True
                    # end warning
                else:
                    step_size = group['lr']

                state['step'] += 1

                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = (step_size * math.sqrt(bias_correction2) / bias_correction1)

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is
                # *not* the correct way of using L2 regularization/weight decay
                # with Adam, since that will interact with the m and v
                # parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't
                # interact with the m/v parameters. This is equivalent to
                # adding the square of the weights to the loss with plain
                # (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, epoch_steps, warmup, warmup_epochs):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.epoch_steps = epoch_steps
        self.warmup = warmup
        self.eta_min = 0.
        self.T_max = 20
        self.warmup_epochs = warmup_epochs

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        epoch_ = int(step / self.epoch_steps) - 3

        if self.warmup:
            # if step <= self.epoch_steps * self.warmup_epochs * 1 / 4:
            #     r = self.lr_base * 0.7/5.
            # elif step <= self.epoch_steps * self.warmup_epochs * 2 / 4:
            #     r = self.lr_base * 1.8/5.
            # elif step <= self.epoch_steps * self.warmup_epochs * 3 / 4:
            #     r = self.lr_base * 2.9/5.
            # elif step <= self.epoch_steps * self.warmup_epochs:
            #     r = self.lr_base * 4/5.
            if step <= self.epoch_steps * self.warmup_epochs * 1 / 3:
                r = self.lr_base * 1/4.
            elif step <= self.epoch_steps * self.warmup_epochs * 2 / 3:
                r = self.lr_base * 2/4.
            elif step <= self.epoch_steps * self.warmup_epochs:
                r = self.lr_base * 3/4.
            else:
                r = self.lr_base
                # r = self.eta_min + (self.lr_base - self.eta_min) * (1 + math.cos(math.pi * epoch_ / self.T_max)) / 2
        else:
            r = self.lr_base
            # r = self.eta_min + (self.lr_base - self.eta_min) * (1 + math.cos(math.pi * epoch_ / self.T_max)) / 2

        return r

    def decay(self, decay_r):
        self.lr_base *= decay_r

    def set_start_step(self, step):
        self._step = step

    def get_lr(self):
        return [self._rate]
