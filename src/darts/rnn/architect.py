import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import math
import torch
from torch.optim import Optimizer


class AAdam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, t0=100):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, t0=t0)
        self.t0 = t0
        self.mu_offset = 0
        super(AAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                #if state['step'] % self.t0 == 0:
                #    p.data.copy_(state['ax'])
                #    self.mu_offset = state['step']
                #    # TODO: store an offset for mu, or keep getting harder?

                # update mu
                state['mu'] = 1 / max(1, state['step'] - self.mu_offset)

        return loss


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef


class Architect(object):

  def __init__(self, model, args):
    self.diff_through_unrolled = args.diff_unrolled # TODO: I added this line
    self.extrapolate_past = args.extrapolate_past
    #TODO: Add averaging the iterates
    self.arch_step = 0
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)
    # TODO: I changed this from Adam.
    #self.optimizer = torch.optim.ASGD(self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay, t0=200)
    # t0 = too is about 2 epochs

  def _compute_unrolled_model(self, hidden, input, target, eta, network_optimizer):
    theta = _concat(self.model.parameters()).data

    # This is the original unrolling for hessian approx
    if self.diff_through_unrolled:
        # Extrapolation
        loss, hidden_next = self.model._loss(hidden, input, target)
        grads = torch.autograd.grad(loss, self.model.parameters())
        clip_coef = _clip(grads, self.network_clip)
        grads = _concat(grads).data
        dtheta = grads + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
        return unrolled_model, clip_coef

    # This breaks early incase we have no prior grad
    clip_coef = None
    for p in network_optimizer.param_groups[0]['params']:
        if p.grad is None:
            unrolled_model = self._construct_model_from_theta(theta)
            return unrolled_model, clip_coef

    if self.extrapolate_past:
        # Extrapolation through the past
        grads = [p.grad for p in self.model.parameters()]
        grads = _concat(grads).data
    else:
        # Extrapolation
        loss, hidden_next = self.model._loss(hidden, input, target)
        grads = torch.autograd.grad(loss, self.model.parameters())
        clip_coef = _clip(grads, self.network_clip)
        grads = _concat(grads).data


    '''
    # The adam formulations
    exp_avg = _concat([network_optimizer.state[p]['exp_avg'] if p.grad is not None else None
              for p in network_optimizer.param_groups[0]['params']])
    exp_avg_sq = _concat([network_optimizer.state[p]['exp_avg_sq'] if p.grad is not None else None
              for p in network_optimizer.param_groups[0]['params']])
    grads, clip_coef = _clip(exp_avg.div(exp_avg_sq.sqrt().add_(1e-8)), self.network_clip, use_data=False)'''

    '''
    # The momentum formulation
    moms = _concat([network_optimizer.state[p]['momentum_buffer'] if p.grad is not None else None
              for p in network_optimizer.param_groups[0]['params']])
    grads, clip_coef = _clip(moms, self.network_clip, use_data=False)'''

    dtheta = grads + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))

    return unrolled_model, clip_coef

  def step(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid,
          network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']
    self.optimizer.zero_grad()
    if unrolled: # and self.arch_step < 300:
        hidden = self._backward_step_unrolled(hidden_train, input_train, target_train, hidden_valid, input_valid,
                                              target_valid, eta, network_optimizer)  # TODO: I added network optimizer
    else:
        hidden = self._backward_step(hidden_valid, input_valid, target_valid)
    self.optimizer.step()
    self.arch_step += 1
    if self.arch_step % 100 == 0:
        print(f"step = {self.arch_step}")
    return hidden, None

  def _backward_step(self, hidden, input, target):
    loss, hidden_next = self.model._loss(hidden, input, target)
    loss.backward()
    return hidden_next

  def _backward_step_unrolled(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid, eta, network_optimizer): # TODO: I added network optimizer

    unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta,
                                                             network_optimizer) # TODO: I added network optimizer
    for parameter in unrolled_model.parameters():
        parameter.require_grad = False

    unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)

    unrolled_loss.backward()
    #if not self.diff_through_unrolled:  # TODO: I added this conditional return.
    #    return hidden_next
    for parameter in unrolled_model.parameters():
        parameter.require_grad = True

    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    if self.diff_through_unrolled:  # TODO: Only do the hessian approx if we are diffing through unrolled.
        dtheta = [v.grad for v in unrolled_model.parameters()]
        _clip(dtheta, self.network_clip)
        vector = [dt.data for dt in dtheta]
        implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta * clip_coef, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    return hidden_next

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, hidden, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss, _ = self.model._loss(hidden, input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss, _ = self.model._loss(hidden, input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

