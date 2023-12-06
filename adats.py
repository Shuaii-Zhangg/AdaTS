import math
import torch
from torch.optim.optimizer import Optimizer
version_higher = (torch.__version__ >= "1.5.0")

class AdaTS(Optimizer):
    r"""Implements AdaTS algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        iters(int, required): iterations
            iters = math.ceil(trainSampleSize / batchSize) * epochs
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
    """
    def __init__(self, params,  lr=1e-3, betas=(0.9, 0.999), eps=1e-8, final_lr=0.01,
                 weight_decay=0, iters=78200,  delta=1e-20, weight_decouple=False, fixed_decay=False ):

        if not 0.0 <= iters:
             raise ValueError("Invalid iters: {}".format(iters))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))

        defaults = dict(lr=lr, betas=betas, eps=eps, final_lr=final_lr, weight_decay=weight_decay, iters=iters, delta=delta)
        super(AdaTS, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
    def reset(self):
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                state = self.state[p]

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                beta1, beta2 = group['betas']
                iters = group['iters']
                delta = group['delta']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                max_exp_avg_var = state['max_exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                torch.max(max_exp_avg_var, (exp_avg_var.add_(group['eps'])) / bias_correction2, out=max_exp_avg_var)

                denom = group['lr'] / max_exp_avg_var.sqrt()
                final_lr = group['final_lr'] * group['lr'] / base_lr
                step_size = denom * ((math.cos((math.pi * state['step']) / (2 * iters))) ** 2) + ((math.sin((math.pi * state['step']) / (2 * iters))) ** 2) * final_lr + delta
                p.data.add_(-step_size * exp_avg *(1 / bias_correction1))

        return loss
