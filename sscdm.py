import math
import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim import SGD

class SSCDM(Optimizer):
    r"""Implements SSCDM algorithm proposed by Manevich, Boudinov `An efficient conjugate directions method_ without
linear minimization`, 2000.
    Arguments (todo: update):
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups        
        cd_max_steps (integer): number of conjugate directions, if set to 1 SSCDM is simplified to gradient descent method
        lr (float, optional): learning rate (default: 1e-3)
        
    .. _An efficient conjugate directions method\: Manevich, A. I., & Boudinov, E. (2000). 
    An efficient conjugate directions method without linear minimization. 
    Nuclear Instruments and Methods in Physics Research Section A: 
    Accelerators, Spectrometers, Detectors and Associated Equipment, 
    455(3), 698-705.

    Example:
        >>> optimizer = torch.optim.SSCDM(model.parameters(), cd_max_steps=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
 
    """

    def __init__(self, params, lr=required, cd_max_steps=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if cd_max_steps is not required and cd_max_steps < 0:
            raise ValueError("Invalid conjugate directions steps: {}".format(cd_max_steps))
        
        defaults = dict(lr=lr, cd_max_steps=cd_max_steps)
        
        super(SSCDM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SSCDM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    def g_k(self, p):
        return p.grad.data
    def n_k(self, p):
        return -g_k(p)/p.L2.data
    def d_k(self,p):
        return nk(p)
    def sigma_k(self,p, _lr):
        return _lr*g_k.L2
    def lambda_k_1(self):
        return 0
       
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure() #only placeholder todo: continue
        if not sigma_k:
            sigma_k = p.defaults['lr']
        print(self.param_groups)
        temp_step = sigma_k*d_n+lambda_k_1*d_k_1

        for group in self.param_groups:
                        
            cd_max_steps = group['cd_max_steps']
            print(group['params'])
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                x1 = g = d_p
                print(f'p.shape={p.shape}')
                for k in p.data:
                    print(f'k={k}')

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
if __name__ == "__main__":
    print('hi\n')
    pass