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
            loss = closure()

        for group in self.param_groups:
                        
            cd_max_steps = group['cd_max_steps']
            
            #p is space of particular neural network layer parameters 
            for p in group['params']:
                print(f"p.data.size()={p.data.size()}")
                if p.grad is None:
                    continue
                state = self.state[p]
                #print('p min=', torch.min(p.data))
                #todo: check for distribution of weights/parameters
                #todo: clarify if negative weights/params makes sense
                # State initialization, state is used to save previous step variables e.g. gradient
                if len(state) == 0:
                    state['step'] = 0

                if state['step'] == 0:
                    state['g0'] = p.grad.data.clone()
                    assert (state['g0']==state['g0']).all(), f"p.grad.data={p.grad.data}\np.data={p.data}"
                    state['d0'] = state['n0'] = -state['g0'] / state['g0'].norm()

                    #update parameters
                    state['step_size_g0'] = group['lr'] / 1000 * state['g0'].norm()
                    p.data.add_(state['step_size_g0'], state['d0'])
                    assert (p.data==p.data).all(), f"after p.data.size={p.data.size()}, state['step']={state['step']} \nstate['d0']={state['d0']}\nstate['g0']={state['g0']}"
                else:
                    
                    state['g'+str(state['step'])] = p.grad.data.clone()
                    #state['gamma_k' + str(state['step']) + '_k' + str(state['step']-1)] = torch.mul(state['g' + str(state['step'])], state['n' + str(state['step']-1)])
                    #state['n_1'] = torch.addcmul(state['g'+str(state['step'])], state['gamma_k' + str(state['step']) + '_k' + str(state['step']-1)])
                    
                    #alpha_numerator = torch.mul(state['g' + str(state['step'])], state['d' + str(state['step']-1)])
                    g=state['g' + str(state['step'])]
                    d_prev=state['d' + str(state['step']-1)]
                    g_prev=state['g' + str(state['step']-1)]
                    step_size_prev = state['step_size_g'+ str(state['step']-1)]
                    assert g.shape==d_prev.shape==g_prev.shape
                    s=g.shape
                    if len(s)==4:
                        a=s[0]; b=s[1]; c=s[2]; d=s[3]
                        alpha_numerator = torch.bmm(g.view(a,b,c*d),d_prev.view(a,c*d,b))
                        abc = torch.bmm(g_prev.view(a,b,c*d),d_prev.view(a,c*d,b))
                        alpha_denominator = alpha_numerator - abc
                        c = step_size_prev / alpha_denominator
                    if len(s)==2:
                        a=s[0]; b=s[1]
                        alpha_numerator = torch.mm(g,d_prev.view(b,a))
                        abc = torch.mm(g_prev,d_prev.view(b,a))
                        alpha_denominator = alpha_numerator - abc
                        c = step_size_prev / alpha_denominator 
                    if len(s)==1:
                        a=s[0]
                        alpha_numerator = torch.dot(g,d_prev)
                        abc = torch.dot(g_prev,d_prev)
                        alpha_denominator = alpha_numerator - abc
                        c = step_size_prev / alpha_denumerator
                    if c<0:
                        print('consider correcting concave c')
                    #Alpha_k,k-1
                    state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)] = -(torch.div(alpha_numerator, alpha_denominator))*step_size_prev
                    #fix nan values in tensor, consider removing comment of below line
                    #state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)][torch.isnan(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)])] = 0
                    #print('g.shape=',s)
                    #print('step=', state['step'])
                    #print('|alpha| =', torch.norm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]))
                    
                    if torch.norm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]) > 2 or torch.norm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]) < -2:
                        #print('big alpha!!!! current step=',state['step'])
                        #print('alpha.size=',state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].size())
                        #print(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].dtype)
                        state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)] = state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].new_ones(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].size())
                        state['step'] = cd_max_steps-1
                    if len(s)==4:
                        assert (p.data==p.data).all(), f"before p.data.size={p.data.size()}, state['step']={state['step']} \np.data={p.data}"
                        p.data.add_(group['lr'], torch.bmm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].view(a,b,b), state['d0'].view(a,b,c*d)).view(a,b,c,d))
                        assert (p.data==p.data).all(), 'after'
                        #print('len(s)=',len(s), ' pdelta=',torch.norm(torch.bmm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].view(a,b,b), state['d0'].view(a,b,c*d)).view(a,b,c,d)))
                        #print('|alpha|=',torch.norm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]))
                        #print('alpha_numerator=',torch.norm(alpha_numerator))
                        #print('alpha_denominator=',torch.norm(alpha_denominator))
                        #print('alphanum_size=',alpha_numerator.size())
                    if len(s)==2:
                        
                        p.data.add_(group['lr'], torch.mm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)], state['d0']))
                        #print('len(s)=',len(s), ' pdelta=',torch.norm(torch.mm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)], state['d0'])))
                        #print('|alpha|=',torch.norm(state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]))
                        #alpha turns to nan
                        #print('|alpha_numerator*delta|=',torch.norm(alpha_numerator*group['lr']))
                        #print('|alpha_denominator|=',torch.norm(alpha_denominator))
                        #print('alphanum_size=',alpha_numerator.size())
                        #print('alphadenom_size=',alpha_denominator.size())
                        #print('|alpha_dup|=',torch.norm(torch.div(alpha_numerator*group['lr'],alpha_denominator)))
                        #print('detect_nan=',torch.div(alpha_numerator*group['lr'],alpha_denominator)!=torch.div(alpha_numerator*group['lr'],alpha_denominator))
                        #print(p)
                        #assert (alpha_denominator==torch.div(alpha_numerator*group['lr'],alpha_denominator)).any()
                        assert (p.data==p.data).all()
                        #assert (p.data==math.nan).all()
                        pass
                    if len(s)==1:
                        #print('alphs_size=',state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)].size())
                        #print('d0=',state['d0'].size())
                        p.data.add_(group['lr'], state['alpha_k' + str(state['step']) + '_k' + str(state['step']-1)]*state['d0'])
                assert (p.data==p.data).all(), f"after p.data.size={p.data.size()}, state['step']={state['step']} \np.data={p.data}"
                if state['step'] == cd_max_steps-1:
                    #print(f"Step #{state['step']} reached max steps of {cd_max_steps}, start over.")
                    
                    self.state[p] = {}
                    #print("parameter's state set to {}")
                    continue
                
                state['step'] += 1
        return loss
if __name__ == "__main__":
    print('hi\n')
    pass

"""
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
"""