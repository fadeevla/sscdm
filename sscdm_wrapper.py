import math
import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim import SGD
from cd_optim import cd_optim
from torchvision import datasets, transforms



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

    def __init__(self, params, lr=required, cd_max_steps=required, method="cd"):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if cd_max_steps is not required and cd_max_steps < 0:
            raise ValueError("Invalid conjugate directions steps: {}".format(cd_max_steps))
        
        defaults = dict(lr=lr, cd_max_steps=cd_max_steps)
        self.all_layers_shapes = {}

        self.cd_optim_obj = cd_optim(method = method, lr = lr)
        self.last_step = None
        self.method = method
        
        super(SSCDM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SSCDM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def get_layer_gradient(self, layer):
        """Get gradient of a DNN layer
        Arguments:
            layer (,mandatory): a DNN layer (parameters' group)
        Returns:
            torch.Tensor (size of given layer/group parameters)
        """
        grad = None
        grad = layer.grad.data
        shape = grad.shape
        flatten_grad = grad.reshape(-1)

        return shape, flatten_grad
    
    def get_global_gradient(self, dnn):
        """Get gradients of each DNN layer and append vector
        Arguments:
            whole DNN including all layers (self.param_groups[0]['params'])
        Returns:
            torch.Tensor(number of layers)
        """
        grad = torch.Tensor(0)
        if self.all_layers_shapes == {}:
            
            for i,layer in enumerate(dnn):
                self.all_layers_shapes[i] = layer.shape
        for i, layer in enumerate(dnn):
            g = self.get_layer_gradient(layer)
            grad = torch.cat([grad,g[1]])
        
            
        return grad

    def vector2vlayers(self, vector):
        """Unpack global vector into layers vectors
        Arguments:
            vector (tuple): 
        Return:
            dict of tuples, indexed by number of DNN layer
        """
        vlayers = {}
        start = 0
        end = 0
        for i, lshape in enumerate(self.all_layers_shapes.values()):
            size = torch.prod(torch.Tensor(list(lshape))) #todo: consider replacing conversion to list
            end += int(size)
            vlayers[i] = vector[start:end].reshape(lshape)
            start = end

        return vlayers
    def test_sscdm(self):
        grad = self.get_global_gradient(self.param_groups[0]['params'])
        temp = self.vector2vlayers(grad)

        return 0
    def step_global(self, closure = None):
        """Performs global optimization step across all layers
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            grads = self.get_global_gradient(group['params'])
            deltas = self.cd_optim_obj.get_step(grads, method=self.method)
            self.last_step = deltas
            deltas_tensor = self.vector2vlayers(deltas)
            for i,p in enumerate(group['params']):
                p = p.add(deltas_tensor[i])
                #x.add_(1, deltas_tensor[i])
            
        return deltas_tensor
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
            self.step_global()
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

    def get_mini_batch(
        dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), 
        is_fulldataset=1, batch_size=None, current_dataset = None, action=0):
        """
        Arguments:
            dataset - (Dataloader with parameter batch_size equal to full dataset) total dataset
            is_fulldataset - (boolean)
            batch_size - (int) size of batch
            current_dataset - iterator 
            action - 0,1,2 (0-return current dataset, 1-full update/create of dataset, 2-extend dataset)
        """
        #_, (data, target) = [dataset][0]
        #data, target = data.to(device), target.to(device)
        if action == 0:
            return current_dataset
        if action == 1:
            #(sampler) = SequentialSampler
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = args.test_batch_size, shuffle=True, **kwargs)

"""
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
"""

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