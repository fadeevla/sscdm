MNIST dataset format training set of 60000 28x28 grayscale images (1 color channel) 
The original MNIST images are normalised (see trasform=...) with mean of 0.1307 and std of 0.3081
values of mean and std were computed for MNIST images (todo: assess dataset normalization effect 
for sscdm, according to A.I. data normalization is )

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

(below derived from http://yann.lecun.com/exdb/mnist/):

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

## Mathematics

Norm used is Frobenius norm (known also asL2 norm)

$$|A| = \sqrt {\sum a_i^2}$$

## Pytorch

working with states to save previous state of parameter, e.g. save gradients, etc.

    torch/optim/adadelta.py

        state = self.state[p]
        state['step'] += 1

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p.data)
            state['acc_delta'] = torch.zeros_like(p.data)

        square_avg, acc_delta = state['square_avg'], state['acc_delta']
        rho, eps = group['rho'], group['eps']

        state['step'] += 1
    torch/optim/adam.py
        state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

    torch/optim/asgd.py
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['eta'] = group['lr']
            state['mu'] = 1
            state['ax'] = torch.zeros_like(p.data)

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # decay term
        p.data.mul_(1 - group['lambd'] * state['eta'])

        # update parameter
        p.data.add_(-state['eta'], grad)

        # averaging
        if state['mu'] != 1:
            state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
        else:
            state['ax'].copy_(p.data)

        # update eta and mu
        state['eta'] = (group['lr'] /
                        math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
        state['mu'] = 1 / max(1, state['step'] - group['t0'])
-----

parameter update
    pytorch/torch/optim/adadelta.py
        square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        p.data.add_(-group['lr'], delta)
        acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)
    torch/optim/adagrad.py
        clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
        state['sum'].addcmul_(1, grad, grad)
        std = state['sum'].sqrt().add_(1e-10)
        p.data.addcdiv_(-clr, grad, std)

    torch/optim/adam.py
        denom = exp_avg_sq.sqrt().add_(group['eps'])
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        p.data.addcdiv_(-step_size, exp_avg, denom)