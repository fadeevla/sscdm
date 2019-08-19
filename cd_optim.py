import torch

class cd_optim():
    """Conjugate directions optimization for deep learning

    Arguments:
        K_max (int, optional): maximum number of conjugate directions for optimization (default: 1)
        lr (float, optional): learning rate (default: 1e-3)
        M_Max (int, optional): window for momentum accumulation (default None if None M_Max equal to K_Max)
        method (str) optimiaztion method (sscdm, adam, gd, rmsprop)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self, K_max=1, lr=0.001, M_max=None, method = 'sscdm', adam_rho1=0.9, adam_rho2=0.999, adam_delta=0.00000001):
        self.last_step=None
        self.empty = None
        self.lr = lr
        self.supported_optim_methods = ['gd','adam','sscdm']
        assert method in self.supported_optim_methods, f'method must be one of supported optimization methods {self.supported_optim_methods}'
        
        #self.optim_calls number of performed optimization steps
        self.optim_calls = None
        self.K_max = K_max
        self.K = None
        self.g = None
        self.d = None
        self.n = None
        self.alpha = None
        self.c = None
        self.M_max = M_max
        self.adam_s = None # s vector of Adam
        self.adam_r = 0 # r vector of Adam
        self.method = method
  
        self.method = 'gd'
        self.adam_rho1 = adam_rho1
        self.adam_rho2 = adam_rho2
        self.adam_delta = adam_delta
        self.adam_t = 0
    
        if M_max is None:
            M_max = self.K_max
    
    def _gd_step(self, grad, epsilon):
        return -epsilon * grad 

    def _adam_step(self, grad, epsilon):
        if type(self.adam_s) == type(None):
            self.adam_s = (1 - self.adam_rho1) * grad
        else:
            self.adam_s = self.adam_rho1 * self.adam_s + (1 - self.adam_rho1) * grad
        self.adam_r = self.adam_rho2 * self.adam_r + (1 - self.adam_rho2) * grad.norm()**2
        self.adam_t += 1
        self.adam_s /= (1-self.adam_rho1**self.adam_t)
        self.adam_r /= (1-self.adam_rho2**self.adam_t)
        lr = - epsilon / torch.sqrt(self.adam_r + self.adam_delta)
        step = lr * self.adam_s
        
        
        print(f'adam_t={self.adam_t}=================> adam_s.norm()={self.adam_s.norm()}')
        return step

    def get_step(self, g, method = None):
        """Performs optimization step on function with parameters in x given gradient g and returns optimized x
        Arguments:
            method: str - Current optimization method (default: None, if None it performs a method in supported_optim_methods additionally to sscdm, sscdm is performed only once)
            g: torh.Tensor - Gradient
        """
        #assert isinstance(x, torch.Tensor), 'x must be torch.Tensor'
        assert isinstance(g, torch.Tensor), 'g must be torch.Tensor'
        assert method in self.supported_optim_methods
        step = None
        if 'gd' == method:
            step = self._gd_step(grad=g, epsilon = self.lr)
        elif 'adam' == method:
            step = self._adam_step(grad=g, epsilon = self.lr)
        self.last_step = step
        return step

def test():
    test_optim = cd_optim()
    x = torch.rand(3,3,3,3)
    test_optim.get_step(x = x, g = None)
    return test_obj.get_step(g=None)

if __name__ == "__main__":
    
    x = test()
    print(x)