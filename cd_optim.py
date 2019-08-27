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
        self.supported_optim_methods = ['gd','adam','sscdm','cdwm','cd']
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

        #cdwm variables
        self.cdwm_C = None #vector curvature per coordinate aka optimized parameter
        self.cdwm_last_step = None #vector last cdwm step
        self.cdwm_Total_step = None #vector cumulative step per coordinate
        self.cdwm_t = 0 #counter number of steps after last reset (g0,C, total_step)
        self.cdwm_g0 = None #initial gradient
        # parameters of cojudate directions method: 
        self.cd_cvectors = {} # conjugate vectors list: [m][i]
        self.cd_nvectors = {} # normal vectors list: [m][i]
        self.cd_k = 0 # number of vectors: [n]
        self.cd_N_max = 2 # maximal number of vectors
        self.cd_dAd = {} # list of curvatures along conjugate vectors: [n]
        self.cd_beta = {} # list of : [n][l]
        self.cd_stepD = {} # steps along d-vectors
        self.cd_gd = {} # list of scalars [n]
        self.cd_grad0norm = None


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
    
    def _cdwm_step_atomic(self, step, grad, grad_prev, epsilon, step_prev, block_end, block_start=0):
        for i in range(block_start, block_end): #todo: review pytroch way to iterate over variables
            #delta_grad = grad[i]-self.cdwm_g0[i]
            delta_grad = grad[i]-grad_prev[i]
            
            C_prev = self.cdwm_C[i].clone()
            if abs(delta_grad) < 0.00000001:
                self.cdwm_C[i] *= 1
            else:
                #self.cdwm_C[i] = self.cdwm_Total_step[i] / (grad[i]-self.cdwm_g0[i])
                self.cdwm_C[i] = self.cdwm_last_step[i] / (grad[i]-grad_prev[i])
            if abs(self.cdwm_C[i]) == 0:
                self.cdwm_C[i] = C_prev
            if abs(self.cdwm_C[i]) > 2 * abs(C_prev):
                self.cdwm_C[i] *= 2 * abs(C_prev) / abs(self.cdwm_C[i])
            elif abs(self.cdwm_C[i]) < 0.5 * abs(C_prev):
                self.cdwm_C[i] *= 0.5 * abs(C_prev) / abs(self.cdwm_C[i])
            if abs(self.cdwm_C[i])>1:
                self.cdwm_C[i] /= 2
            #l = self.cdwm_C[i].sign() * max(abs(self.cdwm_C[i]), epsilon)
            step[i] = - self.cdwm_C[i].sign() * grad[i] * self.cdwm_C[i]
            #if self.cdwm_C[i] < 0:
            #    step[i] = grad[i] * l
            #else:
            #    step[i] = -grad[i] * l
                
        
        return step

    def _cdwm_step(self, grad, epsilon, step_prev):
    
        if type(self.cdwm_g0) == type(None):
            step = -grad*epsilon
            self.cdwm_g0 = grad.clone()
            self.cdwm_last_step = torch.Tensor().new_zeros(size = grad.shape)
            self.cdwm_Total_step = torch.Tensor().new_zeros(size = grad.shape)
            
            self.cdwm_C = torch.Tensor(grad.shape)
            self.cdwm_C = self.cdwm_C.new_full(size=grad.shape, fill_value=epsilon)
            

        else:
            self.cdwm_last_step = step_prev.clone()
            self.cdwm_Total_step += self.cdwm_last_step

            step = torch.Tensor().new_zeros(size = grad.shape)
        #def _cdwm_step_atomic(self, grad, epsilon, step_prev, block_end, block_start=0):
            self._cdwm_step_atomic(out = step, grad=grad, grad_prev = self.cdwm_grad_prev, epsilon=epsilon, step_prev=step_prev, block_start=0, block_end=int(grad.shape[0]))
        self.cdwm_grad_prev = grad
        return step

    def _cdwm_step_old(self, grad, epsilon, step_prev):
    
        if type(self.cdwm_g0) == type(None):
            step = -grad*epsilon
            self.cdwm_g0 = grad.clone()
            self.cdwm_last_step = torch.Tensor().new_zeros(size = grad.shape)
            self.cdwm_Total_step = torch.Tensor().new_zeros(size = grad.shape)
            
            self.cdwm_C = torch.Tensor(grad.shape)
            self.cdwm_C = self.cdwm_C.new_full(size=grad.shape, fill_value=epsilon)
            

        else:
            self.cdwm_last_step = step_prev.clone()
            self.cdwm_Total_step += self.cdwm_last_step

            step = torch.Tensor().new_zeros(size = grad.shape)
            
            for i in range(int(grad.shape[0])): #todo: review pytroch way to iterate over variables
                #delta_grad = grad[i]-self.cdwm_g0[i]
                delta_grad = grad[i] - self.cdwm_grad_prev[i]
                
                C_prev = self.cdwm_C[i].clone()
                if abs(delta_grad) < 0.00000001:
                    self.cdwm_C[i] *= 1
                else:
                    #self.cdwm_C[i] = self.cdwm_Total_step[i] / (grad[i]-self.cdwm_g0[i])
                    self.cdwm_C[i] = self.cdwm_last_step[i] / (grad[i]-self.cdwm_grad_prev[i])
                if abs(self.cdwm_C[i]) == 0:
                    self.cdwm_C[i] = C_prev
                if abs(self.cdwm_C[i]) > 2 * abs(C_prev):
                    self.cdwm_C[i] *= 2 * abs(C_prev) / abs(self.cdwm_C[i])
                elif abs(self.cdwm_C[i]) < 0.5 * abs(C_prev):
                    self.cdwm_C[i] *= 0.5 * abs(C_prev) / abs(self.cdwm_C[i])
                #if abs(self.cdwm_C[i])>1:
                #    self.cdwm_C[i] /= 2
                #l = self.cdwm_C[i].sign() * max(abs(self.cdwm_C[i]), epsilon)
                step[i] = - self.cdwm_C[i].sign() * grad[i] * self.cdwm_C[i]
                #if self.cdwm_C[i] < 0:
                #    step[i] = grad[i] * l
                #else:
                #    step[i] = -grad[i] * l
                
        self.cdwm_grad_prev = grad
        return step
    
    def _cd_step(self, grad, epsilon, step_prev):
        
        """
        self.cd_cvectors = None # conjugate vectors list: [m][i]
        self.cd_nvectors = None # normal vectors list: [m][i]
        self.cd_k = 0 # number of vectors: [n]
        self.cd_N_max = 2 # maximal number of vectors
        self.cd_dAd # list of curvatures along conjugate vectors: [n]
        self.cd_beta # list of : [n][l]
        self.cd_stepD # steps along d-vectors
        self.cd_gd = None # list of scalars [n]
        """
        k = self.cd_k
        step = None

        if k == 0:
            gradnorm = grad.norm()
            self.cd_grad0norm = gradnorm
            self.cd_cvectors[k] = - grad / gradnorm
            self.cd_nvectors[k] = self.cd_cvectors[k].clone()
            step = epsilon * self.cd_cvectors[k]
            self.cd_stepD[k] = epsilon
            self.cd_k += 1
            self.cd_gd[k] = torch.dot(grad, self.cd_cvectors[k])

        elif k > 0:
            gamma = torch.dot(grad, self.cd_nvectors[k - 1])
            
            gd = torch.dot(grad, self.cd_cvectors[k - 1])
            c = self.cd_stepD[k - 1] / (gd - self.cd_gd[k-1])
            alpha = - gd * c
            
            self.cd_nvectors[k] = ( - grad + gamma * self.cd_nvectors[k-1] )
            nnorm = self.cd_nvectors[k].norm()
            g_estimate = abs(nnorm * (self.cd_stepD[k - 1] + alpha) / self.cd_stepD[k - 1])
            if c<=0:
                self.cd_k = 0
                return -alpha * self.cd_cvectors[k-1]
            elif g_estimate < self.cd_grad0norm * 0.01:
                self.cd_k = 0
                return alpha * self.cd_cvectors[k-1]
            elif k>0:
                
                self.cd_gd[k] = gd
                self.cd_stepD[k] = alpha
                self.cd_cvectors[k] = self.cd_cvectors[k - 1]
                self.cd_nvectors[k] = self.cd_nvectors[k-1]
                if alpha <= 0:
                    self.cd_k = 0
                else:
                    self.cd_k += 1
                return alpha * self.cd_cvectors[k-1]

            self.cd_nvectors[k] /= nnorm
            self.cd_beta[k-1] = nnorm / ( gd - self.cd_gd[k-1] )
            self.cd_cvectors[k] = (self.cd_nvectors[k] + self.cd_beta[k-1] * self.cd_cvectors[k-1])
            
            normcv = torch.sqrt(1+self.cd_beta[k-1] * self.cd_beta[k-1])
            self.cd_cvectors[k] /= normcv
            self.cd_stepD[k] = epsilon/10
            step = alpha * self.cd_cvectors[k-1] + self.cd_cvectors[k]*self.cd_stepD[k]
            self.cd_gd[k] = torch.dot(grad, self.cd_cvectors[k])
            self.cd_k += 1
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
        elif 'cdwm' == method:
            step = self._cdwm_step_old(grad=g, epsilon = self.lr, step_prev = self.last_step)
        elif 'cd' == method:
            step = self._cd_step(grad=g, epsilon = self.lr, step_prev = self.last_step)
        else:
            raise('not supported optimization method ', method)
        self.last_step = step
        return step

def test():
    test_optim = cd_optim()
    x = torch.rand(3,3,3,3)
    testresult = test_optim.get_step(g = torch.Tensor([1]), step_prev=None, method = 'cd')
    return testresult

if __name__ == "__main__":
    
    x = test()
    print(x)