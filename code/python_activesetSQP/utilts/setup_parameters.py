import numpy as np


class HyperParameters:
    def __init__(self, max_n, max_m, noise_type="gaussian", noi_std_grad_hess=[1e-1, 1e-1], noi_stu_t_freed=[3, 3],
                 min_eig=1e-1, max_eig=1e2, Newton=False, repeat=1,
                 max_iter=int(1e5)):
        self.max_n = max_n
        self.max_m = max_m
        self.noise_type = noise_type
        self.noi_std_grad_hess = noi_std_grad_hess
        self.noi_stu_t_freed = noi_stu_t_freed

        self.repeat = repeat
        self.max_iter = max_iter

        self.min_eig = min_eig
        self.max_eig = max_eig

        # the big O constant in all derived bounds
        self.c = 2.0

        self.p_grad = 0.1
        self.p_f = 0.1
        self.k_grad = 1.0
        self.x_grad = 1.0
        self.alpha_max = 1.5
        self.x_err = 1.0
        self.eta = 1e-4
        self.rho = 2.0
        self.beta = 0.3
        self.k_f = 0.05
        self.x_f = 1.0

        self.Newton = Newton


class Variables:
    def __init__(self, prob, prop_prob):
        # all parameters to be stored
        self.x = prob.x0

        self.dual_general = np.zeros(shape=(prop_prob.mc,))

        self.dual_general[0:prop_prob.mce] = prob.v0[prop_prob.ice]
        self.dual_general[prop_prob.mce:(prop_prob.mce + prop_prob.mcl)] = -np.minimum(prob.v0[prop_prob.icl], 0.0)
        self.dual_general[(prop_prob.mce + prop_prob.mcl):] = np.maximum(prob.v0[prop_prob.icu], 0.0)



        self.dual_bound = np.zeros(shape=(prop_prob.nxlu,)) + 1.0

        self.iter = 0
        self.xi_1 = 1
        self.xi_2 = 1
        self.alpha = 1e-1

        self.cont = np.ones(shape=(prop_prob.mc,))

        self.store_kkt = []


class Temp_Variables:
    def __init__(self, prop_prob):
        self.grad = np.zeros(shape=(prop_prob.dim_n,))
        self.grad_lag = np.zeros(shape=(prop_prob.dim_n,))
        self.hess_lag = np.zeros(shape=(prop_prob.dim_n, prop_prob.dim_n))
        self.J = np.zeros(shape=(prop_prob.dim_m, prop_prob.dim_n))
        self.Jac = np.zeros(shape=(prop_prob.mc, prop_prob.dim_n))

        self.noi_grad1 = np.zeros(shape=(prop_prob.dim_n,))
        self.noi_hess1 = np.zeros(shape=(prop_prob.dim_n, prop_prob.dim_n))
        self.noi_grad2 = np.zeros(shape=(prop_prob.dim_n,))
        self.noi_hess2 = np.zeros(shape=(prop_prob.dim_n, prop_prob.dim_n))

        self.delta = 1.0
        self.v = 1.0
        self.eps = 1e-2
