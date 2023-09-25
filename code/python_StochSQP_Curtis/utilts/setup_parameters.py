import numpy as np


class HyperParameters:
    def __init__(self, max_n, max_m, noise_type="gaussian", noi_std_grad_hess=[1e-1, 1e-1], noi_stu_t_freed=[3, 3],
                 decay_var=0.8,
                 min_eig=1e-1, max_eig=1e2, adaptive_lip=True, adaptive_hess=False, repeat=200,
                 max_iter=int(1e5)):
        self.max_n = max_n
        self.max_m = max_m
        self.noise_type = noise_type
        self.noi_std_grad_hess = noi_std_grad_hess
        self.noi_stu_t_freed = noi_stu_t_freed
        self.decay_var = decay_var


        self.repeat = repeat
        self.max_iter = max_iter
        self.buffer_size = max_iter // 10

        self.min_eig = min_eig
        self.max_eig = max_eig

        self.mu = 1e-8
        self.sig = 0.1
        self.ratio_tau = 0.01
        self.ratio_xi = 0.01
        self.lip_gradf = 10.0
        self.lip_gradc = 10.0
        self.theta = 1e4
        self.eta = 0.5

        self.adaptive_hess = adaptive_hess
        self.adaptive_lip = adaptive_lip
        self.adaptive = True

        self.ratio_lip_gradf = 0.1
        self.ratio_lip_gradc = 0.1
        self.num_est_grad = 3


class Variables:
    def __init__(self, prob, prop_prob):
        # all parameters to be stored
        self.x = prob.x0
        self.y = np.zeros(shape=(prop_prob.mcl,)) - 1.0
        self.z = np.zeros(shape=(prop_prob.mcu,)) + 1.0
        self.xyz = np.append(self.x, np.append(self.y, self.z))
        self.dual_eq = np.zeros(shape=(prop_prob.mc,))
        size_dual_bound = prop_prob.nxl + prop_prob.nxu + prop_prob.mcl + prop_prob.mcu
        self.dual_bound = np.zeros(shape=(size_dual_bound,))
        self.iter = 0

        self.est_grad = np.zeros(shape=(prop_prob.dim_n,))
        self.est_hess = np.identity(n=prop_prob.dim_n)

        self.cont = np.ones(shape=(prop_prob.mc,))

        # penalty parameters
        self.tau = 0.1
        self.xi = 1.0

        self.store_kkt = []

