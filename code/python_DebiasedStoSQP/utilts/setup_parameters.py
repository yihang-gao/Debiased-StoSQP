import numpy as np


class HyperParameters:
    def __init__(self, max_n, max_m, noise_type="gaussian", noi_std_grad_hess=[1e-1, 1e-1], noi_stu_t_freed=[3, 3],
                 decay_grad=0.5,
                 decay_var=0.8,
                 decay_relax=0.5, tol_relax_param=1e-3, min_eig=1e-1, max_eig=1e2, adaptive=False, repeat=200,
                 max_iter=int(1e5)):
        self.max_n = max_n
        self.max_m = max_m
        self.noise_type = noise_type
        self.noi_std_grad_hess = noi_std_grad_hess
        self.noi_stu_t_freed = noi_stu_t_freed
        self.decay_grad = decay_grad
        self.decay_var = decay_var
        self.decay_relax = decay_relax

        self.repeat = repeat
        self.max_iter = max_iter
        self.buffer_size = max_iter // 10

        self.tol_relax_param = tol_relax_param
        self.tol_relax_loss = 1e-6

        self.min_eig = min_eig
        self.max_eig = max_eig

        self.sig = 0.1
        self.adaptive = adaptive
        self.ratio_pen = 0.1
        self.ratio_xi = 0.1
        self.lip_gradf = 10.0
        self.lip_gradc = 10.0
        self.varrho = 1.0


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
        self.relax = 1.0

        self.avg_grad = np.zeros(shape=(prop_prob.dim_n,))
        self.avg_hess = np.identity(n=prop_prob.dim_n)

        self.cont = np.ones(shape=(prop_prob.mc,))

        # penalty parameters
        self.pen = 0.0
        self.xi = 10.0

        self.store_kkt = []

