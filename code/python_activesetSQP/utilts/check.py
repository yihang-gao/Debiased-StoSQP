import numpy as np
import pycutest


class CheckRequirement:
    def __init__(self, max_n, max_m):
        # requirements of dims of objective and constraints
        self.max_n = max_n
        self.max_m = max_m


class PropertyProblem:
    def __init__(self, prob, seed):
        # properties of the problem
        self.dim_n = prob.n
        self.dim_m = prob.m
        self.name = prob.name

        # lower and upper bounds for variables
        self.xl = np.minimum(prob.bl, prob.bu)
        self.xu = np.maximum(prob.bl, prob.bu)
        self.ixl = np.where(self.xl > -1e15)
        self.ixu = np.where(self.xu < 1e15)
        # self.ixl = (np.array([], dtype=np.int64),)
        # self.ixu = (np.array([], dtype=np.int64),)
        self.nxl = np.shape(self.ixl)[1]
        self.nxu = np.shape(self.ixu)[1]
        self.nxlu = self.nxl + self.nxu

        # lower and upper bounds for constaints
        self.cl = np.minimum(prob.cl, prob.cu)
        self.cu = np.maximum(prob.cl, prob.cu)
        self.ice = np.where(self.cl == self.cu)
        self.icl = np.where(np.logical_and(-1e15 < self.cl, self.cl != self.cu))
        self.icu = np.where(np.logical_and(self.cu < 1e15, self.cl != self.cu))
        self.mce = np.shape(self.ice)[1]
        self.mcl = np.shape(self.icl)[1]
        self.mcu = np.shape(self.icu)[1]
        self.mc = self.mce + self.mcl + self.mcu
        self.mieq = self.mcl + self.mcu + self.nxlu

        # name
        self.name = prob.name

        # set seed
        self.seed = seed


def check(prob, require: CheckRequirement):
    correct = True

    # problem dimension should not be greater than required
    if prob.n > require.max_n:
        correct = False
        print("dim_n too large, exit.")
        return correct

    # req on numbers of constraints
    if prob.m > require.max_m:
        correct = False
        print("dim_m too large, exit.")
        return correct

    if prob.m == 0:
        correct = False
        print("no constraints, exit.")
        return correct

    # check the variable types, must be real(0)
    if 1 in prob.vartype or 2 in prob.vartype:
        correct = False
        print("variable type not correct, exit.")
        return correct

    # check whether constant objective
    count = 0
    try:
        x0 = prob.x0
        f = prob.obj(x0)
        for i in range(10):
            x_other = x0 + np.random.normal(size=np.shape(x0))
            f_other = prob.obj(x_other)
            if np.abs(f - f_other) > 1e-15:
                count += 1

        if count == 0:
            correct = False
            print("constant function, exit.")
            return correct
    except:
        print("An error occurred when checking the constant objective")
        correct = False
        return correct

    # check whether the function is smooth enough
    try:
        x0 = prob.x0
        g, J = prob.lagjac(x0)
        H = prob.ihess(x0)
        # if not g or not J or not H:
        #     correct = False
        #     print("gradient or hessian does not exist, exit.")
        #     return correct
        #
        # if np.isnan(g).any() or np.isnan(J).any() or np.isnan(H).any():
        #     correct = False
        #     print("gradient or hessian does not exist, exit.")
        #     return correct
        #
        # for i in range(prob.m):
        #     H = prob.ihess(x0, cons_index=i)
        #     if not H or np.isnan(H).any():
        #         print("hessian does not exist for constaints, exit.")
        #         correct = False
        #         return correct
    except:
        print("An error occurred when checking the smoothness of objective")
        correct = False
        return correct

    return correct
