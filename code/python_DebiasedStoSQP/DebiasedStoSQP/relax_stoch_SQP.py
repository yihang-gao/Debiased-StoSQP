import numpy as np
from .useful_functions import *
import tracemalloc
import linecache
import os
import pycutest


def solve_relax_stoch_SQP(prob, prop_prob, variables, hyper):

    for j in range(hyper.max_iter):

        err = find_relaxing_param(prob, prop_prob, variables, hyper)
        if err:
            return err

        err = get_update_grad_hess(prob, prop_prob, variables, hyper)
        if err:
            return err

        err, B = make_hess_pd(variables, hyper)
        if err:
            return err

        err, d_xyz, d_dual_eq, d_dual_bound = solve_relax_sqp_subprob(B, prob, prop_prob, variables, hyper)
        if err:
            return err

        pk = d_xyz[0:prop_prob.dim_n]
        # step size calculation and parameters/vars update
        err, step_size = get_step_size(pk, B, variables, hyper)
        if err:
            return err

        err = update_vars(d_xyz, d_dual_eq, d_dual_bound, step_size, prop_prob, variables, hyper)
        if err:
            return err

        if (j + 1) % 100 == 0:
            err, kkt, cont = cal_kkt_res_cont(prob, prop_prob, variables, hyper)
            variables.store_kkt.append([kkt, cont])
            print(
                "finish {:d}-th iteration, KKT Residual is {:.3e}, feasibility is {:.3e}.".format(
                    j + 1, kkt, cont))

        variables.iter += 1

        if variables.iter == hyper.buffer_size:
            hyper.adaptive = False

    return False
