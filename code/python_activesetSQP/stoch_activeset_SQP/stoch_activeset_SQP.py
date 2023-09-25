import numpy as np
from .useful_functions import *
import tracemalloc
import linecache
import os
import pycutest


def solve_stoch_activeset_SQP(prob, prop_prob, variables, temp_variables, hyper):

    for j in range(hyper.max_iter):

        # step0: update temporal variables
        err = update_temp_variables(prob, prop_prob, variables, temp_variables)
        if err:
            return err

        # step1: estimate derivatives
        err = estimate_derivatives(prob, prop_prob, variables, temp_variables, hyper)
        if err:
            return err

        # step2: set epsilon
        err = set_epsilon(prob, prop_prob, variables, temp_variables, hyper)
        if err:
            return err

        # step3: decide step
        err = decide_step(prob, prop_prob, variables, temp_variables, hyper)
        if err:
            return err

        # step4-5: estimate merit function and then do the line search
        err = estimate_merit_function(prob, prop_prob, variables, temp_variables, hyper)
        if err:
            return err

        # check kkt residual and feasibility error
        if (j + 1) % 100 == 0:
            err, kkt, cont = cal_kkt_res_cont(prob, prop_prob, variables, temp_variables)
            variables.store_kkt.append([kkt, cont])
            print(
                "finish {:d}-th iteration, KKT Residual is {:.3e}, feasibility is {:.3e}.".format(
                    j + 1, kkt, cont))

        variables.iter += 1

    return False
