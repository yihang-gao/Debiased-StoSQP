# load problems
import pycutest
import numpy as np

def load_problems(n, m, num):
    # n: the maximal dimentions for objective; m: number of eq-constraints; num: number of problems returned
    problems_name = []
    count = 0
    potential_problems_name = pycutest.find_problems(n=[1, n], m=[1, m])
    for prob_name in potential_problems_name:

        if count >= num:
            break

        prob = pycutest.import_problem(problemName=prob_name, drop_fixed_variables=True)
        num_eq = np.count_nonzero(prob.is_eq_cons)
        num_c = prob.m
        # we don't consider the problem without eq or ineq constraints
        if num_eq == num_c or num_eq == 0:
            continue

        problems_name.append(prob_name)
        count += 1

    return problems_name