import pycutest
import numpy as np
from utilts import load_problem, check, setup_parameters
from stoch_activeset_SQP.stoch_activeset_SQP import *
from stoch_activeset_SQP.useful_functions import *
import sys, os

print("begin.")
num_probs = 1
max_n, max_m = 200, 200
count = 0
np.random.seed(66)

# problems_name = load_problem.load_problems(n=max_n, m=max_m, num=num_probs)

# print("finish finding the problems.")
require = check.CheckRequirement(max_n, max_m)

problems_name = ["GENHS28", "MISTAKE", "MAKELA1", "HS12", "HS22"]


noi_std = 1e0

for prob_name in problems_name:

    prob = pycutest.import_problem(prob_name, drop_fixed_variables=True)
    print("finish loading the problem {}.".format(prob_name))
    print("dimensions of variables are {:d}, numbers of constraints are {:d}.".format(prob.n, prob.m))
    correct = check.check(prob, require)


    if correct:
        prop_prob = check.PropertyProblem(prob, seed=666)
        print("The problem {} satisfies all requirements.".format(prob_name))
    else:
        print("The problem {} does not satisfy requirements.".format(prob_name))
        # sys.exit(0)
        continue

    hyper = setup_parameters.HyperParameters(max_n, max_m, noise_type="gaussian", noi_std_grad_hess=[noi_std, noi_std],
                                             Newton=False, repeat=10,
                                             max_iter=int(5e3))
    err = True

    for i in range(hyper.repeat):
        variables = setup_parameters.Variables(prob, prop_prob)
        temp_variables = setup_parameters.Temp_Variables(prop_prob)

        err = solve_stoch_activeset_SQP(prob, prop_prob, variables, temp_variables, hyper)
        if err:
            break

        err, kkt, cont = cal_kkt_res_cont(prob, prop_prob, variables, temp_variables)
        if err:
            break
        print("{:d}-th time, KKT Residual is {:.3e}, feasibility is {:.3e}.".format(i + 1, kkt, cont))
#        file_path = "./results/gaussian/std{}/".format(str(int(np.log10(noi_std))))
#        if not os.path.exists(file_path):
#            os.makedirs(file_path)
#        file_path = file_path + "{}-{}th.npy".format(prob_name, i+1)
#        np.save(file_path, variables)
