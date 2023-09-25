import pycutest
from utilts import load_problem, check, setup_parameters
from stoch_SQP.stoch_SQP import solve_relax_stoch_SQP, cal_kkt_res_cont
from stoch_SQP.useful_functions import *
import sys, os

print("begin.")
num_probs = 1
max_n, max_m = 200, 200
count = 0
np.random.seed(666)

# problems_name = load_problem.load_problems(n=max_n, m=max_m, num=num_probs)

# print("finish finding the problems.")
require = check.CheckRequirement(max_n, max_m)

problems_name = ["HS32", "HS41", "HS65", "HS68", "HS71", "HS81", "HS107", "BT13", "HS113", "GENHS28", "GOULDQP1",
                 "HAIFAS", "FCCU", "DISC2", "GOFFIN", "HIMMELBK", "POLAK3", "ZECEVIC3", "ZY2", "MISTAKE"]

noi_std = 1e0

for prob_name in problems_name:
    # prob_name = "ORTHREGB"
    #
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
                                             noi_stu_t_freed=[4, 4], decay_var=0.51,
                                             adaptive_lip=True,
                                             adaptive_hess=False,
                                             repeat=5,
                                             max_iter=int(1e5))
    variables = setup_parameters.Variables(prob, prop_prob)
    err = True

    for i in range(hyper.repeat):
        hyper.adaptive = True
        variables = setup_parameters.Variables(prob, prop_prob)

        err = solve_relax_stoch_SQP(prob, prop_prob, variables, hyper)
        if err:
            break

        err, kkt, cont = cal_kkt_res_cont(prob, prop_prob, variables, hyper)
        if err:
            break
        print("{:d}-th time, KKT Residual is {:.3e}, feasibility is {:.3e}.".format(i + 1, kkt, cont))
        file_path = "./results/gaussian/std{}/".format(str(int(np.log10(noi_std))))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = file_path + "{}-{}th.npy".format(prob_name, i+1)
        np.save(file_path, variables)
