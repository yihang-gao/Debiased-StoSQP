import numpy as np
from qpsolvers import solve_qp, Problem, solve_problem
from scipy import sparse
import scipy.sparse.linalg as scipyla


def evaluate_jacobian_x(prob, prop_prob, variables):
    [_, J] = prob.lagjac(variables.x)
    Jac = np.zeros(shape=(prop_prob.mc, prop_prob.dim_n))
    Jac[0:prop_prob.mce, :] = J[prop_prob.ice, :]
    Jac[prop_prob.mce:prop_prob.mce + prop_prob.mcl, :] = J[prop_prob.icl, :]
    Jac[prop_prob.mce + prop_prob.mcl:, :] = J[prop_prob.icu, :]
    return Jac


def evaluate_jacobian(prob, prop_prob, variables):
    [_, J] = prob.lagjac(variables.x)
    Jac = np.zeros(shape=(prop_prob.mc, prop_prob.new_dim))
    if prop_prob.mce > 0:
        Jac[0:prop_prob.mce, 0:prop_prob.dim_n] = J[prop_prob.ice, :]
    if prop_prob.mcl > 0:
        Jac[prop_prob.mce:(prop_prob.mce + prop_prob.mcl), 0:prop_prob.dim_n] = J[prop_prob.icl, :]
    if prop_prob.mcu > 0:
        Jac[(prop_prob.mce + prop_prob.mcl):, 0:prop_prob.dim_n] = J[prop_prob.icu, :]
    # the assignment here can be improved
    if prop_prob.mcl + prop_prob.mcu > 0:
        Jac[prop_prob.mce:, prop_prob.dim_n:] = np.identity(n=prop_prob.mcl + prop_prob.mcu)
    return Jac


# the function evaluate jacobian of constraints for finding relaxing parameters
def evaluate_jacobian_relax_param(prob, prop_prob, variables):
    Jac = evaluate_jacobian(prob, prop_prob, variables)
    return np.concatenate((Jac, np.identity(n=prop_prob.mc)), axis=1)


def evaluate_constraint_violation(prob, prop_prob, variables):
    const = prob.cons(variables.x)
    const_eq = const[prop_prob.ice] - prop_prob.cl[prop_prob.ice]
    const_ieql = const[prop_prob.icl] + variables.y - prop_prob.cl[prop_prob.icl]
    const_iequ = const[prop_prob.icu] + variables.z - prop_prob.cu[prop_prob.icu]
    return np.append(const_eq, np.append(const_ieql, const_iequ))


def evaluate_constraint_violation_eq_ieq(prob, prop_prob, variables):
    const = prob.cons(variables.x)
    const_eq = const[prop_prob.ice] - prop_prob.cl[prop_prob.ice]
    const_ieql = np.minimum(const[prop_prob.icl] - prop_prob.cl[prop_prob.icl], 0.0)
    const_iequ = np.maximum(const[prop_prob.icu] - prop_prob.cu[prop_prob.icu], 0.0)
    return np.append(const_eq, np.append(const_ieql, const_iequ))


def evaluate_est_grad(prob, prop_prob, variables, hyper, noise=True):
    # estimate the gradient. If noise=True, then return the estimated gradient
    _, g = prob.obj(variables.x, gradient=True)
    if noise:
        if hyper.noise_type == "gaussian":
            omega = np.identity(n=prop_prob.dim_n) + 1.0
            g = g + np.sqrt(hyper.noi_std_grad_hess[0]) * \
                np.random.multivariate_normal(mean=np.zeros(shape=np.shape(g)), cov=omega, size=1)[0]
            return g
        elif hyper.noise_type == "t_distribution":
            return g + np.random.standard_t(df=hyper.noi_stu_t_freed[0], size=np.shape(g))
        else:
            return g
    else:
        return g


def evaluate_est_hess(prob, prop_prob, variables, hyper, noise=True):
    # estimate the hessian of lagrangian.
    # If noise=True, then return the estimated Hessian.
    dual_e_orig = np.zeros(prop_prob.dim_m)
    dual_l_orig = np.zeros(prop_prob.dim_m)
    dual_u_orig = np.zeros(prop_prob.dim_m)
    dual_e_orig[prop_prob.ice] = variables.dual_eq[0:prop_prob.mce]
    dual_l_orig[prop_prob.icl] = variables.dual_eq[prop_prob.mce:prop_prob.mce + prop_prob.mcl]
    dual_u_orig[prop_prob.icu] = variables.dual_eq[prop_prob.mce + prop_prob.mcl:]
    H = prob.hess(variables.x, dual_e_orig + dual_l_orig + dual_u_orig)

    if noise:
        if hyper.noise_type == "gaussian":
            H = H + np.random.normal(loc=0.0, scale=hyper.noi_std_grad_hess[1], size=np.shape(H))
            return (H + np.transpose(H)) / 2
        elif hyper.noise_type == "t_distribution":
            H = H + np.random.standard_t(df=hyper.noi_stu_t_freed[1], size=np.shape(H))
            return (H + np.transpose(H)) / 2
        else:
            return H
    else:
        return H


def step_size_grad(variables, hyper):
    beta_k = 1. / (1 + variables.iter - hyper.buffer_size) ** hyper.decay_grad
    return beta_k


def step_size_var(variables, hyper):
    if variables.iter < hyper.buffer_size:
        # alpha_k = 5e-1
        alpha_k = 1e-1
    else:
        # alpha_k = min(10.0 / (1 + variables.iter - hyper.buffer_size) ** hyper.decay_var, 1.0)
        alpha_k = min(1.0 / (1 + variables.iter - hyper.buffer_size) ** hyper.decay_var, 1e-1)
        # alpha_k = 1.0
    return alpha_k




def find_relaxing_param(prob, prop_prob, variables, hyper):
    variables.relax = 1.0

    # here we use QP to select the relaxing parameter
    # we use the same notation as https://github.com/qpsolvers/qpsolvers
    try:
        Jac = evaluate_jacobian(prob, prop_prob, variables)
        P = np.matmul(np.transpose(Jac), Jac)

        cont_vio = evaluate_constraint_violation(prob, prop_prob, variables)
        variables.cont = cont_vio
        q = np.matmul(cont_vio, Jac)

        lb = prop_prob.xyzl - variables.xyz
        ub = prop_prob.xyzu - variables.xyz

        while variables.relax > hyper.tol_relax_param:
            q_prime = variables.relax * q
            w = solve_qp(P=P, q=q_prime, G=None, h=None, A=None, b=None, lb=lb, ub=ub, solver="proxqp")
            if w is None:
                print("An error occured, cannot find suitable relaxing parameters, {:.3e}.".format(variables.relax))
                return True
            loss = variables.relax * cont_vio + np.matmul(Jac, w)
            loss = np.sqrt(np.sum(loss ** 2))
            if loss < hyper.tol_relax_loss:
                return False
            else:
                variables.relax = variables.relax * hyper.decay_relax

        print("exceed tolerance, cannot find suitable relaxing parameters, {:.3e}.".format(variables.relax))

        return True
    except Exception as e:
        print("An error occurred when finding the proper relaxing parameters.")
        print(e)
        return True


def get_update_grad_hess(prob, prop_prob, variables, hyper):
    try:
        est_grad = evaluate_est_grad(prob, prop_prob, variables, hyper, noise=True)
        est_hess = evaluate_est_hess(prob, prop_prob, variables, hyper, noise=True)
        if variables.iter < hyper.buffer_size:
            step_s_hess = 1.0 / (variables.iter + 1)
            variables.avg_grad = est_grad
            variables.avg_hess = step_s_hess * est_hess + (1 - step_s_hess) * variables.avg_hess
        else:
            beta_k = step_size_grad(variables, hyper)
            step_s_hess = 1.0 / (variables.iter - hyper.buffer_size + 1)
            variables.avg_grad = beta_k * est_grad + (1 - beta_k) * variables.avg_grad
            variables.avg_hess = step_s_hess * est_hess + (1 - step_s_hess) * variables.avg_hess

        return False
    except:
        print("An error occurred when updating grad and hess.")
        return True


def make_hess_pd(variables, hyper):
    try:
        H = (variables.avg_hess + np.transpose(variables.avg_hess)) / 2
        D, U = np.linalg.eigh(a=H)
        D = np.maximum(np.minimum(D, hyper.max_eig), hyper.min_eig)
        # D = np.maximum(D, hyper.min_eig)
        # min_eig_val = scipyla.eigsh(H, k=1, which="SA", tol=1e-2, return_eigenvectors=False)
        D = np.tile(np.reshape(D, (len(D), 1)), len(D))
        B = np.matmul(U, np.multiply(D, np.transpose(U)))
        return False, (B + np.transpose(B)) / 2
        # return False, H + max(-min_eig_val[0], hyper.min_eig) * np.identity(n=np.shape(H)[0])
    except:
        print("An error occurred when making hess pd.")
        return True, None


def solve_relax_sqp_subprob(B, prob, prop_prob, variables, hyper):
    try:
        P = np.zeros(shape=(prop_prob.new_dim, prop_prob.new_dim))

        P[0:prop_prob.dim_n, 0:prop_prob.dim_n] = B

        q = np.zeros(shape=(prop_prob.new_dim,))
        q[0:prop_prob.dim_n] = variables.avg_grad
        A = evaluate_jacobian(prob, prop_prob, variables)

        b = - variables.relax * variables.cont
        lb = prop_prob.xyzl - variables.xyz
        ub = prop_prob.xyzu - variables.xyz
        # calculate the step by solving the sqp subproblem
        sqp_subproblem = Problem(P=P, q=q, G=None, h=None, A=A, b=b, lb=lb, ub=ub)

        solution = solve_problem(sqp_subproblem, solver="proxqp")


        d_xyz = solution.x
        d_dual_eq = solution.y
        d_dual_bound = solution.z_box

    except:
        print("An error occurred when solving SQP subproblem.")
        return True, None, None, None

    return False, d_xyz, d_dual_eq, d_dual_bound


# stochastic step size
def get_step_size(pk, B, variables, hyper):
    try:
        if hyper.adaptive:
            g_pk = np.sum(variables.avg_grad * pk)
            pk_B_pk = np.sum(pk * np.matmul(B, pk))
            if g_pk + pk_B_pk <= 0.0:
                pen_trial = 0.0
            else:
                pen_trial = (g_pk + pk_B_pk) / (
                        (1 - hyper.sig) * variables.relax * np.sqrt(np.sum(variables.cont ** 2)) + 1e-4)
            if pen_trial > variables.pen:
                variables.pen = (1 + hyper.ratio_pen) * pen_trial

            delta_q = - g_pk - pk_B_pk / 2 + variables.pen * variables.relax * np.sqrt(np.sum(variables.cont ** 2))
            xi_trial = delta_q / (np.sum(pk ** 2) + 1e-4)

            if xi_trial < variables.xi:
                variables.xi = min(xi_trial, (1 - hyper.ratio_xi) * variables.xi)

            gamma_k = step_size_var(variables, hyper)
            alpha_k_min = variables.xi * gamma_k / (hyper.lip_gradf + variables.pen * hyper.lip_gradc)
            alpha_k_max = alpha_k_min + hyper.varrho * gamma_k ** 2
            alph_k_trial = xi_trial * gamma_k / (hyper.lip_gradf + variables.pen * hyper.lip_gradc)

            if alph_k_trial <= alpha_k_max:
                return False, alph_k_trial
            else:
                return False, alpha_k_max
        else:
            alpha_k = step_size_var(variables, hyper)
            return False, alpha_k
    except:
        print("An error occurred when calculating step size.")
        return True, None


def l2_regularized_merit(pk, alpha_k, prob, prop_prob, variables):
    if pk is None:
        const_vio = evaluate_constraint_violation(prob, prop_prob, variables)
        const_vio = np.sqrt(np.sum(const_vio ** 2))
        f = prob.obj(variables.x)
        return f + const_vio * variables.pen
    else:
        x = variables.x + alpha_k * pk
        const = prob.cons(x)
        const_eq = const[prop_prob.ice] - prop_prob.cl[prop_prob.ice]
        const_ieql = const[prop_prob.icl] + variables.y - prop_prob.cl[prop_prob.icl]
        const_iequ = const[prop_prob.icu] + variables.z - prop_prob.cu[prop_prob.icu]
        const_vio = np.append(const_eq, np.append(const_ieql, const_iequ))
        const_vio = np.sqrt(np.sum(const_vio ** 2))
        f = prob.obj(x)
        return f + const_vio * variables.pen


def update_vars(d_xyz, d_dual_eq, d_dual_bound, alpha_k, prop_prob, variables, hyper):
    variables.xyz = variables.xyz + alpha_k * d_xyz
    variables.xyz = np.maximum(np.minimum(variables.xyz, prop_prob.xyzu), prop_prob.xyzl)

    if variables.iter < hyper.buffer_size:
        variables.dual_eq = d_dual_eq
    else:
        variables.dual_eq = (1 - alpha_k) * variables.dual_eq + alpha_k * d_dual_eq

    d_dual_bound_l = -np.minimum(d_dual_bound, 0.0)
    d_dual_bound_u = np.maximum(d_dual_bound, 0.0)
    d_dual_bound_ = np.zeros(shape=np.shape(variables.dual_bound))
    if prop_prob.nxl > 0:
        d_dual_bound_[0: prop_prob.nxl] = d_dual_bound_l[prop_prob.ixl]
    if prop_prob.nxu > 0:
        d_dual_bound_[prop_prob.nxl: (prop_prob.nxl + prop_prob.nxu)] = d_dual_bound_u[prop_prob.ixu]
    if prop_prob.mcl > 0:
        d_dual_bound_[(prop_prob.nxl + prop_prob.nxu):(prop_prob.nxl + prop_prob.nxu + prop_prob.mcl)] = d_dual_bound_u[
                                                                                                         prop_prob.dim_n: (
                                                                                                                 prop_prob.dim_n + prop_prob.mcl)]
    if prop_prob.mcu > 0:
        d_dual_bound_[-prop_prob.mcu:] = d_dual_bound_l[-prop_prob.mcu:]

    if variables.iter < hyper.buffer_size:
        variables.dual_bound = d_dual_bound_
    else:
        variables.dual_bound = (1 - alpha_k) * variables.dual_bound + alpha_k * d_dual_bound_
    # variables.dual_bound = d_dual_bound_
    variables.x = variables.xyz[0:prop_prob.dim_n]
    variables.y = variables.xyz[prop_prob.dim_n: prop_prob.dim_n + prop_prob.mcl]
    variables.z = variables.xyz[prop_prob.dim_n + prop_prob.mcl:]

    if np.isnan(variables.xyz).any() or np.isnan(variables.dual_eq).any() or np.isnan(variables.dual_bound).any():
        print("Nan values exist in variables.")
        return True
    else:
        return False


def cal_kkt_res_cont(prob, prop_prob, variables, hyper):
    g = evaluate_est_grad(prob, prop_prob, variables, hyper, noise=False)
    Jac = evaluate_jacobian_x(prob, prop_prob, variables)
    dual_bound_l = np.zeros(shape=(prop_prob.dim_n,))
    dual_bound_u = np.zeros(shape=(prop_prob.dim_n,))
    dual_bound_l[prop_prob.ixl] = variables.dual_bound[0:prop_prob.nxl]
    dual_bound_u[prop_prob.ixu] = variables.dual_bound[prop_prob.nxl:(prop_prob.nxl + prop_prob.nxu)]
    kkt = g + np.matmul(np.transpose(Jac), variables.dual_eq) - dual_bound_l + dual_bound_u
    cont = evaluate_constraint_violation_eq_ieq(prob, prop_prob, variables)

    if np.isnan(kkt).any() or np.isnan(cont).any():
        print("Nan values exist in KKT / feasibility evaluation.")
        return True, None, None
    else:
        return False, np.sqrt(np.sum(kkt ** 2)), np.sqrt(np.sum(cont ** 2))
