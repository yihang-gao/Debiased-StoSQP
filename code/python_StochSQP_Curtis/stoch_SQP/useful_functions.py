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


def evaluate_est_grad_in(prob, prop_prob, x, hyper, noise=True):
    # estimate the gradient. If noise=True, then return the estimated gradient
    _, g = prob.obj(x, gradient=True)
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
        alpha_k = 1e-1
    else:

        alpha_k = min(1.0 / (1 + variables.iter - hyper.buffer_size) ** hyper.decay_var, 1e-1)
        # alpha_k = 1.0
    return alpha_k


def feasibility_subprob(prob, prop_prob, variables, hyper):
    try:
        cont_vio = evaluate_constraint_violation(prob, prop_prob, variables)
        hyper.mu = max(1e-8, np.sum(cont_vio ** 2) * 1e-4)
        P = hyper.mu * np.identity(n=prop_prob.mc + prop_prob.new_dim)
        Jac = evaluate_jacobian(prob, prop_prob, variables)
        Jac2 = np.matmul(Jac, np.transpose(Jac))
        Jac4 = np.matmul(Jac2, Jac2)
        P[0:prop_prob.mc, 0:prop_prob.mc] = Jac4
        variables.cont = cont_vio
        q = np.matmul(cont_vio, Jac2)
        q = np.append(q, np.zeros(shape=(prop_prob.new_dim,)))

        lb = variables.xyz - prop_prob.xyzl
        ub = prop_prob.xyzu - variables.xyz
        h = np.append(ub, lb)
        G = np.append(np.transpose(Jac), np.identity(n=prop_prob.new_dim), axis=1)
        G = np.append(G, -G, axis=0)

        A = np.zeros(shape=(prop_prob.mc, prop_prob.mc + prop_prob.new_dim))
        A[:, prop_prob.mc:] = Jac
        b = np.zeros(shape=(prop_prob.mc,))
        w = solve_qp(P=P, q=q, G=G, h=h, A=A, b=b, lb=None, ub=None, solver="proxqp")
        if w is None:
            print("An error occured, cannot minimize quadratic feasibility error")
            return True, None
        v = np.matmul(np.transpose(Jac), w[0:prop_prob.mc]) + w[prop_prob.mc:]
        v = np.matmul(Jac, v)
        return False, v
    except Exception as e:
        print("An error occurred when minimizing quadratic feasibility error.")
        print(e)
        return True, None


def get_update_grad_hess(prob, prop_prob, variables, hyper):
    try:
        est_grad = evaluate_est_grad(prob, prop_prob, variables, hyper, noise=True)
        variables.est_grad = est_grad
        if hyper.adaptive_hess:
            est_hess = evaluate_est_hess(prob, prop_prob, variables, hyper, noise=True)
            variables.est_hess = est_hess

        return False
    except:
        print("An error occurred when updating grad and hess.")
        return True


def make_hess_pd(variables, hyper):
    if not hyper.adaptive_hess:
        # if not adaptive hessian, the default hess is identity
        return False, variables.est_hess
    try:
        H = (variables.est_hess + np.transpose(variables.est_hess)) / 2
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


def solve_sqp_subprob(B, v, prob, prop_prob, variables, hyper):
    try:
        P = np.zeros(shape=(prop_prob.new_dim, prop_prob.new_dim))

        P[0:prop_prob.dim_n, 0:prop_prob.dim_n] = B

        q = np.zeros(shape=(prop_prob.new_dim,))
        q[0:prop_prob.dim_n] = variables.est_grad
        A = evaluate_jacobian(prob, prop_prob, variables)

        b = v
        lb = prop_prob.xyzl - variables.xyz
        ub = prop_prob.xyzu - variables.xyz
        # calculate the step by solving the sqp subproblem
        sqp_subproblem = Problem(P=P, q=q, G=None, h=None, A=A, b=b, lb=lb, ub=ub)

        solution = solve_problem(sqp_subproblem, solver="proxqp")

        # if solution is None:
        #     print("An error occurred, cannot solve SQP subproblem.")
        #     return True, None, None, None

        d_xyz = solution.x
        d_dual_eq = solution.y
        d_dual_bound = solution.z_box

    except:
        print("An error occurred when solving SQP subproblem.")
        return True, None, None, None

    return False, d_xyz, d_dual_eq, d_dual_bound


# update the lipschitz constant by finite difference
def update_lipschitz_constant(prob, prop_prob, variables, hyper):
    if hyper.adaptive_lip:
        random_vec_x = np.random.normal(loc=0.0, scale=1e-4, size=(prop_prob.dim_n,))

        new_x = variables.x + random_vec_x

        avg_grad_new = np.zeros(shape=(prop_prob.dim_n,))
        avg_grad = np.zeros(shape=(prop_prob.dim_n,))
        for i in range(hyper.num_est_grad):
            avg_grad_new = (1 - 1 / (i + 1)) * avg_grad_new + 1 / (i + 1) * evaluate_est_grad_in(prob, prop_prob, new_x,
                                                                                                 hyper,
                                                                                                 noise=True)

        for i in range(hyper.num_est_grad):
            avg_grad = (1 - 1 / (i + 1)) * avg_grad + 1 / (i + 1) * evaluate_est_grad(prob, prop_prob, variables, hyper,
                                                                                      noise=True)

        lip_gradf = np.sqrt(np.sum((avg_grad - avg_grad_new) ** 2)) / (np.sqrt(np.sum(random_vec_x ** 2)) + 1e-8)

        [_, Jac] = prob.lagjac(variables.x)
        [_, new_Jac] = prob.lagjac(new_x)
        lip_gradc = np.sqrt(np.sum((new_Jac - Jac) ** 2)) / (np.sqrt(np.sum(random_vec_x ** 2)) + 1e-8)

        hyper.lip_gradf = hyper.ratio_lip_gradf * lip_gradf + (1 - hyper.ratio_lip_gradf) * hyper.lip_gradf
        hyper.lip_gradc = hyper.ratio_lip_gradc * lip_gradc + (1 - hyper.ratio_lip_gradc) * hyper.lip_gradc


# stochastic step size
def get_step_size(d_xyz, B, prob, prop_prob, variables, hyper):
    try:
        if np.sum(d_xyz ** 2) < 1e-16:
            return False, 1.0
        if hyper.adaptive:
            pk = d_xyz[0:prop_prob.dim_n]
            g_pk = np.sum(variables.est_grad * pk)
            pk_B_pk = np.sum(pk * np.matmul(B, pk))
            Jac = evaluate_jacobian(prob, prop_prob, variables)
            gradc_d = np.matmul(Jac, d_xyz)
            new_approx_const_vio = variables.cont + gradc_d
            linear_imprv_const = np.sqrt(np.sum(variables.cont ** 2)) - np.sqrt(np.sum(new_approx_const_vio ** 2))
            linear_imprv_const = max(1e-8, linear_imprv_const)
            if g_pk + pk_B_pk / 2 > 0.0:
                tau_trial = (1 - hyper.sig) * linear_imprv_const / (g_pk + pk_B_pk / 2 + 1e-6)
                if variables.tau > tau_trial:
                    variables.tau = min(tau_trial, (1 - hyper.ratio_xi) * variables.xi)

            delta_q = - variables.tau * g_pk + linear_imprv_const
            xi_trial = delta_q / (np.sum(d_xyz ** 2) + 1e-6) / variables.tau

            if xi_trial < variables.xi:
                variables.xi = min(xi_trial, (1 - hyper.ratio_xi) * variables.xi)

            gamma_k = step_size_var(variables, hyper)
            update_lipschitz_constant(prob, prop_prob, variables, hyper)
            alpha_k_min = 2 * (1 - hyper.eta) * variables.tau * variables.xi * gamma_k / (
                    hyper.lip_gradf * variables.tau + hyper.lip_gradc)
            aux1 = (hyper.eta - 1) * gamma_k * delta_q
            aux2 = gradc_d
            aux3 = np.sqrt(np.sum(variables.cont ** 2))
            aux4 = linear_imprv_const
            aux5 = 0.5 * (hyper.lip_gradf * variables.tau + hyper.lip_gradc) * np.sum(d_xyz ** 2)
            alpha_k_phi = alpha_k_min
            phi_k_alpha = aux1 * alpha_k_phi + np.sqrt(np.sum(
                (variables.cont + alpha_k_phi * aux2) ** 2)) - aux3 + alpha_k_phi * aux4 + alpha_k_phi ** 2 * aux5
            while alpha_k_phi < 0.9 and phi_k_alpha <= 0:
                alpha_k_phi = alpha_k_phi * 1.1
                phi_k_alpha = aux1 * alpha_k_phi + np.sqrt(np.sum(
                    (variables.cont + alpha_k_phi * aux2) ** 2)) - aux3 + alpha_k_phi * aux4 + alpha_k_phi ** 2 * aux5

            alpha_k_max = min(alpha_k_phi, alpha_k_min + hyper.theta * gamma_k)


            return False, alpha_k_max
        else:
            gamma_k = step_size_var(variables, hyper)
            return False, gamma_k
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

    variables.dual_eq = d_dual_eq

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

    variables.dual_bound = d_dual_bound_
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
