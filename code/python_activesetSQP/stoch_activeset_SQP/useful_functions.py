import numpy as np
import scipy


# update temporary variables
def update_temp_variables(prob, prop_prob, variables, temp_variables):
    try:
        temp_variables.gradf, temp_variables.J = prob.lagjac(variables.x)

        dual_general1 = np.zeros(shape=(prop_prob.dim_m,))
        dual_general2 = np.zeros(shape=(prop_prob.dim_m,))
        if prop_prob.mcl > 0:
            dual_general1[prop_prob.icl] = variables.dual_general[prop_prob.mce:(prop_prob.mce + prop_prob.mcl)]
        if prop_prob.mcu > 0:
            dual_general2[prop_prob.icu] = variables.dual_general[(prop_prob.mce + prop_prob.mcl):]
        dual_general = dual_general2 - dual_general1
        dual_general[prop_prob.ice] = variables.dual_general[0:prop_prob.mce]

        temp_variables.hess_lag = prob.hess(variables.x, dual_general)
        return False
    except:
        print("An error occurred when updating temporary variables.")
        return True


# evaluate the constraints in the order of equality, l-inequality, u-inequality, l-box and u-box constraints.
def evaluate_constraints(prob, prop_prob, variables, temp_variables):
    const = prob.cons(variables.x)
    c_eq = const[prop_prob.ice] - prop_prob.cl[prop_prob.ice]
    c_cl = - const[prop_prob.icl] + prop_prob.cl[prop_prob.icl]
    c_cu = const[prop_prob.icu] - prop_prob.cu[prop_prob.icu]
    c_xl = prop_prob.xl[prop_prob.ixl] - variables.x[prop_prob.ixl]
    c_xu = variables.x[prop_prob.ixu] - prop_prob.xu[prop_prob.ixu]

    c_ieq = np.concatenate((c_cl, c_cu, c_xl, c_xu), axis=None)

    temp_variables.const = np.concatenate((c_eq, c_ieq), axis=None)

    # we also check v
    c_ieq = np.maximum(c_ieq, 1e-10)
    v_trial = 2 * np.sum(c_ieq ** 3)
    if temp_variables.v < v_trial:
        temp_variables.v = v_trial


# get the gradient of the lagrangian
def get_grad_lag(prop_prob, variables, temp_variables):
    # Jac, the jacobian of constraints in order of eq, l-ieq, u-ieq, l-box and u-box.
    Jac = np.zeros(shape=(prop_prob.mc, prop_prob.dim_n))
    if prop_prob.mce > 0:
        Jac[0:prop_prob.mce, :] = temp_variables.J[prop_prob.ice, :]
    if prop_prob.mcl > 0:
        Jac[prop_prob.mce:(prop_prob.mce + prop_prob.mcl), :] = - temp_variables.J[prop_prob.icl, :]
    if prop_prob.mcu > 0:
        Jac[(prop_prob.mce + prop_prob.mcl):, :] = temp_variables.J[prop_prob.icu, :]

    temp_variables.Jac = Jac

    dual_bound1 = np.zeros(shape=(prop_prob.dim_n,))
    dual_bound2 = np.zeros(shape=(prop_prob.dim_n,))
    dual_bound1[prop_prob.ixl] = variables.dual_bound[0:prop_prob.nxl]
    dual_bound2[prop_prob.ixu] = variables.dual_bound[prop_prob.nxl:]

    temp_variables.grad_lag = temp_variables.gradf + np.matmul(np.transpose(Jac),
                                                               variables.dual_general) - dual_bound1 + dual_bound2


# obtain the noise of gradient, given the step size
def get_noise_grad(prop_prob, temp_variables, hyper, xi_1t):
    if hyper.noise_type == "gaussian":
        omega = (np.identity(n=prop_prob.dim_n) + 1.0)
        temp_variables.noi_grad1 = np.sqrt(hyper.noi_std_grad_hess[0]) * \
                                   np.random.multivariate_normal(mean=np.zeros(shape=(prop_prob.dim_n,)), cov=omega,
                                                                 size=1)[0] / np.sqrt(xi_1t)
    elif hyper.noise_type == "t_distribution":
        temp_variables.noi_grad1 = np.random.standard_t(df=hyper.noi_stu_t_freed[0], size=(prop_prob.dim_n,)) / np.sqrt(
            xi_1t)


# obtain the noise of hessian, given the step size
def get_noise_hess(prop_prob, temp_variables, hyper, xi_1t):
    if hyper.noise_type == "gaussian":
        noi_hess1 = np.random.normal(loc=0.0, scale=hyper.noi_std_grad_hess[1],
                                     size=(prop_prob.dim_n, prop_prob.dim_n)) / np.sqrt(xi_1t)
        noi_hess1 = (noi_hess1 + np.transpose(noi_hess1)) / 2
        temp_variables.noi_hess1 = noi_hess1
    elif hyper.noise_type == "t_distribution":
        noi_hess1 = np.random.standard_t(df=hyper.noi_stu_t_freed[1],
                                         size=(prop_prob.dim_n, prop_prob.dim_n)) / np.sqrt(xi_1t)
        noi_hess1 = (noi_hess1 + np.transpose(noi_hess1)) / 2
        temp_variables.noi_hess1 = noi_hess1


# to satisfy (15) and (16)
# equivalently to satisfy (37)
def get_sample_size_fir(prob, prop_prob, variables, temp_variables, hyper):
    evaluate_constraints(prob, prop_prob, variables, temp_variables)
    xi_1t = 1
    num_loop = 0
    max_loop = 10
    get_grad_lag(prop_prob, variables, temp_variables)

    # here, since right-hand-side of (37) also involve the step size
    # as mentioned in the paper, we dynamically select the step size
    while num_loop < max_loop:
        num_loop += 1
        get_noise_grad(prop_prob, temp_variables, hyper, xi_1t)

        R_t = np.sum((temp_variables.noi_grad1 + temp_variables.grad_lag) ** 2)
        if prop_prob.mce > 0:
            R_t += np.sum(temp_variables.const[0:prop_prob.mce] ** 2)
        if prop_prob.mieq > 0:
            const_ = temp_variables.const[prop_prob.mce:]
            const_ = np.maximum(const_, - np.concatenate((variables.dual_general[prop_prob.mce:], variables.dual_bound),
                                                         axis=None))
            R_t += np.sum(const_ ** 2)
        R_t = np.sqrt(R_t)

        xi_1t_trial = hyper.c * (
                np.log(prop_prob.dim_n / hyper.p_grad) / np.maximum(
            np.minimum(hyper.k_grad ** 2 * variables.alpha ** 2 * R_t ** 2,
                       hyper.x_grad ** 2 * temp_variables.delta / variables.alpha), 1e-20))

        variables.xi_1t = xi_1t
        temp_variables.Rt = R_t

        if xi_1t_trial <= xi_1t:
            break
        else:
            xi_1t = xi_1t * 5

    # for simplicity, we let tau_t = xi_t
    get_noise_hess(prop_prob, temp_variables, hyper, variables.xi_1t)


# equation (9), calculate matrix Q1
def get_matrix_Q1(prob, prop_prob, variables, temp_variables):
    if prop_prob.mce > 0:
        Q11 = np.matmul(temp_variables.hess_lag + temp_variables.noi_hess1,
                        np.transpose(temp_variables.Jac[0:prop_prob.mce, :]))
        Q12 = np.zeros(shape=(prop_prob.dim_n, prop_prob.mce))
        count = 0
        grad_lag_noi = temp_variables.grad_lag + temp_variables.noi_grad1
        for i in prop_prob.ice[0]:
            hess_cons = prob.ihess(variables.x, cons_index=i)
            hess_cons = np.matmul(hess_cons, grad_lag_noi)
            Q12[:, count] = hess_cons
            count += 1

        temp_variables.Q1 = Q11 + Q12


# equation (9), calculate matrix Q2
def get_matrix_Q2(prob, prop_prob, variables, temp_variables):
    if prop_prob.mieq > 0:
        Q22 = np.zeros(shape=(prop_prob.dim_n, prop_prob.mieq))
        # Q23 = np.zeros(shape=(prop_prob.dim_n, prop_prob.mieq))
        temp_variables.G = temp_variables.Jac[prop_prob.mce:, :]
        I = np.identity(n=prop_prob.dim_n)
        if prop_prob.nxl > 0:
            temp_variables.G = np.concatenate((temp_variables.G, -I[prop_prob.ixl[0], :]), axis=0)
        if prop_prob.nxu > 0:
            temp_variables.G = np.concatenate((temp_variables.G, I[prop_prob.ixu[0], :]), axis=0)
        Q21 = np.matmul(temp_variables.hess_lag + temp_variables.noi_hess1, np.transpose(temp_variables.G))
        count = 0
        grad_lag_noi = temp_variables.grad_lag + temp_variables.noi_grad1
        for i in prop_prob.icl[0]:
            hess_cons = -prob.ihess(variables.x, cons_index=i)
            hess_cons = np.matmul(hess_cons, grad_lag_noi)
            Q22[:, count] = hess_cons
            count += 1

        for i in prop_prob.icu[0]:
            hess_cons = prob.ihess(variables.x, cons_index=i)
            hess_cons = np.matmul(hess_cons, grad_lag_noi)
            Q22[:, count] = hess_cons
            count += 1

        Q23 = 2 * np.transpose(temp_variables.G) * (
                temp_variables.const[prop_prob.mce:] * np.concatenate((variables.dual_general[prop_prob.mce:],
                                                                       variables.dual_bound), axis=None))

        temp_variables.Q2 = Q21 + Q22 + Q23


# equation (9), calculate matrix M
def get_matrix_M(prop_prob, temp_variables):
    m = prop_prob.mce + prop_prob.mieq
    M = np.zeros(shape=(m, m))
    if prop_prob.mce > 0:
        M[0:prop_prob.mce, 0:prop_prob.mce] = np.matmul(temp_variables.Jac[0:prop_prob.mce, :],
                                                        np.transpose(temp_variables.Jac[0:prop_prob.mce, :]))
    if prop_prob.mieq > 0:
        M[prop_prob.mce:, prop_prob.mce:] = np.matmul(temp_variables.G, np.transpose(temp_variables.G)) + np.diag(
            temp_variables.const[prop_prob.mce:] ** 2)
    if prop_prob.mce > 0 and prop_prob.mieq > 0:
        JG = np.matmul(temp_variables.Jac[0:prop_prob.mce, :], np.transpose(temp_variables.G))
        M[0:prop_prob.mce, prop_prob.mce:] = JG
        M[prop_prob.mce:, 0:prop_prob.mce] = np.transpose(JG)
    temp_variables.M = M


# step1: estimate derivatives
def estimate_derivatives(prob, prop_prob, variables, temp_variables, hyper):
    try:
        get_sample_size_fir(prob, prop_prob, variables, temp_variables, hyper)
        get_matrix_Q1(prob, prop_prob, variables, temp_variables)
        get_matrix_Q2(prob, prop_prob, variables, temp_variables)
        get_matrix_M(prop_prob, temp_variables)

        return False
    except:
        print("An error occurred when estimating derivatives.")
        return True


# equation (10), get the gradient of the augmented lagrangian
def get_grad_aug_lag(prob, prop_prob, variables, temp_variables, hyper):
    grad = np.zeros(shape=(prop_prob.dim_n + prop_prob.mce + prop_prob.mieq,))
    temp_variables.av = temp_variables.v - np.sum(np.maximum(temp_variables.const[prop_prob.mce:], 1e-10) ** 3)
    temp_variables.qv = temp_variables.av / (
            1 + np.sum(variables.dual_general[prop_prob.mce:, ] ** 2) + np.sum(variables.dual_bound ** 2))
    temp_variables.w = np.maximum(temp_variables.const[prop_prob.mce:],
                                  - temp_variables.eps * temp_variables.qv * np.concatenate((
                                      variables.dual_general[prop_prob.mce:, ], variables.dual_bound), axis=None))

    grad_lag_noi = temp_variables.grad_lag + temp_variables.noi_grad1
    grad[0:prop_prob.dim_n] = grad_lag_noi
    QM = np.zeros(shape=(prop_prob.dim_n + prop_prob.mce + prop_prob.mieq, prop_prob.mce + prop_prob.mieq))
    vec_grad_lag = np.zeros(shape=(prop_prob.mce + prop_prob.mieq,))
    if prop_prob.mce > 0:
        grad[0:prop_prob.dim_n] += np.matmul(
            np.transpose(temp_variables.Jac[0:prop_prob.mce, :]),
            temp_variables.const[0:prop_prob.mce]) / temp_variables.eps
        grad[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] = temp_variables.const[0:prop_prob.mce]
        vec_grad_lag[0:prop_prob.mce] += np.matmul(temp_variables.Jac[0:prop_prob.mce, :], grad_lag_noi)
        QM[0:prop_prob.dim_n, 0:prop_prob.mce] = temp_variables.Q1
    if prop_prob.mieq > 0:
        ell = np.maximum(temp_variables.const[prop_prob.mce:], 0.0) ** 2
        grad[0:prop_prob.dim_n] += np.matmul(np.transpose(temp_variables.G), temp_variables.w) / (
                temp_variables.eps * temp_variables.qv + 1e-20) + 3 / 2 * np.sum(temp_variables.w ** 2) / (
                                           temp_variables.eps * temp_variables.qv * temp_variables.av + 1e-20) * np.matmul(
            np.transpose(temp_variables.G), ell)
        grad[(prop_prob.dim_n + prop_prob.mce):] = temp_variables.w + np.sum(temp_variables.w ** 2) / (
                temp_variables.eps * temp_variables.av + 1e-20) * np.concatenate(
            (variables.dual_general[prop_prob.mce:],
             variables.dual_bound), axis=None)
        vec_grad_lag[prop_prob.mce:] = np.matmul(temp_variables.G, grad_lag_noi) + (
                temp_variables.const[prop_prob.mce:] ** 2) * (
                                           np.concatenate(
                                               (variables.dual_general[prop_prob.mce:], variables.dual_bound),
                                               axis=None))
        QM[0:prop_prob.dim_n, prop_prob.mce:] = temp_variables.Q2
    QM[prop_prob.dim_n:, :] = temp_variables.M
    grad += np.matmul(QM, vec_grad_lag) * hyper.eta
    temp_variables.grad_aug_lag = grad


# whether equation (17) holds
def check_feas_gradlag(prob, prop_prob, variables, temp_variables, hyper):
    # try:
    aux1 = hyper.x_err * np.sqrt(np.sum(temp_variables.grad_aug_lag ** 2))
    if aux1 <= temp_variables.Rt:
        if np.sqrt(np.sum(temp_variables.const[0:prop_prob.mce] ** 2) + np.sum(temp_variables.w ** 2)) <= aux1:
            return True
        else:
            return False
    return True
    # except:
    #     print(temp_variables.w)


# solve equations (12)
def solve_subp(prob, prop_prob, variables, temp_variables, hyper):
    try:
        idx = np.where(
            temp_variables.const[prop_prob.mce:] >= -temp_variables.eps * temp_variables.qv * np.concatenate((
                variables.dual_general[prop_prob.mce:], variables.dual_bound), axis=None))[0]
        m = prop_prob.dim_n + prop_prob.mce + len(idx)
        K_a = np.zeros(shape=(m, m))

        # here, we may select other positive definite matrices
        # K_a[0:prop_prob.dim_n, 0:prop_prob.dim_n] = np.identity(n=prop_prob.dim_n)
        B = temp_variables.hess_lag + temp_variables.noi_hess1
        D, U = np.linalg.eigh(a=B)
        D = np.maximum(np.minimum(D, hyper.max_eig), hyper.min_eig)
        D = np.tile(np.reshape(D, (len(D), 1)), len(D))
        B = np.matmul(U, np.multiply(D, np.transpose(U)))

        K_a[0:prop_prob.dim_n, 0:prop_prob.dim_n] = B
        vec_rhs = np.zeros(shape=(m,))
        grad_lag_noi = temp_variables.grad_lag + temp_variables.noi_grad1
        if prop_prob.mce > 0:
            K_a[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce), 0:prop_prob.dim_n] = temp_variables.Jac[
                                                                                        0:prop_prob.mce, :]
            K_a[0:prop_prob.dim_n, prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] = np.transpose(
                temp_variables.Jac[0:prop_prob.mce, :])
            vec_rhs[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] = temp_variables.const[0:prop_prob.mce]
        if len(idx) > 0:
            G_a = temp_variables.G[idx, :]
            K_a[(prop_prob.dim_n + prop_prob.mce):, 0:prop_prob.dim_n] = G_a
            K_a[0:prop_prob.dim_n, (prop_prob.dim_n + prop_prob.mce):] = np.transpose(G_a)
            g_a = temp_variables.const[prop_prob.mce:]
            g_a = g_a[idx]
            vec_rhs[(prop_prob.dim_n + prop_prob.mce):] = g_a
            dual_ieq = np.concatenate((variables.dual_general[prop_prob.mce:], variables.dual_bound), axis=None)
            vec_rhs[0:prop_prob.dim_n] += - np.matmul(np.transpose(G_a), dual_ieq[idx])

        vec_rhs[0:prop_prob.dim_n] += grad_lag_noi
        x = scipy.linalg.solve(K_a + 1e-10 * np.identity(n=len(K_a)), -vec_rhs, assume_a='sym')
        delta_x = x[0:prop_prob.dim_n]

        m = prop_prob.dim_n + prop_prob.mce + prop_prob.mieq
        new_vec = np.zeros(shape=(m,))
        new_vec[0:prop_prob.dim_n] = delta_x

        m = prop_prob.mce + prop_prob.mieq
        M = np.zeros(shape=(m, m))
        vec_rhs = np.zeros(shape=(m,))
        if prop_prob.mieq > 0:
            vec = temp_variables.const[prop_prob.mce:] ** 2 * np.concatenate((variables.dual_general[prop_prob.mce:],
                                                                              variables.dual_bound), axis=None)

            if len(idx) > 0:
                vec[idx] = 0.0

            new_vec[(prop_prob.dim_n + prop_prob.mce):] = vec + np.matmul(temp_variables.G, grad_lag_noi)
            vec_rhs[prop_prob.mce:] += new_vec[(prop_prob.dim_n + prop_prob.mce):] + np.matmul(
                np.transpose(temp_variables.Q2), delta_x)
            GG = np.matmul(temp_variables.G, np.transpose(temp_variables.G))
            M[prop_prob.mce:, prop_prob.mce:] = GG + np.diag(temp_variables.const[prop_prob.mce:] ** 2)
        if prop_prob.mce > 0:
            new_vec[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] = np.matmul(
                temp_variables.Jac[0:prop_prob.mce, :], grad_lag_noi)
            vec_rhs[0:prop_prob.mce] += new_vec[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] + np.matmul(
                np.transpose(temp_variables.Q1), delta_x)
            JJ = np.matmul(temp_variables.Jac[0:prop_prob.mce, :],
                           np.transpose(temp_variables.Jac[0:prop_prob.mce, :]))
            M[0:prop_prob.mce, 0:prop_prob.mce] = JJ
        if prop_prob.mce > 0 and prop_prob.mieq > 0:
            JG = np.matmul(temp_variables.Jac[0:prop_prob.mce, :], np.transpose(temp_variables.G))
            M[0:prop_prob.mce, prop_prob.mce:] = JG
            M[prop_prob.mce:, 0:prop_prob.mce] = np.transpose(JG)
        delta_dual = scipy.linalg.solve(M, -vec_rhs, assume_a='sym')

        m = prop_prob.dim_n + prop_prob.mce + prop_prob.mieq
        grad_auglag = np.zeros(shape=(m,))
        grad_auglag[0:prop_prob.dim_n] = grad_lag_noi
        QM = np.zeros(shape=(prop_prob.dim_n + prop_prob.mce + prop_prob.mieq, prop_prob.mce + prop_prob.mieq))
        if prop_prob.mce > 0:
            grad_auglag[0:prop_prob.dim_n] += np.matmul(
                np.transpose(temp_variables.Jac[0:prop_prob.mce, :]),
                temp_variables.const[0:prop_prob.mce]) / temp_variables.eps
            grad_auglag[prop_prob.dim_n:(prop_prob.dim_n + prop_prob.mce)] = temp_variables.const[0:prop_prob.mce]
            QM[0:prop_prob.dim_n, 0:prop_prob.mce] = temp_variables.Q1
        if prop_prob.mieq > 0:
            grad_auglag[0:prop_prob.dim_n] += np.matmul(np.transpose(temp_variables.G), temp_variables.w) / (
                    temp_variables.eps * temp_variables.qv + 1e-20)
            grad_auglag[(prop_prob.dim_n + prop_prob.mce):] = temp_variables.w
            QM[0:prop_prob.dim_n, prop_prob.mce:] = temp_variables.Q2
        QM[prop_prob.dim_n:, :] = temp_variables.M
        grad_auglag += hyper.eta * (np.matmul(QM, new_vec[prop_prob.dim_n:]))
        temp_variables.grad_aug_lag1 = grad_auglag
        temp_variables.sol = np.concatenate((delta_x, delta_dual), axis=None)
        temp_variables.vec = new_vec

        # calculate the Hessian matrix for the Newton step, equation (21)
        if hyper.Newton:
            H_N = np.zeros(shape=(m, m))
            H_N21 = np.zeros(shape=(prop_prob.mce + prop_prob.mieq, prop_prob.dim_n))
            H_N22 = np.zeros(shape=(prop_prob.mce + prop_prob.mieq, prop_prob.mce + prop_prob.mieq))
            H_N11 = np.copy(B)
            mtx1 = np.zeros(shape=(prop_prob.mce + prop_prob.mieq, prop_prob.dim_n))
            mtx2 = np.zeros(shape=(prop_prob.mce + prop_prob.mieq, prop_prob.dim_n))
            g_ = np.copy(temp_variables.const[prop_prob.mce:])

            one_ = np.ones(shape=(prop_prob.mieq,))
            if prop_prob.mce > 0:
                JB = np.matmul(temp_variables.Jac[0:prop_prob.mce, :], B)
                H_N11 += hyper.eta * np.matmul(np.transpose(JB), JB) + np.matmul(
                    np.transpose(temp_variables.Jac[0:prop_prob.mce, :]),
                    temp_variables.Jac[0:prop_prob.mce, :]) / temp_variables.eps
                mtx1[0:prop_prob.mce, :] = temp_variables.Jac[0:prop_prob.mce, :]
                mtx2[0:prop_prob.mce, :] = temp_variables.Jac[0:prop_prob.mce, :]
            if prop_prob.mieq > 0:
                GB = np.matmul(temp_variables.G, B)
                H_N11 += hyper.eta * np.matmul(np.transpose(GB), GB)
                G_ = np.zeros(shape=(prop_prob.mieq, prop_prob.dim_n))
                if len(idx) > 0:
                    H_N11 += np.matmul(np.transpose(G_a),G_a)
                    G_[idx, :] = np.copy(G_a)
                    g_[idx] = 0.0
                    one_[idx] = 0.0
                mtx1[prop_prob.mce:,:] = G_
                mtx2[prop_prob.mce:,:] = np.copy(temp_variables.G)
            M_ = np.copy(M)
            M_[prop_prob.mce:, prop_prob.mce:] = GG + np.diag(g_ ** 2)
            H_N21 = mtx1 + hyper.eta * np.matmul(M_, np.matmul(mtx2, B))
            H_N22 = hyper.eta * np.matmul(M_, M_)
            if prop_prob.mieq > 0:
                H_N22[prop_prob.mce:, prop_prob.mce:] += - temp_variables.eps * temp_variables.qv * np.diag(one_)
            H_N[0:prop_prob.dim_n,0:prop_prob.dim_n] = H_N11
            H_N[0:prop_prob.dim_n,prop_prob.dim_n:] = np.transpose(H_N21)
            H_N[prop_prob.dim_n:, 0:prop_prob.dim_n] = H_N21
            H_N[prop_prob.dim_n:, prop_prob.dim_n:] = H_N22

            D, U = np.linalg.eigh(a=H_N)
            D = np.maximum(np.minimum(D, hyper.max_eig), hyper.min_eig)
            D = np.tile(np.reshape(D, (len(D), 1)), len(D))
            H_N = np.matmul(U, np.multiply(D, np.transpose(U)))

            temp_variables.H_N = H_N

            # temp_variables.H_N = np.identity(n=m)
        return True
    except:
        # print("An error occurred when solving SQP subproblem (12).")
        m = prop_prob.dim_n + prop_prob.mce + prop_prob.mieq
        temp_variables.grad_aug_lag1 = np.zeros(shape=(m,))
        temp_variables.sol = np.zeros(shape=(m,))
        temp_variables.vec = np.zeros(shape=(m,))
        return False


# check whether equation (18) holds
def check_sufficient_decrease(temp_variables, hyper):
    if np.sum(temp_variables.grad_aug_lag1 * temp_variables.sol) <= - hyper.eta / 2 * np.sum(temp_variables.vec ** 2):
        return True
    else:
        return False


# step2: set epsilon
def set_epsilon(prob, prop_prob, variables, temp_variables, hyper):
    # try:
    get_grad_aug_lag(prob, prop_prob, variables, temp_variables, hyper)
    ck1 = check_feas_gradlag(prob, prop_prob, variables, temp_variables, hyper)
    ck2 = solve_subp(prob, prop_prob, variables, temp_variables, hyper)
    ck3 = check_sufficient_decrease(temp_variables, hyper)
    while not ck1 or (ck2 and not ck3):
        temp_variables.eps = temp_variables.eps / hyper.rho
        get_grad_aug_lag(prob, prop_prob, variables, temp_variables, hyper)
        ck1 = check_feas_gradlag(prob, prop_prob, variables, temp_variables, hyper)
        ck2 = solve_subp(prob, prop_prob, variables, temp_variables, hyper)
        ck3 = check_sufficient_decrease(temp_variables, hyper)

        if temp_variables.eps < 1e-6:
            break
    return False
    # except:
    #     print("An error occurred when setting epsilon.")
    #     return True


# check whether equation (19) holds
def check_not_sufficient_decrease(temp_variables, hyper):
    grad_auglag2 = temp_variables.grad_aug_lag - temp_variables.grad_aug_lag1
    if np.sum(grad_auglag2 * temp_variables.sol) >= hyper.eta / 4 * np.sum(temp_variables.vec ** 2):
        return True
    else:
        return False


# step3: decide step
def decide_step(prob, prop_prob, variables, temp_variables, hyper):
    # try:
    ck1 = solve_subp(prob, prop_prob, variables, temp_variables, hyper)
    if not ck1:
        temp_variables.sol = -temp_variables.grad_aug_lag
    else:
        ck2 = check_not_sufficient_decrease(temp_variables, hyper)
        if ck2:
            if hyper.Newton:
                sol = scipy.linalg.solve(temp_variables.H_N, -temp_variables.grad_aug_lag, assume_a='sym')
                temp_variables.sol = sol
            else:
                temp_variables.sol = -temp_variables.grad_aug_lag
    # print(np.sum(temp_variables.sol ** 2))
    return False
    # except:
    #     print("An error occurred when deciding the step.")
    #     return True


# obtain the noise of gradient, given the step size xi_2t
def get_noise_grad2(prop_prob, temp_variables, hyper, xi_2t):
    if hyper.noise_type == "gaussian":
        omega = (np.identity(n=prop_prob.dim_n) + 1.0)
        temp_variables.noi_grad2 = np.sqrt(hyper.noi_std_grad_hess[0]) * \
                                   np.random.multivariate_normal(mean=np.zeros(shape=(prop_prob.dim_n,)), cov=omega,
                                                                 size=1)[0] / np.sqrt(xi_2t)
    elif hyper.noise_type == "t_distribution":
        temp_variables.noi_grad2 = np.random.standard_t(df=hyper.noi_stu_t_freed[0], size=(prop_prob.dim_n,)) / np.sqrt(
            xi_2t)


# to satisfy (24) and (25)
# equivalently to satisfy (41)
def get_sample_size_sec(prob, prop_prob, variables, temp_variables, hyper):
    aux3 = hyper.x_f * temp_variables.delta ** 2 + 1e-10

    xi_2t = 1
    num_loop = 0
    max_loop = 10
    while num_loop < max_loop:
        num_loop += 1
        get_noise_grad(prop_prob, temp_variables, hyper, xi_2t)
        get_grad_aug_lag(prob, prop_prob, variables, temp_variables, hyper)
        aux1 = np.sum(temp_variables.grad_aug_lag * temp_variables.sol)
        aux2 = hyper.k_f ** 2 * variables.alpha ** 4 * aux1 ** 2 + 1e-20
        xi_2t_trial = hyper.c * np.log(prop_prob.dim_n / hyper.p_f) * np.minimum(aux2, aux3)
        variables.xi_2t = xi_2t

        if xi_2t_trial <= xi_2t:
            break
        else:
            xi_2t = xi_2t * 5

    # for simplicity, we let tau_t = xi_t
    get_noise_hess(prop_prob, temp_variables, hyper, variables.xi_2t)


# step5: line search
def line_search(prob, prop_prob, variables, temp_variables, hyper, const_trial, a_trial):
    auglag_trial = prob.obj(temp_variables.x_trial)
    _, J_trial = prob.cons(temp_variables.x_trial, gradient=True)
    Jac_trial = np.zeros(shape=(prop_prob.mc, prop_prob.dim_n))

    c_eq = const_trial[prop_prob.ice] - prop_prob.cl[prop_prob.ice]
    c_cl = - const_trial[prop_prob.icl] + prop_prob.cl[prop_prob.icl]
    c_cu = const_trial[prop_prob.icu] - prop_prob.cu[prop_prob.icu]
    c_xl = prop_prob.xl[prop_prob.ixl] - variables.x[prop_prob.ixl]
    c_xu = variables.x[prop_prob.ixu] - prop_prob.xu[prop_prob.ixu]

    const_trial_ = np.concatenate((c_eq, c_cl, c_cu, c_xl, c_xu), axis=None)
    g_trial = const_trial_[prop_prob.mce:]

    _, gradf_trial = prob.obj(temp_variables.x_trial, gradient=True)
    grad_lag_trial = gradf_trial + temp_variables.noi_grad1

    if prop_prob.mce > 0:
        auglag_trial += np.sum(temp_variables.dual_general_trial[0:prop_prob.mce] * const_trial_[0:prop_prob.mce])
        auglag_trial += np.sum(c_eq ** 2) / (2 * temp_variables.eps)
        Jac_trial[0:prop_prob.mce, :] = J_trial[prop_prob.ice, :]
    if prop_prob.mcl > 0:
        auglag_trial += np.sum(
            temp_variables.dual_general_trial[prop_prob.mce:(prop_prob.mce + prop_prob.mcl)] * const_trial_[
                                                                                               prop_prob.mce:(
                                                                                                       prop_prob.mce + prop_prob.mcl)])
        Jac_trial[prop_prob.mce:(prop_prob.mce + prop_prob.mcl), :] = - J_trial[prop_prob.icl, :]
    if prop_prob.mcu > 0:
        auglag_trial += np.sum(
            temp_variables.dual_general_trial[(prop_prob.mce + prop_prob.mcl):] * const_trial_[
                                                                                  (
                                                                                          prop_prob.mce + prop_prob.mcl):prop_prob.mc])
        Jac_trial[(prop_prob.mce + prop_prob.mcl):, :] = J_trial[prop_prob.icu, :]

    av_trial = temp_variables.v - a_trial
    dual_ieq_trial = temp_variables.dual_general_trial[prop_prob.mce:]
    if prop_prob.nxlu > 0:
        dual_ieq_trial = np.concatenate((dual_ieq_trial, temp_variables.dual_bound_trial), axis=None)
        auglag_trial += np.sum(temp_variables.dual_bound_trial * const_trial_[prop_prob.mc:])
    qv_trial = av_trial / (1 + np.sum(dual_ieq_trial ** 2))
    bv_trial = np.minimum(0.0, g_trial + temp_variables.eps * qv_trial * dual_ieq_trial)
    auglag_trial += (np.sum(g_trial ** 2) - np.sum(bv_trial ** 2)) / (2 * temp_variables.eps * qv_trial + 1e-20)

    if prop_prob.mc > 0:
        grad_lag_trial += np.matmul(np.transpose(Jac_trial), temp_variables.dual_general_trial)
    if prop_prob.nxlu > 0:
        dual_bound1 = np.zeros(shape=(prop_prob.dim_n,))
        dual_bound2 = np.zeros(shape=(prop_prob.dim_n,))
        dual_bound1[prop_prob.ixl] = temp_variables.dual_bound_trial[0:prop_prob.nxl]
        dual_bound2[prop_prob.ixu] = temp_variables.dual_bound_trial[prop_prob.nxl:]

        grad_lag_trial += - dual_bound1 + dual_bound2

    if prop_prob.mce > 0:
        auglag_trial += hyper.eta / 2 * np.sum(
            np.matmul(Jac_trial[0:prop_prob.mce, :], grad_lag_trial) ** 2)

    if prop_prob.nxlu > 0:
        G_trial = np.zeros(shape=(prop_prob.mieq, prop_prob.dim_n))
        G_trial[0:(prop_prob.mcl + prop_prob.mcu), :] = Jac_trial[prop_prob.mce:, :]
        I = np.identity(n=prop_prob.dim_n)
        if prop_prob.nxl > 0:
            G_trial[(prop_prob.mcl + prop_prob.mcu):(prop_prob.mcl + prop_prob.mcu + prop_prob.nxl), :] = -I[
                                                                                                           prop_prob.ixl[
                                                                                                               0], :]
        if prop_prob.nxu > 0:
            G_trial[(prop_prob.mcl + prop_prob.mcu + prop_prob.nxl):, :] = I[prop_prob.ixu[0], :]
    else:
        G_trial = Jac_trial[prop_prob.mce:, :]

    if prop_prob.mieq > 0:
        auglag_trial += hyper.eta / 2 * np.sum(
            (np.matmul(G_trial, grad_lag_trial) ** 2 + (g_trial ** 2 * dual_ieq_trial) ** 2))

    aug_lag = 0
    aug_lag += prob.obj(variables.x)
    aug_lag += np.sum(temp_variables.const * np.concatenate((variables.dual_general, variables.dual_bound), axis=None))
    grad_lag = temp_variables.grad_lag + temp_variables.noi_grad1
    if prop_prob.mce > 0:
        aug_lag += np.sum(temp_variables.const[0:prop_prob.mce] ** 2) / (2 * temp_variables.eps) + np.sum(
            np.matmul(temp_variables.Jac[0:prop_prob.mce, :], grad_lag) ** 2) * hyper.eta / 2
    if prop_prob.mieq > 0:
        g_ = temp_variables.const[prop_prob.mce:]
        dual_ieq = np.concatenate((variables.dual_general[prop_prob.mce:], variables.dual_bound), axis=None)
        b_ = np.minimum(0.0, g_ + temp_variables.eps * temp_variables.qv * dual_ieq)
        aug_lag += np.sum(g_ ** 2 - b_ ** 2) / (2 * temp_variables.eps * temp_variables.qv + 1e-20) + np.sum(
            (np.matmul(temp_variables.G, grad_lag) + g_ ** 2 * dual_ieq) ** 2) * hyper.eta / 2

    # print(auglag_trial)
    # print(aug_lag)
    # print(hyper.beta * variables.alpha * np.sum(
    #         (temp_variables.grad_aug_lag * temp_variables.sol)))
    # print(variables.alpha)
    if auglag_trial <= aug_lag + hyper.beta * variables.alpha * np.sum(
            (temp_variables.grad_aug_lag * temp_variables.sol)):
        variables.x = temp_variables.x_trial
        if prop_prob.mc > 0:
            variables.dual_general = temp_variables.dual_general_trial
        if prop_prob.nxlu > 0:
            variables.dual_bound = temp_variables.dual_bound_trial
        variables.alpha = np.minimum(hyper.rho * variables.alpha, hyper.alpha_max)
        if - hyper.beta * variables.alpha * np.sum(
                (temp_variables.grad_aug_lag * temp_variables.sol)) >= temp_variables.delta:
            temp_variables.delta = np.minimum(temp_variables.delta * hyper.rho, 1e6)
        else:
            temp_variables.delta = temp_variables.delta / hyper.rho
    else:
        variables.alpha = np.maximum(variables.alpha / hyper.rho, 1e-8)
        temp_variables.delta = temp_variables.delta / hyper.rho


# step4-5: estimate merit function and then do the line search
def estimate_merit_function(prob, prop_prob, variables, temp_variables, hyper):
    try:
        temp_variables.x_trial = variables.x + variables.alpha * temp_variables.sol[0:prop_prob.dim_n]
        if prop_prob.mc > 0:
            temp_variables.dual_general_trial = variables.dual_general + variables.alpha * temp_variables.sol[
                                                                                           prop_prob.dim_n:(
                                                                                                   prop_prob.dim_n + prop_prob.mc)]
        if prop_prob.nxlu > 0:
            temp_variables.dual_bound_trial = variables.dual_bound + variables.alpha * temp_variables.sol[(
                                                                                                                  prop_prob.dim_n + prop_prob.mc):]

        const_trial = prob.cons(temp_variables.x_trial)
        c_cl = np.maximum(- const_trial[prop_prob.icl] + prop_prob.cl[prop_prob.icl], 1e-10) ** 3
        c_cu = np.maximum(const_trial[prop_prob.icu] - prop_prob.cu[prop_prob.icu], 1e-10) ** 3
        c_xl = np.maximum(prop_prob.xl[prop_prob.ixl] - temp_variables.x_trial[prop_prob.ixl], 1e-10) ** 3
        c_xu = np.maximum(temp_variables.x_trial[prop_prob.ixu] - prop_prob.xu[prop_prob.ixu], 1e-10) ** 3
        a_trial = np.sum(c_cl) + np.sum(c_cu) + np.sum(c_xl) + np.sum(c_xu)
        if a_trial > temp_variables.v / 2:
            j = np.ceil(np.log(2 * a_trial / temp_variables.v + 1e-20) / np.log(hyper.rho))
            temp_variables.v = temp_variables.v * hyper.rho ** (j + 1)
            # print(a_trial)
            # print(np.sqrt(np.sum(temp_variables.sol ** 2)))
            return False

        get_sample_size_sec(prob, prop_prob, variables, temp_variables, hyper)
        line_search(prob, prop_prob, variables, temp_variables, hyper, const_trial, a_trial)
        return False
    except:
        print("An error occurred when estimating the merit function and doing the line search.")
        return True


# check kkt residual and feasibility error
def cal_kkt_res_cont(prob, prop_prob, variables, temp_variables):
    try:
        feas = 0.0
        evaluate_constraints(prob, prop_prob, variables, temp_variables)
        const = np.zeros(shape=np.shape(temp_variables.const))
        if prop_prob.mce > 0:
            const[0:prop_prob.mce] = temp_variables.const[0:prop_prob.mce]
        if prop_prob.mieq > 0:
            const[prop_prob.mce:] = np.maximum(temp_variables.const[prop_prob.mce:], 0.0)
        feas += np.sqrt(np.sum(const ** 2))
        kkt = np.sqrt(np.sum(temp_variables.grad_lag ** 2))
        return False, kkt, feas
    except:
        print("An error occurred when calculating kkt residual.")
        return True, None, None
