"""The car-following model of Section V and the quantities of Sections III-IV evaluated on it.

Vectorised over the last axis where it makes sense (grids in certify.py); the simulator calls
the same functions with scalars.
"""
import numpy as np

import params as p


# ------------------------------------------------------------------ follower policy, eq. (ovm)
def V(h):
    """Range policy V(h) = min(max(0, kappa (h - h_st)), v_max)."""
    return np.minimum(np.maximum(p.kappa * (h - p.h_st), 0.0), p.v_max)


def W(v):
    """Speed policy W(v) = min(v, v_max)."""
    return np.minimum(v, p.v_max)


def Phi(gaps, v, v_pred):
    """Regressor of the queue, eq. (queue): row i = [V(h_i) - v_i, W(v_{i-1}) - v_i], shape (n, 2).

    gaps, v: the gaps h_i and speeds v_i of the followers E_1..E_n; v_pred: the predecessor
    speeds (v_A, v_1, ..., v_{n-1}).  For n = 1 this is eq. (inst-blocks)."""
    gaps, v, v_pred = (np.atleast_1d(np.asarray(a, float)) for a in (gaps, v, v_pred))
    return np.column_stack([V(gaps) - v, W(v_pred) - v])


def pi_E(h, vE, vA, theta):
    """Follower acceleration pi_E(x; theta) = theta_1 (V(h) - v_E) + theta_2 (W(v_A) - v_E)."""
    return theta[0] * (V(h) - vE) + theta[1] * (W(vA) - vE)


# ------------------------------------------------------------------ barrier, eq. (car-following_CBF)
def H(h, vA, vE):
    """H(x) = h - d_min - tau (v_E - v_A)."""
    return (h - p.d_min) - p.tau * (vE - vA)


def c(n=1):
    """c(x) = [-tau, 0, ..., 0]: only E_1 enters dot H."""
    out = np.zeros(n)
    out[0] = -p.tau
    return out


def a(h, vA, vE, u, Phi_1):
    """a(x, u) = v_A - v_E + tau u - tau Phi_1 theta_hat, eq. (inst-ac); Phi_1 is the first row of Phi."""
    return (vA - vE) + p.tau * u - p.tau * (Phi_1 @ p.theta_hat)


def sup_a(h, vA, vE, Phi_1):
    """sup_{u in U} a(x, u) = a(x, u_max)."""
    return a(h, vA, vE, p.u_max, Phi_1)


def alpha(Hval, gamma_minus=p.gamma):
    """Extended class K function: gamma H for H >= 0, gamma_minus H for H < 0 (Fig. 1(h))."""
    return np.where(Hval >= 0.0, p.gamma * Hval, gamma_minus * Hval)


# ------------------------------------------------------------------ margins, eqs. (margin_*)
def delta_bar(n=1):
    """Uniform bound sup_X ||Phi delta theta|| on the queue: sqrt(n) v_max (dtheta_1 + dtheta_2)."""
    return np.sqrt(n) * p.v_max * float(p.dtheta.sum())


def psi(Phi_x):
    """psi(x)_k = |c(x) Phi_k(x)|, eq. (state-dependent_bound), for the regressor Phi_x of shape (n, 2)."""
    return np.abs(c(Phi_x.shape[0]) @ Phi_x)


def eta_str(Phi_x):
    """Structured margin eta_str(x) = psi(x)^T dtheta = sum_k |c Phi_k| dtheta_k (exact worst case over Theta)."""
    return float(psi(Phi_x) @ p.dtheta)


def eta_sd(Phi_x, gains="shared"):
    """State-dependent-radius margin eta_sd(x) = ||c(x)|| sum_k ||Phi_.k(x)|| dtheta_k, eq. (uniform_bound).

    gains="independent": every follower has its own pair (theta_1i, theta_2i), so Phi is block
    diagonal and every column norm reduces to a single |Phi_ik|; the sum then runs over all
    n columns per gain."""
    if gains == "independent":
        col = np.abs(Phi_x).sum(axis=0)
    else:
        col = np.sqrt((Phi_x ** 2).sum(axis=0))
    return float(p.tau * (col @ p.dtheta))


def eta_cl(n=1):
    """Classical margin eta_cl = ||c|| delta_bar_n = tau sqrt(n) v_max (dtheta_1 + dtheta_2)."""
    return p.tau * delta_bar(n)


def margin(name, Phi_x, gains="shared"):
    """eta in {0, eta_str, eta_sd, eta_cl} by name: 'none', 'str', 'sd', 'cl'."""
    if name == "none":
        return 0.0
    if name == "str":
        return eta_str(Phi_x)
    if name == "sd":
        return eta_sd(Phi_x, gains)
    if name == "cl":
        return eta_cl(Phi_x.shape[0])
    raise ValueError(name)


# ------------------------------------------------------------------ the filters
def u_d(vA, v_ref):
    """Nominal controller u_d = K_p (v_ref - v_A), saturated to U."""
    return float(np.clip(p.K_p * (v_ref - vA), p.u_min, p.u_max))


def u_H(h, vA, vE, Phi_1, eta):
    """Lower bound of the robust CBF condition a(x, u) + gamma H(x) >= eta:  u >= u_H(x)."""
    return (-p.gamma * H(h, vA, vE) + eta - (vA - vE) + p.tau * (Phi_1 @ p.theta_hat)) / p.tau


def B(sA, vA):
    """Braking-distance CBF of the mandatory stop: B = (s_line - s_A) - v_A^2 / (2 a_b)."""
    return (p.s_line - sA) - vA * vA / (2.0 * p.a_b)


def u_B(sA, vA):
    """Upper bound from dot B >= -gamma_s B:  u <= u_B(x) = a_b (gamma_s B - v_A) / max(v_A, v_B_min)."""
    return p.a_b * (p.gamma_s * B(sA, vA) - vA) / max(vA, p.v_B_min)


def apply_filter(ud, uH, uB=None, priority="stop"):
    """The applied input, projected onto U.

    Without a stop line: u = max(u_d, u_H).  With one, priority 'stop' (the paper's rule):
    u = min(max(u_d, u_H), u_B); priority 'safety': u = max(min(u_d, u_B), u_H)."""
    if uB is None:
        u = max(ud, uH)
    elif priority == "stop":
        u = min(max(ud, uH), uB)
    elif priority == "safety":
        u = max(min(ud, uB), uH)
    else:
        raise ValueError(priority)
    return float(np.clip(u, p.u_min, p.u_max))


def infeasible(uH, uB):
    """The two constraints are incompatible: u_H > u_B or u_H > u_max."""
    return uH > min(uB, p.u_max) + 1e-9


# ------------------------------------------------------------------ derived constants of Section V
def v_floor():
    """Lowest steady speed the classical filter certifies: kappa (tau delta_bar / gamma + d_min - h_st)."""
    return p.kappa * (p.tau * delta_bar(1) / p.gamma + p.d_min - p.h_st)


def H_settled_classical():
    """Level at which H settles under the classical margin: tau delta_bar / gamma = alpha^{-1}(eta_cl)."""
    return p.tau * delta_bar(1) / p.gamma


# ------------------------------------------------------------------ residual on the operating set, eq. (residual)
def residual_terms(h, vA, vE, dtheta_0=p.dtheta):
    """At states (h, v_A, v_E) of X (arrays): the pieces of D(x; rho dtheta_0) = slack(x) - rho psi(x)^T dtheta_0.

    Returns (H, sup_u a, psi^T dtheta_0) with psi = tau |Phi| for the one-follower regressor."""
    P1, P2 = V(h) - vE, W(vA) - vE
    Hval = H(h, vA, vE)
    sup = sup_a(h, vA, vE, np.column_stack([P1, P2]))
    psi_d = p.tau * (np.abs(P1) * dtheta_0[0] + np.abs(P2) * dtheta_0[1])
    return Hval, sup, psi_d


def lipschitz_constants(rho, gamma_minus=p.gamma):
    """Per-axis Lipschitz constants of x -> D(x; rho dtheta_0) on X (Remark on computation).

    D = (v_A - v_E) + tau u_max - tau (theta_1 P_1 + theta_2 P_2) + alpha(H) - rho tau (|P_1| dtheta_1 + |P_2| dtheta_2)
    with P_1 = V(h) - v_E, P_2 = W(v_A) - v_E, V' in {0, kappa}, W' in {0, 1}, dH = (1, tau, -tau), and the
    alpha term contributes at most max(gamma, gamma_minus).  Along v_E the first three terms are affine
    with slope -1 + tau (theta_1 + theta_2), so that slope enters as one constant."""
    g = max(p.gamma, gamma_minus)
    t, k = p.tau, p.kappa
    d1, d2 = p.dtheta
    th1, th2 = p.theta_hat
    L_h = k * t * (th1 + rho * d1) + g
    L_vA = 1.0 + t * th2 + g * t + rho * t * d2
    L_vE = abs(-1.0 + t * (th1 + th2)) + g * t + rho * t * (d1 + d2)
    return L_h, L_vA, L_vE
