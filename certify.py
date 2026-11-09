"""Certifiable uncertainty on the operating set (Section IV applied to Section V; Fig. 1(h)).

Residual of the tightened CBF condition at x in X for the box rho dtheta_0, eq. (residual):

    D(x; rho dtheta_0) = sup_{u in U} a(x, u) + alpha(H(x)) - rho psi(x)^T dtheta_0,

affine in rho, so on a certification set Omega the level of eq. (rhostar) is the ratio infimum

    rho*_Omega = inf_{x in Omega, psi^T dtheta_0 > 0} (sup_u a + alpha(H)) / (psi^T dtheta_0)

when sup_u a + alpha(H) >= 0 on Omega (Assumption on nominal safety), and 0 otherwise.  The sets
Omega_eps = {x in X : H(x) >= -eps} are nested, so the sweep over eps bins every grid state by
the smallest eps that contains it and takes running minima over the bins; the grid is visited
one h-slice at a time (memory stays small at the 0.05 m spacing, 28 million states in X).

Also: rho*_S by bisection on the sign of min_S D(rho) (Remark on computation), a check that brackets
the ratio infimum; the certified lower bound of the Remark, min_Omega D(rho) >= min_grid D(rho) -
(Lipschitz constants x grid spacing), as `certified_lower_bound`; the sharpness state of the
Corollary on feasibility; and the threshold on gamma_- below which rho*_X = rho*_S.
"""
import numpy as np

import params as p
import model as m

GKEY = lambda g: f"{g:g}"           # json key of a gamma value


def axes(step=p.grid_step):
    h = np.round(np.arange(p.d_min, p.h_max + 1e-9, step), 10)
    v = np.round(np.arange(0.0, p.v_max + 1e-9, step), 10)
    return h, v


def slice_X(h, v):
    """All grid states of X with gap h: arrays (vA, vE) with v_E - v_A <= w_max."""
    VA, VE = np.meshgrid(v, v, indexing="ij")
    VA, VE = VA.ravel(), VE.ravel()
    keep = (VE - VA) <= p.w_max + 1e-12
    return VA[keep], VE[keep]


def _bin_min(k, val, K):
    """Minimum of val within each bin k (0..K-1) and the first index attaining it; inf / -1 for empty bins."""
    order = np.argsort(k, kind="stable")
    ks, vs = k[order], val[order]
    starts = np.searchsorted(ks, np.arange(K + 1))
    mins = np.full(K, np.inf)
    args = np.full(K, -1)
    for j in range(K):
        a, b = starts[j], starts[j + 1]
        if b > a:
            i = a + int(np.argmin(vs[a:b]))
            mins[j], args[j] = vs[i], order[i]
    return mins, args


class _Running:
    """Running minimum per eps bin with the attaining state (smallest (h, vA, vE) among ties)."""

    def __init__(self, K):
        self.val = np.full(K, np.inf)
        self.state = np.full((K, 3), np.nan)

    def update(self, mins, args, h, VA, VE):
        better = mins < self.val
        self.val[better] = mins[better]
        for j in np.flatnonzero(better):
            self.state[j] = (h, VA[args[j]], VE[args[j]])

    def prefix(self):
        """Nested sets: the minimum over Omega_eps is the running minimum over the bins <= eps."""
        val = np.minimum.accumulate(self.val)
        state = self.state.copy()
        for j in range(1, len(val)):
            if not self.val[j] < val[j - 1]:
                state[j] = state[j - 1]
        return val, state


def pareto_front(A, B, A0=None, B0=None):
    """Points that can minimise A - rho B for some rho >= 0 (B descending, A below every earlier one)."""
    if A0 is not None:
        A, B = np.concatenate([A0, A]), np.concatenate([B0, B])
    o = np.argsort(-B, kind="stable")
    Ao, Bo = A[o], B[o]
    keep = np.empty(len(o), dtype=bool)
    keep[0] = True
    keep[1:] = Ao[1:] < np.minimum.accumulate(Ao)[:-1]
    return Ao[keep], Bo[keep]


def sweep(step=p.grid_step, eps_step=p.eps_step, gammas=p.gamma_minus, alpha_mode="one_sided",
          dtheta_0=p.dtheta, verbose=False):
    """rho*_Omega_eps, min_Omega_eps D(x; 0) and the certified lower bound for every eps and gamma.

    alpha_mode 'one_sided': alpha(H) = gamma H (H >= 0), gamma_minus H (H < 0) with gamma_minus in
    gammas (the paper); 'single': alpha(H) = gamma H on all of X with gamma in gammas (figE1(b))."""
    hs, vs = axes(step)
    eps = np.round(np.arange(0.0, p.eps_max + 1e-9, eps_step), 10)
    K = len(eps)
    s0, s1 = lipschitz_slack(step, step)
    run = {GKEY(g): {"ratio": _Running(K), "A": _Running(K), "cert": _Running(K)} for g in gammas}
    front_A = front_B = None                      # Pareto front of (A, psi^T dtheta_0) on S, alpha = gamma H
    n_X = 0
    for h in hs:
        VA, VE = slice_X(h, vs)
        n_X += len(VA)
        Hval, sup, psi_d = m.residual_terms(h, VA, VE, dtheta_0)
        k = np.clip(np.ceil((-Hval - 1e-9) / eps_step).astype(int), 0, K - 1)   # smallest eps bin containing x
        pos = psi_d > 1e-12
        safe = Hval >= -1e-9
        for g in gammas:
            A = sup + (g * Hval if alpha_mode == "single" else m.alpha(Hval, g))
            ratio = np.where(pos, A / np.where(pos, psi_d, 1.0), np.inf)
            cert = (A - s0) / (psi_d + s1)
            R = run[GKEY(g)]
            for key, val in (("ratio", ratio), ("A", A), ("cert", cert)):
                mins, args = _bin_min(k, val, K)
                R[key].update(mins, args, h, VA, VE)
            if alpha_mode == "one_sided" and g == gammas[0] and safe.any():
                front_A, front_B = pareto_front(A[safe], psi_d[safe], front_A, front_B)
        if verbose and int(round((h - p.d_min) / step)) % 100 == 0:
            print(f"  h = {h:5.2f} m")
    out = {"eps": [float(e) for e in eps], "alpha": alpha_mode, "step": float(step), "eps_step": float(eps_step),
           "n_points_X": int(n_X), "per_gamma": {}}
    for g in gammas:
        R = run[GKEY(g)]
        ratio, ratio_state = R["ratio"].prefix()
        Amin, A_state = R["A"].prefix()
        cert, _ = R["cert"].prefix()
        feas = Amin >= 0.0
        rho = np.where(feas, ratio, 0.0)
        rho_cert = np.where(Amin - s0 >= 0.0, cert, 0.0)
        bind = np.where(feas[:, None], ratio_state, A_state)
        ok = rho >= 1.0
        out["per_gamma"][GKEY(g)] = {
            "rho_star": [float(r) for r in rho],
            "min_D0": [float(a) for a in Amin],
            "binding_state": [[float(v) for v in s] for s in bind],
            "binding_H": [float(m.H(*s)) for s in bind],
            "min_D0_state": [[float(v) for v in s] for s in A_state],
            "certified_lower_bound": [float(r) for r in rho_cert],
            "eps_max": float(eps[ok].max()) if ok.any() else None,
            "monotone_nonincreasing": bool(np.all(np.diff(rho) <= 1e-12)),
        }
    if front_A is not None:
        out["front_S"] = {"A": [float(a) for a in front_A], "psi_dtheta_0": [float(b) for b in front_B]}
    return out


def min_D_front(front, rho):
    """min_S D(x; rho dtheta_0) from the Pareto front of S."""
    A, B = np.asarray(front["A"]), np.asarray(front["psi_dtheta_0"])
    return float(np.min(A - rho * B))


def bisection(front, bracket=p.rho_bracket, tol=p.rho_tol):
    """rho*_S by bisection on min_S D(rho) = 0 (Remark on computation)."""
    lo, hi = bracket
    assert min_D_front(front, lo) >= 0.0 > min_D_front(front, hi), "bracket invalid"
    mids = []
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        mids.append(mid)
        if min_D_front(front, mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return {"bisection": 0.5 * (lo + hi), "bracket": [lo, hi], "iterations": mids}


def lipschitz_slack(dh, dv):
    """min_Omega D(rho) >= min_grid D(rho) - (s0 + s1 rho) with s(rho) = L_h dh + (L_vA + L_vE) dv.

    Every x in Omega_eps has a grid point of Omega_eps within one step per axis: round h and v_A up
    and v_E down (H does not decrease, v_E - v_A does not increase, and h_max, v_max are grid points),
    so the Lipschitz constants times the full spacing bound the gap between the grid minimum and the
    infimum over the set."""
    s = lambda L: L[0] * dh + (L[1] + L[2]) * dv
    s_0 = s(m.lipschitz_constants(0.0))
    return float(s_0), float(s(m.lipschitz_constants(1.0)) - s_0)


def rho_star_X(gammas, step=p.grid_step, dtheta_0=p.dtheta):
    """rho*_X for alpha(H) = gamma H (H >= 0), gamma_minus H (H < 0), several gamma_minus at once."""
    hs, vs = axes(step)
    ratio_min = {GKEY(g): np.inf for g in gammas}
    A_min = {GKEY(g): np.inf for g in gammas}
    for h in hs:
        VA, VE = slice_X(h, vs)
        Hval, sup, psi_d = m.residual_terms(h, VA, VE, dtheta_0)
        pos = psi_d > 1e-12
        for g in gammas:
            A = sup + m.alpha(Hval, g)
            ratio_min[GKEY(g)] = min(ratio_min[GKEY(g)], float(np.min(A[pos] / psi_d[pos])))
            A_min[GKEY(g)] = min(A_min[GKEY(g)], float(np.min(A)))
    return {g: (ratio_min[g] if A_min[g] >= 0.0 else 0.0) for g in ratio_min}


def certify(step=p.grid_step, eps_step=p.eps_step, verbose=False):
    """Everything Fig. 1(h), its paragraph and figE1 need, as one json-serialisable dict."""
    say = print if verbose else (lambda *a: None)
    say(f"one-sided alpha sweep on the {step} m grid ...")
    one = sweep(step, eps_step, p.gamma_minus, "one_sided", verbose=verbose)
    say("single-slope sweep ...")
    single = sweep(step, eps_step, p.gamma_single, "single", verbose=verbose)
    g0 = GKEY(p.gamma)
    rho_S = one["per_gamma"][g0]["rho_star"][0]
    state_S = one["per_gamma"][g0]["binding_state"][0]
    bis = bisection(one["front_S"])
    s0, s1 = lipschitz_slack(step, step)
    # Corollary on feasibility: at rho = rho_sharpness rho*_S the tightened condition fails on the boundary of S
    rho_s = p.rho_sharpness * rho_S
    Hb, supb, psib = m.residual_terms(*(np.array([v]) for v in state_S))
    sharp = {"rho": rho_s, "state": state_S, "H": float(Hb[0]), "sup_Hdot": float(supb[0] - rho_s * psib[0])}
    # gamma_- threshold: rho*_X = rho*_S iff gamma_- <= (sup_u a - rho*_S psi^T dtheta_0) / (-H) at the corner of X
    corner = [p.d_min, 0.0, p.w_max]
    Hc, supc, psic = m.residual_terms(*(np.array([v]) for v in corner))
    g_thr = float((supc[0] - rho_S * psic[0]) / (-Hc[0]))
    say("rho*_X around the threshold ...")
    extra = sorted(set(list(p.gamma_minus) + [round(g_thr, 3), 0.11, 0.12, 0.0]))
    rho_X = rho_star_X(extra, step)
    say("done")
    return {
        "grid": {"step": float(step), "eps_step": float(eps_step), "n_h": len(axes(step)[0]),
                 "n_v": len(axes(step)[1]), "n_points_X": one["n_points_X"],
                 "X": {"h": [p.d_min, p.h_max], "v": [0.0, p.v_max], "vE_minus_vA_max": p.w_max}},
        "dtheta_0": [float(v) for v in p.dtheta],
        "eps": one["eps"],
        "one_sided": one["per_gamma"],
        "single": single["per_gamma"],
        "rho_star_S": {"ratio_infimum": rho_S, "binding_state": state_S,
                       "budget_used_percent": 100.0 / rho_S, **bis,
                       "bisection_brackets_ratio_infimum": bool(bis["bracket"][0] <= rho_S <= bis["bracket"][1]),
                       "certified_lower_bound": one["per_gamma"][g0]["certified_lower_bound"][0]},
        "min_D0_X": {"value": one["per_gamma"][g0]["min_D0"][-1],
                     "state": one["per_gamma"][g0]["min_D0_state"][-1]},
        "sharpness": sharp,
        "gamma_minus_threshold": {"value": g_thr, "corner_state": corner, "corner_H": float(Hc[0]),
                                  "corner_sup_a": float(supc[0]), "corner_psi_dtheta_0": float(psic[0]),
                                  "rho_star_X": rho_X},
        "lipschitz": {"s0": s0, "s1": s1, "constants_rho0": list(m.lipschitz_constants(0.0)),
                      "constants_rho_star_S": list(m.lipschitz_constants(rho_S)),
                      "bound": "min_Omega D(rho) >= min_grid D(rho) - (s0 + s1 rho)"},
        "front_S": one["front_S"],
    }


if __name__ == "__main__":
    import json
    import sys
    R = certify(step=float(sys.argv[1]) if len(sys.argv) > 1 else p.grid_step, verbose=True)
    print(json.dumps({k: R[k] for k in ("rho_star_S", "sharpness", "gamma_minus_threshold", "min_D0_X")}, indent=1))
    for g, rows in R["one_sided"].items():
        print(f"gamma_- = {g}: eps_max = {rows['eps_max']}, rho*_X = {rows['rho_star'][-1]:.4f}")
