"""Closed-loop simulation of Section V: leader A with the robust CBF filter, n reactive followers.

Forward Euler with step dt over T seconds.  Positions are rear bumpers; the origin is the
leader's rear bumper at t = 0, so the stop line sits at s_line = 45 m as in the paper.  Speeds
are clamped at zero (the vehicles do not reverse).
"""
import numpy as np

import params as p
import model as m

MARGINS = ("none", "str", "sd", "cl")
PRIORITIES = ("stop", "safety")


def theta_samples(n_samples=p.n_samples, seed=p.seed):
    """Fig. 1(b): the four vertices of Theta followed by n_samples uniform samples in Theta."""
    d1, d2 = p.dtheta
    verts = np.array([[s1 * d1, s2 * d2] for s1 in (1, -1) for s2 in (1, -1)])
    rng = np.random.default_rng(seed)
    inner = rng.uniform(-1.0, 1.0, size=(n_samples, 2)) * p.dtheta
    return p.theta_hat + np.vstack([verts, inner])


def simulate(theta, margin, n=1, v_ref=0.0, stop_line=False, priority="stop",
             init="rest", gains="shared", T=p.T):
    """One closed-loop run.

    theta:     the followers' true gains (shared by the chain)
    margin:    'none' | 'str' | 'sd' | 'cl'  (eta in {0, eta_str, eta_sd, eta_cl})
    n:         number of followers; E_1 starts at the equilibrium gap behind A at v_ref_0
    v_ref:     final reference speed of the leader (v_ref_0 until t_step, then v_ref)
    stop_line: add the braking-distance CBF of the mandatory stop (u <= u_B)
    priority:  'stop' (the paper's rule) | 'safety', see model.apply_filter
    init:      E_2..E_n 'rest' (from standstill, gap_0 apart) | 'equilibrium' (all at v_ref_0)
    gains:     'shared' | 'independent' bookkeeping of eta_sd, see model.eta_sd

    Returns a dict of time series: t, h (= h_1), H, vA, vE (= v_1), u, u_d, u_H, u_B (nan without
    stop line), d = s_line - s_A, eta_str, eta_sd, eta_cl, infeasible (0/1), h_min_queue
    (smallest gap among E_2..E_n, inf for n = 1), and the regressor rows Phi_11, Phi_12."""
    assert margin in MARGINS and priority in PRIORITIES and n >= 1
    theta = np.asarray(theta, float)
    steps = int(round(T / p.dt))
    h0 = p.h_0()
    sA, vA = 0.0, p.v_ref_0
    s = np.empty(n)
    v = np.full(n, p.v_ref_0)
    s[0] = sA - p.L - h0                                   # E_1 at the equilibrium gap
    for i in range(1, n):
        if init == "equilibrium":
            s[i] = s[i - 1] - p.L - h0
        elif init == "rest":
            s[i], v[i] = s[i - 1] - gap0_rear(), 0.0
        else:
            raise ValueError(init)
    keys = ("t", "h", "H", "vA", "vE", "u", "u_d", "u_H", "u_B", "d", "eta_str", "eta_sd", "eta_cl",
            "infeasible", "h_min_queue", "Phi_11", "Phi_12")
    log = {k: np.empty(steps) for k in keys}
    for k in range(steps):
        t = k * p.dt
        gaps = np.concatenate([[sA], s[:-1]]) - s - p.L     # h_1, ..., h_n (front of E_i to rear of its predecessor)
        v_pred = np.concatenate([[vA], v[:-1]])
        Phi = m.Phi(gaps, v, v_pred)
        Hval = m.H(gaps[0], vA, v[0])
        e_str, e_sd, e_cl = m.eta_str(Phi), m.eta_sd(Phi, gains), m.eta_cl(n)
        eta = {"none": 0.0, "str": e_str, "sd": e_sd, "cl": e_cl}[margin]
        ud = m.u_d(vA, p.v_ref_0 if t < p.t_step else v_ref)
        uH = m.u_H(gaps[0], vA, v[0], Phi[0], eta)
        uB = m.u_B(sA, vA) if stop_line else None
        u = m.apply_filter(ud, uH, uB, priority)
        acc = Phi @ theta                                   # true follower reactions, eq. (ovm)
        row = (t, gaps[0], Hval, vA, v[0], u, ud, uH, np.nan if uB is None else uB, p.s_line - sA,
               e_str, e_sd, e_cl, float(uB is not None and m.infeasible(uH, uB)),
               float(gaps[1:].min()) if n > 1 else np.inf, Phi[0, 0], Phi[0, 1])
        for key, val in zip(keys, row):
            log[key][k] = val
        sA += vA * p.dt
        s += v * p.dt
        vA = max(vA + u * p.dt, 0.0)
        v = np.maximum(v + acc * p.dt, 0.0)
    log["n"] = n
    return log


def gap0_rear():
    """Rear-to-rear spacing of E_2..E_n at rest: the gap gap_0 plus one vehicle length."""
    return p.gap_0 + p.L


def mean_tail(x, t_mean=p.t_mean):
    """Mean over the last t_mean seconds (Fig. 1(g))."""
    return float(np.mean(x[int(round((p.T - t_mean) / p.dt)):]))
