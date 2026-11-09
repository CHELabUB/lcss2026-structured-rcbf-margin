#!/usr/bin/env python3
"""Run every simulation and the certification sweep of Section V; save the raw data.

Writes results/runs.npz (every time series drawn in Fig. 1 and the extended figures) and
results/results.json (every number quoted in the paper and the ReadMe, plus the certification
sweep).  stats.py prints the numbers, plotter.py draws the figures from these two files.

Usage: python generate_results.py [--step 0.05]      (--step 0.1: coarser grid in Fig. 1(h), faster)
"""
import argparse
import json
import os
import time

import numpy as np

import params as p
import model as m
import simulate as sim
import certify

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SERIES = ("t", "h", "H", "vA", "vE", "u", "u_d", "u_H", "u_B", "d", "eta_str", "eta_sd", "eta_cl",
          "infeasible", "h_min_queue", "Phi_11", "Phi_12")


def store(Z, prefix, log, keys=SERIES):
    for k in keys:
        Z[f"{prefix}/{k}"] = log[k]


def window(log):
    """Infeasible time [s] and first infeasible instant (None if feasible throughout)."""
    inf = log["infeasible"] > 0.5
    return float(inf.sum() * p.dt), (float(log["t"][int(np.argmax(inf))]) if inf.any() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--step", type=float, default=p.grid_step, help="grid spacing of the certification sweep")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    Z, R = {}, {"params": {
        "kappa": p.kappa, "h_st": p.h_st, "v_max": p.v_max, "theta_hat": p.theta_hat.tolist(),
        "dtheta": p.dtheta.tolist(), "L": p.L, "d_min": p.d_min, "tau": p.tau, "gamma": p.gamma,
        "U": [p.u_min, p.u_max], "K_p": p.K_p, "dt": p.dt, "T": p.T, "s_line": p.s_line, "a_b": p.a_b,
        "gamma_s": p.gamma_s, "theta_a": p.theta_a.tolist(), "theta_vertex": p.theta_vertex.tolist(),
        "seed": p.seed, "gap_0": p.gap_0, "t_step": p.t_step, "v_ref_0": p.v_ref_0, "t_mean": p.t_mean,
        "v_ref_sweep": p.v_ref_sweep.tolist(), "h_max": p.h_max, "w_max": p.w_max,
        "rho_sharpness": p.rho_sharpness, "n_list": list(p.n_list)}}

    # --- Setup: delta_bar, initial condition, Assumption on nominal safety
    R["setup"] = {"delta_bar": m.delta_bar(1), "eta_cl": m.eta_cl(1), "h_0": p.h_0(),
                  "H_0": m.H(p.h_0(), p.v_ref_0, p.v_ref_0)}

    # --- Fig. 1(a): margins along the structured run with theta_a
    L = sim.simulate(p.theta_a, "str")
    store(Z, "a", L)
    Z["t"] = L["t"]
    Phi_rows = np.column_stack([L["Phi_11"], L["Phi_12"]])
    Z["a/realized"] = p.tau * np.abs(Phi_rows @ (p.theta_a - p.theta_hat))     # |c Phi delta theta|
    verts = np.array([[s1, s2] for s1 in (1, -1) for s2 in (1, -1)]) * p.dtheta
    Z["a/envelope"] = np.max(p.tau * (Phi_rows @ verts.T), axis=1)              # eq. (envelope)
    active = L["eta_str"] > 1e-9
    R["a"] = {"eta_str_peak": float(L["eta_str"].max()), "eta_str_final": float(L["eta_str"][-1]),
              "envelope_minus_eta_str_max": float(np.max(np.abs(Z["a/envelope"] - L["eta_str"]))),
              "realized_below_margin": bool(np.all(Z["a/realized"][active] < L["eta_str"][active]))}

    # --- Fig. 1(b): H for the vertices and samples of Theta under each margin
    thetas = sim.theta_samples()
    Z["b/theta"] = thetas
    R["b"] = {"n_theta": int(len(thetas))}
    for mg in ("none", "str", "cl"):
        Hs = np.stack([sim.simulate(th, mg)["H"] for th in thetas])
        Z[f"b/{mg}/H"] = Hs
        Hmin = Hs.min(axis=0)
        i = int(np.argmin(Hmin))
        R["b"][mg] = {"min_H": float(Hmin[i]), "t_min": float(L["t"][i]), "H_final": float(Hmin[-1])}
    R["b"]["H_settled_classical"] = m.H_settled_classical()

    # --- Fig. 1(c)-(f): mandatory stop, vertex theta, both priority rules
    R["stop"] = {}
    for pr in ("stop", "safety"):
        for mg in ("none", "str", "cl"):
            S = sim.simulate(p.theta_vertex, mg, stop_line=True, priority=pr)
            store(Z, f"stop/{pr}/{mg}", S)
            win, t1 = window(S)
            R["stop"][f"{pr}/{mg}"] = {
                "min_h": float(S["h"].min()), "min_H": float(S["H"].min()),
                "min_d": float(S["d"].min()), "overrun": float(max(0.0, -S["d"].min())),
                "infeasible_time": win, "first_infeasible_t": t1}

    # --- Fig. 1(g): steady speed vs v_ref
    Z["g/v_ref"] = p.v_ref_sweep
    R["g"] = {"v_floor": m.v_floor()}
    for mg in ("str", "cl"):
        ach = np.array([sim.mean_tail(sim.simulate(p.theta_a, mg, v_ref=float(r))["vA"]) for r in p.v_ref_sweep])
        Z[f"g/{mg}"] = ach
        R["g"][mg] = {"max_abs_error_vs_v_ref": float(np.max(np.abs(ach - p.v_ref_sweep))),
                      "max_abs_error_vs_floor_rule": float(np.max(np.abs(ach - np.maximum(p.v_ref_sweep, m.v_floor()))))}
    print(f"simulations of Fig. 1(a)-(g): {time.time() - t0:.1f} s")

    # --- Fig. 1(i)-(j), figE2: queue of n followers
    R["queue"] = {"n": list(p.n_list), "eta_str_peak": [], "eta_sd_peak": [], "eta_cl": [], "h_min_queue_free": [],
                  "stop": {mg: {"min_H": [], "min_h": [], "infeasible_time": [], "first_infeasible_t": [],
                                "h_min_queue": []} for mg in ("str", "sd", "cl")}}
    for n in p.n_list:
        L = sim.simulate(p.theta_a, "str", n=n)
        R["queue"]["eta_str_peak"].append(float(L["eta_str"].max()))
        R["queue"]["eta_sd_peak"].append(float(L["eta_sd"].max()))
        R["queue"]["eta_cl"].append(float(L["eta_cl"][0]))
        R["queue"]["h_min_queue_free"].append(float(L["h_min_queue"].min()))
        for mg in ("str", "sd", "cl"):
            S = sim.simulate(p.theta_vertex, mg, n=n, stop_line=True)
            win, t1 = window(S)
            q = R["queue"]["stop"][mg]
            q["min_H"].append(float(S["H"].min()))
            q["min_h"].append(float(S["h"].min()))
            q["infeasible_time"].append(win)
            q["first_infeasible_t"].append(t1)
            q["h_min_queue"].append(float(S["h_min_queue"].min()))
    print(f"queue sweep: {time.time() - t0:.1f} s")

    # --- figE3: shared vs independent gains, followers from rest vs at the equilibrium
    R["gains"] = {"n": list(p.n_list_extended), "eta_cl": [float(m.eta_cl(n)) for n in p.n_list_extended],
                  "eta_str_peak": [], "cases": {}}
    for init in ("rest", "equilibrium"):
        for gains in ("shared", "independent"):
            c = {"eta_sd_peak": [], "min_H": [], "infeasible_time": []}
            for n in p.n_list_extended:
                L = sim.simulate(p.theta_a, "str", n=n, init=init, gains=gains)
                S = sim.simulate(p.theta_vertex, "sd", n=n, init=init, gains=gains, stop_line=True)
                c["eta_sd_peak"].append(float(L["eta_sd"].max()))
                c["min_H"].append(float(S["H"].min()))
                c["infeasible_time"].append(window(S)[0])
                if init == "rest" and gains == "shared":
                    R["gains"]["eta_str_peak"].append(float(L["eta_str"].max()))
            R["gains"]["cases"][f"{init}/{gains}"] = c
    print(f"gains sweep: {time.time() - t0:.1f} s")

    # --- Fig. 1(h), figE1: certification sweep
    R["certify"] = certify.certify(step=args.step, verbose=True)
    print(f"certification sweep: {time.time() - t0:.1f} s")

    np.savez_compressed(os.path.join(RESULTS, "runs.npz"), **Z)
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", os.path.join(RESULTS, "runs.npz"), "and", os.path.join(RESULTS, "results.json"),
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
