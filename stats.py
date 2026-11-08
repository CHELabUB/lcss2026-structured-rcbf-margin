#!/usr/bin/env python3
"""Print every number quoted in Section V (and in the ReadMe for the extended figures) from
results/results.json, next to the printed value, with PASS / FAIL.

Usage: python stats.py        (after generate_results.py)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests"))
import manuscript as M                      # noqa: E402
from manuscript import close_to_printed     # noqa: E402


def load(path=os.path.join(HERE, "results", "results.json")):
    with open(path) as f:
        return json.load(f)


def rows(R):
    """(label, computed value, printed string or None, pass) for the table."""
    c, q, g, s = R["certify"], R["queue"], R["gains"]["cases"]["rest/independent"], R["stop"]
    n = q["n"]
    i2, i10 = n.index(2), n.index(10)
    one, single = c["one_sided"], c["single"]
    first_n = lambda vals, pred: next((nn for nn, v in zip(n, vals) if pred(v)), None)
    ends = lambda t1, w: t1 is not None and abs(t1 + w - R["params"]["T"]) < 1.5 * R["params"]["dt"]
    sd = q["stop"]["sd"]
    cl = q["stop"]["cl"]
    ge = R["gains"]["n"]
    out = [
        ("Setup", None, None),
        ("delta_bar [m/s^2] = eta_cl [m/s]", R["setup"]["delta_bar"], M.DELTA_BAR),
        ("h(0) [m]", R["setup"]["h_0"], M.H0_GAP),
        ("H(0) [m]", R["setup"]["H_0"], M.H0_BARRIER),
        ("min_X (sup_u a + gamma H)", c["min_D0_X"]["value"], "0"),
        ("  attained at (h, v_A, v_E)", tuple(c["min_D0_X"]["state"]), M.ASSUMPTION_STATE),
        ("Fig. 1(a): gap and minimality", None, None),
        ("peak eta_str [m/s]", R["a"]["eta_str_peak"], M.ETA_STR_PEAK),
        ("Fig. 1(b): invariance", None, None),
        ("min_{theta,t} H, structured [m]", R["b"]["str"]["min_H"], M.MIN_H_STRUCTURED),
        ("min_{theta,t} H, no margin [m]", R["b"]["none"]["min_H"], M.MIN_H_NO_MARGIN),
        ("min_{theta,t} H, classical [m]", R["b"]["cl"]["min_H"], M.MIN_H_CLASSICAL),
        ("H settles at tau delta_bar / gamma [m]", R["b"]["cl"]["H_final"], M.H_SETTLED_CLASSICAL),
        ("Fig. 1(c)-(f): mandatory stop, stop prioritized", None, None),
        ("min_t h, classical [m]", s["stop/cl"]["min_h"], M.STOP_MIN_H_CLASSICAL),
        ("min_t h, no margin [m]", s["stop/none"]["min_h"], M.STOP_MIN_H_NO_MARGIN),
        ("stop margin, structured [m]", s["stop/str"]["min_d"], M.STOP_MARGIN_STRUCTURED),
        ("min_t H, structured [m] (|.| <= 1e-6)", s["stop/str"]["min_H"],
         abs(s["stop/str"]["min_H"]) <= M.STOP_MIN_BARRIER_ABS_TOL),
        ("infeasible time, classical [s]", s["stop/cl"]["infeasible_time"], s["stop/cl"]["infeasible_time"] > 0),
        ("infeasible time, structured [s]", s["stop/str"]["infeasible_time"], s["stop/str"]["infeasible_time"] == 0),
        ("safety prioritized", None, None),
        ("min_t h, structured [m]", s["safety/str"]["min_h"], M.SAFETY_MIN_H_STRUCTURED),
        ("min_t h, classical [m]", s["safety/cl"]["min_h"], M.SAFETY_MIN_H_CLASSICAL),
        ("overrun of the line, classical [m]", s["safety/cl"]["overrun"], M.SAFETY_OVERRUN_CLASSICAL),
        ("Fig. 1(g): certifiable equilibria", None, None),
        ("v_floor [m/s]", R["g"]["v_floor"], M.V_FLOOR),
        ("max |v* - v_ref|, structured", R["g"]["str"]["max_abs_error_vs_v_ref"], R["g"]["str"]["max_abs_error_vs_v_ref"] < 1e-6),
        ("max |v* - max(v_ref, v_floor)|, classical", R["g"]["cl"]["max_abs_error_vs_floor_rule"],
         R["g"]["cl"]["max_abs_error_vs_floor_rule"] < 1e-4),
        ("Fig. 1(h): certifiable uncertainty", None, None),
        ("rho*_S (ratio infimum)", c["rho_star_S"]["ratio_infimum"], M.RHO_STAR_S),
        ("rho*_S (bisection)", c["rho_star_S"]["bisection"], M.RHO_STAR_S),
        ("1 / rho*_S [%]", c["rho_star_S"]["budget_used_percent"], M.BUDGET_USED_PERCENT),
        ("sharpness state (h, v_A, v_E) on H = 0", tuple(c["sharpness"]["state"]), M.SHARPNESS_STATE),
        ("max_u dot H at 1.2 rho*_S there", c["sharpness"]["sup_Hdot"], M.SHARPNESS_SUP_HDOT),
        ("rho*_Omega at eps = 5, gamma_- = 0.5", one["0.5"]["rho_star"][-1], M.RHO_STAR_X_GAMMA_MINUS_05),
        ("eps_max with rho*_Omega >= 1, gamma_- = 0.5 [m]", one["0.5"]["eps_max"], M.EPS_MAX_GAMMA_MINUS_05),
        ("eps_max with rho*_Omega >= 1, gamma_- = 0.3 [m]", one["0.3"]["eps_max"], M.EPS_MAX_GAMMA_MINUS_03),
        ("gamma_- threshold restoring rho*_X = rho*_S [1/s]", c["gamma_minus_threshold"]["value"], M.GAMMA_MINUS_THRESHOLD),
        ("Fig. 1(i)-(j): several followers", None, None),
        ("peak eta_sd, n = 2 [m/s]", q["eta_sd_peak"][i2], M.ETA_SD_PEAK_N2),
        ("peak eta_sd, n = 10 [m/s]", q["eta_sd_peak"][i10], M.ETA_SD_PEAK_N10),
        ("eta_cl, n = 1 [m/s]", q["eta_cl"][0], M.ETA_CL_N1),
        ("eta_cl, n = 10 [m/s]", q["eta_cl"][i10], M.ETA_CL_N10),
        ("peak eta_str invariant in n", max(q["eta_str_peak"]) - min(q["eta_str_peak"]),
         max(q["eta_str_peak"]) - min(q["eta_str_peak"]) < 1e-12),
        ("eta_sd filter infeasible from n =", first_n(sd["infeasible_time"], lambda w: w > 0), M.SD_INFEASIBLE_FROM_N),
        ("eta_sd filter: H < 0 from n =", first_n(sd["min_H"], lambda v: v < 0), M.SD_GUARANTEE_LOST_FROM_N),
        ("structured: min_n min_t H", min(q["stop"]["str"]["min_H"]), min(q["stop"]["str"]["min_H"]) >= 0),
        ("classical: max_n min_t h (rear-ends for every n)", max(cl["min_h"]), max(cl["min_h"]) < 0),
        ("Extended figure E1 (single slope gamma on all of X)", None, None),
        ("rho*_Omega, gamma = 0.25 (flat)", single["0.25"]["rho_star"][0], M.SINGLE_RHO_STAR_FLAT["0.25"]),
        ("rho*_Omega, gamma = 0.2 (flat)", single["0.2"]["rho_star"][0], M.SINGLE_RHO_STAR_FLAT["0.2"]),
        ("rho*_Omega, gamma = 0.1", single["0.1"]["rho_star"][0], M.SINGLE_RHO_STAR_FLAT["0.1"]),
        ("binding state, gamma = 0.25", tuple(single["0.25"]["binding_state"][0]), M.SINGLE_BINDING_STATE_025),
        ("  its H [m]", single["0.25"]["binding_H"][0], M.SINGLE_BINDING_H_025),
        ("largest rho*_X over the single slopes", max(v["rho_star"][-1] for v in single.values()), M.SINGLE_MAX_RHO_STAR_X),
        ("Extended figure E2 (infeasible time vs n)", None, None),
        ("eta_sd: infeasible time, n = 2 [s]", sd["infeasible_time"][i2], M.SD_INFEASIBLE_WINDOW_N2),
        ("eta_sd: first infeasible t, n = 2 [s]", sd["first_infeasible_t"][i2], M.SD_INFEASIBLE_FIRST_N2),
        ("eta_sd: infeasible time, n = 10 [s]", sd["infeasible_time"][i10], M.SD_INFEASIBLE_WINDOW_N10),
        ("eta_sd: first infeasible t, n = 10 [s]", sd["first_infeasible_t"][i10], M.SD_INFEASIBLE_FIRST_N10),
        ("eta_sd: window reaches the end of the run from n =",
         first_n(list(zip(sd["first_infeasible_t"], sd["infeasible_time"])), lambda tw: ends(*tw)), M.SD_WINDOW_TO_END_FROM_N),
        ("eta_cl: first infeasible t, n = 1 [s]", cl["first_infeasible_t"][0], M.CL_INFEASIBLE_FIRST_N1),
        ("eta_cl: first infeasible t, n = 10 [s]", cl["first_infeasible_t"][i10], M.CL_INFEASIBLE_FIRST_N10),
        ("eta_cl: infeasible time, n = 1 [s]", cl["infeasible_time"][0], M.CL_INFEASIBLE_WINDOW_N1),
        ("eta_cl: infeasible time, n = 10 [s]", cl["infeasible_time"][i10], M.CL_INFEASIBLE_WINDOW_N10),
        ("eta_str: infeasible time, every n [s]", max(q["stop"]["str"]["infeasible_time"]),
         max(q["stop"]["str"]["infeasible_time"]) == 0),
        ("Extended figure E3 (independent gains, followers from rest)", None, None),
        ("peak eta_sd, n = 2 [m/s]", g["eta_sd_peak"][ge.index(2)], M.INDEP_ETA_SD_PEAK_N2),
        ("peak eta_sd, n = 10 [m/s]", g["eta_sd_peak"][ge.index(10)], M.INDEP_ETA_SD_PEAK_N10),
        ("eta_sd exceeds eta_cl from n =",
         next((nn for nn, a, b in zip(ge, g["eta_sd_peak"], R["gains"]["eta_cl"]) if a > b), None), M.INDEP_SD_EXCEEDS_CL_FROM_N),
        ("eta_sd: infeasible time, n = 2 [s]", g["infeasible_time"][ge.index(2)], M.INDEP_SD_INFEASIBLE_WINDOW_N2),
        ("eta_sd: infeasible time, n = 3 [s]", g["infeasible_time"][ge.index(3)], M.INDEP_SD_INFEASIBLE_WINDOW_N3),
        ("eta_sd: infeasible for the whole run from n =",
         next((nn for nn, w in zip(ge, g["infeasible_time"]) if w >= R["params"]["T"] - 1e-9), None), M.INDEP_SD_WHOLE_RUN_FROM_N),
        ("min_t H, n = 3 [m]", g["min_H"][ge.index(3)], M.INDEP_MIN_H_N3),
        ("min_t H, n = 10 [m]", g["min_H"][ge.index(10)], M.INDEP_MIN_H_N10),
    ]
    table = []
    for label, value, printed in out:
        if value is None and printed is None:
            table.append((label, None, None, None))
            continue
        if isinstance(printed, bool):
            table.append((label, value, "", printed))
        elif isinstance(printed, tuple):
            table.append((label, value, str(printed), bool(np.allclose(value, printed))))
        elif isinstance(printed, int):
            table.append((label, value, str(printed), value == printed))
        else:
            table.append((label, value, printed, value is not None and close_to_printed(value, printed)))
    return table


def fmt(v):
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:g}" for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main():
    R = load()
    table = rows(R)
    n_fail = 0
    print(f"{'quantity':58s} {'computed':>16s}  {'paper':>8s}  result")
    for label, value, printed, ok in table:
        if value is None and printed is None:
            print(f"\n--- {label}")
            continue
        n_fail += not ok
        print(f"{label:58s} {fmt(value):>16s}  {printed:>8s}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{sum(1 for r in table if r[3] is not None)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
