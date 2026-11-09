"""Every number printed in Section V of the paper, as the string that appears in the text.

``close_to_printed`` passes exactly when the computed value rounds to the printed one (tolerance:
half a unit in the last printed digit).  The last block holds the numbers of the extended figures
(ReadMe), which are not in the paper.
"""

# --- Setup
K_P = 0.8                           # 1/s, u_d = K_p (v_ref - v_A)
V_REF_0 = 10.0                      # m/s, both vehicles; v_ref steps to 0 at T_STEP
T_STEP = 1.0                        # s
DT, T_SIM = 0.02, 40.0              # s, forward Euler step and horizon
N_THETA = 24                        # 4 vertices + 20 uniform samples in Fig. 1(b)
V_REF_SWEEP = (0.5, 4.0, 0.25)      # m/s, Fig. 1(g): first, last, spacing
T_MEAN = 5.0                        # s, v* is the mean of v_A over the last T_MEAN seconds
X_H_MAX, X_W_MAX = 30.0, 5.0        # m, m/s: X = {h in [d_min, 30], v in [0, 12]^2, v_E - v_A <= 5}
DELTA_BAR = "3.24"                  # m/s^2, uniform bound; eta_cl = tau delta_bar = 3.24 m/s
H0_GAP = "18.67"                    # m, h(0) = h_st + v_ref_0 / kappa
H0_BARRIER = "17.67"                # m, H(0)
THETA_A = (0.472, 0.440)            # panels (a), (g), (i): theta_hat + (0.6 dtheta_1, -0.4 dtheta_2)
THETA_VERTEX = (0.28, 0.35)         # panels (c)-(f), (j): theta_hat - dtheta
ASSUMPTION_STATE = (1.0, 0.0, 5.0)  # sup_u a + gamma H = 0 on X only at this (h, v_A, v_E), where H = -5 m
ASSUMPTION_H = "-5"
# --- Gap and minimality, Fig. 1(a)
ETA_STR_PEAK = "0.65"               # m/s
# --- Invariance, Fig. 1(b)
MIN_H_STRUCTURED = "8e-8"           # m, min over theta and t
MIN_H_NO_MARGIN = "-0.24"           # m
MIN_H_CLASSICAL = "6.28"            # m, min over theta and t (the dip)
H_SETTLED_CLASSICAL = "6.48"        # m, H settles at tau delta_bar / gamma
# --- Mandatory stop, Fig. 1(c)-(f), priority to the stop
STOP_MIN_H_CLASSICAL = "-0.45"      # m, min_t h: rear-end collision
STOP_MIN_H_NO_MARGIN = "0.82"       # m
STOP_MARGIN_STRUCTURED = "7.1"      # m, distance to the line at standstill
STOP_MIN_BARRIER_STRUCTURED = 0.0   # min_t H, to numerical precision
STOP_MIN_BARRIER_ABS_TOL = 1e-6
# --- opposite priority rule (safety first)
SAFETY_MIN_H_STRUCTURED = "1.00"    # m
SAFETY_MIN_H_CLASSICAL = "7.31"     # m
SAFETY_OVERRUN_CLASSICAL = "110"    # m past the stop line
# --- Certifiable equilibria, Fig. 1(g)
V_FLOOR = "3.29"                    # m/s
# --- Feasibility and certifiable uncertainty, Fig. 1(h)
RHO_STAR_S = "1.45"
BUDGET_USED_PERCENT = "69"          # 1 / rho*_S
SHARPNESS_RHO_FACTOR = 1.2          # rho = 1.2 rho*_S ...
SHARPNESS_STATE = (6.0, 0.0, 5.0)   # ... at this state on H = 0
SHARPNESS_SUP_HDOT = "-0.31"        # max_u dot H < 0 there, at the adversarial vertex
RHO_STAR_X_GAMMA_MINUS_05 = "0"     # rho*_Omega at eps = 5 m with gamma_- = 0.5
EPS_MAX_GAMMA_MINUS_05 = "1.4"      # m, largest eps with rho*_Omega >= 1 for gamma_- = 0.5
EPS_MAX_GAMMA_MINUS_03 = "3.6"      # m, the same for gamma_- = 0.3
GAMMA_MINUS_THRESHOLD = "0.108"     # 1/s, gamma_- <= this restores rho*_X = rho*_S
# --- Several followers, Fig. 1(i)-(j)
QUEUE_GAP_0 = 40.0                  # m, E_2..E_n start from rest this far apart
ETA_SD_PEAK_N2 = "2.94"             # m/s  ("grows from 2.94 at n = 2 to 5.82 at n = 10")
ETA_SD_PEAK_N10 = "5.82"
ETA_CL_N1 = "3.24"                  # m/s  ("grows from 3.24 to 10.25")
ETA_CL_N10 = "10.25"
SD_INFEASIBLE_FROM_N = 2            # the eta_sd filter is infeasible for n >= 2
SD_GUARANTEE_LOST_FROM_N = 5        # ... and drives H < 0 for n >= 5

# --- Extended figures (ReadMe; not in the paper) -------------------------------------------
# figE1(b): a single slope gamma on all of X
SINGLE_RHO_STAR_FLAT = {"0.25": "1.07", "0.2": "0.57", "0.1": "0"}   # rho*_Omega, flat in eps
SINGLE_BINDING_STATE_025 = (22.0, 0.0, 5.0)                          # binding state for gamma <= 0.25
SINGLE_BINDING_H_025 = "16"                                          # m
SINGLE_MAX_RHO_STAR_X = "0.93"                                       # largest rho*_X over the single slopes
# figE2: infeasible time of the forced stop vs n
SD_INFEASIBLE_WINDOW_N2 = "3.9"     # s
SD_INFEASIBLE_FIRST_N2 = "13.1"     # s
SD_INFEASIBLE_WINDOW_N10 = "31.2"   # s
SD_INFEASIBLE_FIRST_N10 = "8.8"     # s
SD_WINDOW_TO_END_FROM_N = 5         # the eta_sd window reaches the end of the run from this n on
CL_INFEASIBLE_FIRST_N1 = "5.7"      # s
CL_INFEASIBLE_FIRST_N10 = "1.2"     # s
CL_INFEASIBLE_WINDOW_N1 = "34"      # s, to the end of the run
CL_INFEASIBLE_WINDOW_N10 = "39"     # s
# figE3: independent gains, followers from rest
INDEP_ETA_SD_PEAK_N2 = "2.94"       # m/s
INDEP_ETA_SD_PEAK_N10 = "14.46"     # m/s
INDEP_SD_EXCEEDS_CL_FROM_N = 5          # 7.26 > 7.24 m/s at n = 5
INDEP_SD_INFEASIBLE_WINDOW_N2 = "10.4"   # s
INDEP_SD_INFEASIBLE_WINDOW_N3 = "31.6"   # s
INDEP_SD_WHOLE_RUN_FROM_N = 10
INDEP_MIN_H_N3 = "-0.45"            # m
INDEP_MIN_H_N10 = "-3.0"            # m, for n >= 10


def printed_tolerance(printed):
    """Half a unit in the last place of a printed decimal or one-digit scientific value; a printed
    "0" means exactly zero (tolerance 1e-6)."""
    s = printed.strip().lower()
    if float(s) == 0.0:
        return 1e-6
    if "e" in s:
        mant, exp = s.split("e")
        digits = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10.0 ** (int(exp) - digits)
    digits = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10.0 ** (-digits)


def close_to_printed(value, printed, slack=1e-9):
    """True iff ``value`` rounds to the printed string (plus a float-noise slack)."""
    return abs(float(value) - float(printed)) <= printed_tolerance(printed) + slack
