"""Every constant of Section V (Case study) of the paper, with the paper's symbols.

Units: m, s, m/s, m/s^2.  Nothing in the other modules carries a number of its own.
"""
import numpy as np

# --- follower policy, eq. (ovm): pi_E = theta_1 (V(h) - v_E) + theta_2 (W(v_A) - v_E)
kappa = 0.6                     # slope of the range policy V [1/s]
h_st = 2.0                      # standstill gap [m]
v_max = 12.0                    # speed limit of V and W [m/s]
theta_hat = np.array([0.4, 0.5])            # nominal gains
dtheta = 0.3 * theta_hat                    # half-widths of the box Theta (+-30 %)
L = 5.0                         # vehicle length [m]; positions are rear bumpers, h = s_A - s_E - L

# --- barrier, eq. (car-following_CBF), and alpha(H) = gamma H
d_min = 1.0                     # minimum admissible gap [m]
tau = 1.0                       # time headway [s]
gamma = 0.5                     # alpha(H) = gamma H for H >= 0 [1/s]

# --- leader: input set U, nominal controller u_d = K_p (v_ref - v_A)
u_min, u_max = -3.0, 3.0
K_p = 0.8                       # [1/s]
v_ref_0 = 10.0                  # initial speed of both vehicles and v_ref before the step
t_step = 1.0                    # v_ref steps from v_ref_0 to its final value at this time [s]

# --- simulation: forward Euler
dt = 0.02
T = 40.0
t_show = 25.0                   # time span drawn in Fig. 1(a)-(f)

# --- mandatory stop (Fig. 1(c)-(f)): braking-distance CBF B = (s_line - s_A) - v_A^2 / (2 a_b)
s_line = 45.0                   # stop line [m]; the origin is the leader's rear bumper at t = 0
a_b = 3.0                       # braking deceleration in B [m/s^2]
gamma_s = 1.0                   # dot B >= -gamma_s B [1/s]
v_B_min = 0.3                   # u_B = a_b (gamma_s B - v_A) / max(v_A, v_B_min)

# --- parameters of the runs in Fig. 1
theta_a = theta_hat + np.array([0.6, -0.4]) * dtheta      # panels (a), (g), (i): (0.472, 0.440)
theta_vertex = theta_hat - dtheta                         # panels (c)-(f), (j): (0.28, 0.35)
n_samples = 20                  # panel (b): uniform samples in Theta, drawn after the 4 vertices
seed = 0                        # of the sample draw (numpy default_rng)
v_ref_sweep = np.arange(0.5, 4.0 + 1e-9, 0.25)            # panel (g) [m/s]
t_mean = 5.0                    # panel (g): v* is the mean of v_A over the last t_mean seconds

# --- queue of n followers (Fig. 1(i)-(j)): E_2..E_n start from rest, gap_0 apart
n_list = tuple(range(1, 11))
gap_0 = 40.0                    # [m]

# --- operating set X and certification sets Omega_eps = {x in X : H(x) >= -eps}
h_max = 30.0                    # X: h in [d_min, h_max]
w_max = 5.0                     # X: v_E - v_A <= w_max
eps_max = 5.0                   # Omega_eps grows from S (eps = 0) to X (eps = eps_max)
eps_step = 0.05
grid_step = 0.05                # spacing of the grid on X [m and m/s]
gamma_minus = (0.5, 0.3, 0.2, 0.1)          # alpha(H) = gamma_minus H for H < 0, Fig. 1(h)
gamma_minus_drawn = (0.5, 0.3, 0.1)
rho_bracket = (0.0, 4.0)        # bisection for rho*_S (Remark on computation)
rho_tol = 1e-3
rho_sharpness = 1.2             # Corollary on feasibility: rho = 1.2 rho*_S at a state on the boundary of S

# --- extended figures (not in the paper; see ReadMe)
gamma_single = (0.5, 0.4, 0.3, 0.25, 0.2, 0.1)            # figE1(b): alpha(H) = gamma H on all of X
gamma_single_drawn = (0.5, 0.3, 0.25, 0.2, 0.1)
n_list_extended = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30)   # figE3


def h_0():
    """Initial gap: the follower at the equilibrium of eq. (ovm) at v_ref_0, h_0 = h_st + v_ref_0 / kappa."""
    return h_st + v_ref_0 / kappa
