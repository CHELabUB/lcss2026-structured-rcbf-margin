"""Model-level checks at hand-computed states: the follower policy, barrier, margins, bounds and
filters of Section V (numpy only, seconds)."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import params as p                          # noqa: E402
import model as m                           # noqa: E402
import simulate as sim                      # noqa: E402
import manuscript as M                      # noqa: E402
from manuscript import close_to_printed     # noqa: E402


class TestPolicy(unittest.TestCase):
    def test_range_policy_V(self):
        self.assertEqual(m.V(p.h_st), 0.0)
        self.assertEqual(m.V(0.0), 0.0)
        self.assertAlmostEqual(m.V(p.h_st + 1.0), p.kappa, places=12)
        self.assertEqual(m.V(1e3), p.v_max)
        self.assertAlmostEqual(m.V(p.h_st + p.v_max / p.kappa), p.v_max, places=12)

    def test_speed_policy_W(self):
        self.assertEqual(m.W(5.0), 5.0)
        self.assertEqual(m.W(20.0), p.v_max)

    def test_equilibrium_has_zero_acceleration(self):
        for v in (0.0, 3.0, 10.0, 12.0):
            self.assertAlmostEqual(m.pi_E(p.h_st + v / p.kappa, v, v, p.theta_hat), 0.0, places=12)

    def test_hand_values(self):
        h10 = p.h_0()                                               # equilibrium gap at 10 m/s
        self.assertAlmostEqual(m.pi_E(h10, 10.0, 8.0, p.theta_hat), -1.0, places=12)     # leader slower
        self.assertAlmostEqual(m.pi_E(h10 + 2.0, 10.0, 10.0, p.theta_hat), 0.4 * 1.2, places=12)
        self.assertAlmostEqual(m.pi_E(h10 + 5.0, 10.0, 10.0, p.theta_hat), 0.4 * 2.0, places=12)  # V saturates
        self.assertAlmostEqual(m.pi_E(p.h_st, 10.0, 10.0, p.theta_hat), -4.0, places=12)

    def test_regressor_rows(self):
        Phi = m.Phi([p.h_0(), 40.0], [10.0, 0.0], [10.0, 10.0])      # n = 2, E_2 at rest 40 m behind
        np.testing.assert_allclose(Phi[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(Phi[1], [p.v_max, 10.0])


class TestParameters(unittest.TestCase):
    def test_box(self):
        np.testing.assert_allclose(p.theta_hat, [0.4, 0.5])
        np.testing.assert_allclose(p.dtheta, [0.12, 0.15])
        np.testing.assert_allclose(p.theta_a, M.THETA_A)
        np.testing.assert_allclose(p.theta_vertex, M.THETA_VERTEX)

    def test_uniform_bound_and_classical_margin(self):
        self.assertAlmostEqual(m.delta_bar(1), p.v_max * p.dtheta.sum(), places=12)
        self.assertTrue(close_to_printed(m.delta_bar(1), M.DELTA_BAR))
        self.assertTrue(close_to_printed(m.eta_cl(1), M.DELTA_BAR))
        self.assertTrue(close_to_printed(m.H_settled_classical(), M.H_SETTLED_CLASSICAL))
        for n in (1, 2, 5, 10):
            self.assertAlmostEqual(m.eta_cl(n), p.tau * np.sqrt(n) * p.v_max * p.dtheta.sum(), places=12)

    def test_speed_floor(self):
        self.assertAlmostEqual(m.v_floor(), p.kappa * (p.tau * m.delta_bar(1) / p.gamma + p.d_min - p.h_st), places=12)
        self.assertTrue(close_to_printed(m.v_floor(), M.V_FLOOR))

    def test_initial_condition(self):
        log = sim.simulate(p.theta_hat, "str", T=1.0)
        self.assertTrue(close_to_printed(log["h"][0], M.H0_GAP))
        self.assertTrue(close_to_printed(log["H"][0], M.H0_BARRIER))
        self.assertEqual((log["vA"][0], log["vE"][0]), (p.v_ref_0, p.v_ref_0))
        self.assertEqual(log["d"][0], p.s_line)                     # the origin is the leader's rear at t = 0

    def test_vehicle_length_cancels(self):
        """The gaps, hence every result, do not depend on L (positions shift by a constant)."""
        L0 = p.L
        try:
            ref = sim.simulate(p.theta_vertex, "str", n=3, stop_line=True, T=5.0)
            p.L = 0.0
            alt = sim.simulate(p.theta_vertex, "str", n=3, stop_line=True, T=5.0)
        finally:
            p.L = L0
        for k in ("h", "H", "u", "eta_sd", "h_min_queue", "d"):
            np.testing.assert_allclose(ref[k], alt[k], atol=1e-10, err_msg=k)


class TestMargins(unittest.TestCase):
    def test_structured_margin_closed_form_single_row(self):
        for P1, P2 in ((1.0, -2.0), (0.0, 0.0), (-3.5, 0.25)):
            Phi = np.array([[P1, P2]])
            e_str = m.eta_str(Phi)
            self.assertAlmostEqual(e_str, p.tau * (abs(P1) * p.dtheta[0] + abs(P2) * p.dtheta[1]), places=12)
            self.assertAlmostEqual(m.eta_sd(Phi), e_str, places=12)          # r = 1: psi = |c| varphi
            self.assertEqual(m.margin("none", Phi), 0.0)
            self.assertEqual(m.margin("cl", Phi), m.eta_cl(1))

    def test_ordering_eta_str_le_eta_sd_le_eta_cl(self):
        rng = np.random.default_rng(1)
        for n in (1, 2, 3, 5, 10):
            for _ in range(50):
                Phi = rng.uniform(-p.v_max, p.v_max, size=(n, 2))          # |Phi_ik| <= v_max on X
                e_str, e_sd, e_cl = m.eta_str(Phi), m.eta_sd(Phi), m.eta_cl(n)
                self.assertLessEqual(e_str, e_sd + 1e-12)
                self.assertLessEqual(e_sd, e_cl + 1e-12)
                self.assertLessEqual(e_sd, m.eta_sd(Phi, "independent") + 1e-12)   # column 2-norm <= 1-norm

    def test_eta_sd_minus_eta_str_bounded_by_other_rows(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            Phi = rng.uniform(-p.v_max, p.v_max, size=(3, 2))
            other = p.tau * (np.abs(Phi[1:, 0]).sum() * p.dtheta[0] + np.abs(Phi[1:, 1]).sum() * p.dtheta[1])
            self.assertLessEqual(m.eta_sd(Phi) - m.eta_str(Phi), other + 1e-12)
        Phi = np.array([[1.0, -2.0], [0.0, 0.0]])                          # rows 2.. vanish -> equality
        self.assertAlmostEqual(m.eta_sd(Phi), m.eta_str(Phi), places=12)

    def test_residual_terms_at_quoted_states(self):
        H, sup, psi_d = m.residual_terms(*(np.array([v]) for v in M.SHARPNESS_STATE))
        self.assertEqual(H[0], 0.0)
        self.assertAlmostEqual(sup[0], 1.54, places=12)                     # D((6,0,5); 0) = 1.540
        self.assertAlmostEqual(sup[0] - 1.2 * 1.45 * psi_d[0], -0.31, delta=0.005)
        H, sup, psi_d = m.residual_terms(*(np.array([v]) for v in M.ASSUMPTION_STATE))
        self.assertEqual(H[0], -5.0)
        self.assertAlmostEqual(sup[0] + p.gamma * H[0], 0.0, places=12)   # Assumption tight at the corner


class TestFilter(unittest.TestCase):
    def test_u_H_satisfies_the_robust_cbf_condition_with_equality(self):
        """a(x, u_H) + gamma H - eta = 0 with the selected margin, eq. (inst-ac)."""
        for mg in ("none", "str", "cl"):
            log = sim.simulate(p.theta_vertex, mg, stop_line=True, T=10.0)
            eta = {"none": 0.0, "str": log["eta_str"], "cl": log["eta_cl"]}[mg]
            Phi_1 = np.column_stack([log["Phi_11"], log["Phi_12"]])
            a = (log["vA"] - log["vE"]) + p.tau * log["u_H"] - p.tau * (Phi_1 @ p.theta_hat)
            np.testing.assert_allclose(a + p.gamma * log["H"] - eta, 0.0, atol=1e-9)

    def test_stop_line_bound(self):
        self.assertAlmostEqual(m.B(0.0, 10.0), 45.0 - 100.0 / 6.0, places=12)
        self.assertAlmostEqual(m.u_B(0.0, 10.0), 3.0 * (m.B(0.0, 10.0) - 10.0) / 10.0, places=12)   # 5.5
        self.assertAlmostEqual(m.u_B(0.0, 0.0), 3.0 * 45.0 / 0.3, places=12)                        # floor 0.3

    def test_priority_rules(self):
        self.assertEqual(m.apply_filter(0.0, 2.0, 1.0, "stop"), 1.0)      # min(max(u_d, u_H), u_B)
        self.assertEqual(m.apply_filter(0.0, 2.0, 1.0, "safety"), 2.0)    # max(min(u_d, u_B), u_H)
        self.assertEqual(m.apply_filter(-1.0, -2.0, None), -1.0)          # no stop line
        self.assertEqual(m.apply_filter(5.0, -10.0, None), 3.0)           # projection onto U
        self.assertEqual(m.apply_filter(0.0, 2.0, -10.0, "stop"), -3.0)

    def test_infeasibility_flag(self):
        log = sim.simulate(p.theta_vertex, "cl", stop_line=True, T=10.0)
        flag = (log["u_H"] > np.minimum(log["u_B"], p.u_max) + 1e-9).astype(float)
        np.testing.assert_array_equal(log["infeasible"], flag)
        self.assertGreater(flag.sum(), 0)

    def test_speeds_never_negative_and_queue_is_one_directional(self):
        log = sim.simulate(p.theta_vertex, "sd", n=3, stop_line=True)
        self.assertGreaterEqual(log["vA"].min(), 0.0)
        self.assertGreaterEqual(log["vE"].min(), 0.0)
        ref = sim.simulate(p.theta_vertex, "str", n=1, stop_line=True)
        two = sim.simulate(p.theta_vertex, "str", n=4, stop_line=True)    # the (A, E_1) pair ignores E_2..E_n
        for k in ("h", "H", "vA", "vE", "u"):
            np.testing.assert_allclose(two[k], ref[k], atol=1e-12, err_msg=k)


if __name__ == "__main__":
    unittest.main()
