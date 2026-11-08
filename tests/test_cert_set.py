"""Certifiable uncertainty on the certification sets (certify.py), recomputed on a 0.1 m grid
(seconds): rho*_S, its bisection bracket and Lipschitz certificate, the eps sweep of Fig. 1(h),
the gamma_- threshold, and the single-slope sweep of figE1.  Every quoted number of the paragraph
"Feasibility and certifiable uncertainty" is reproduced on this coarse grid already."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import params as p                          # noqa: E402
import model as m                           # noqa: E402
import certify                              # noqa: E402
import manuscript as M                      # noqa: E402
from manuscript import close_to_printed     # noqa: E402

STEP = 0.1


class TestCertifyCoarse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = certify.certify(step=STEP, eps_step=STEP)
        cls.one = cls.R["one_sided"]
        cls.single = cls.R["single"]

    def test_rho_star_S(self):
        rs = self.R["rho_star_S"]
        self.assertTrue(close_to_printed(rs["ratio_infimum"], M.RHO_STAR_S))
        self.assertTrue(close_to_printed(rs["bisection"], M.RHO_STAR_S))
        self.assertTrue(rs["bisection_brackets_ratio_infimum"])
        self.assertLessEqual(rs["bracket"][1] - rs["bracket"][0], p.rho_tol)
        self.assertTrue(close_to_printed(rs["budget_used_percent"], M.BUDGET_USED_PERCENT))
        self.assertEqual(rs["binding_state"], list(M.SHARPNESS_STATE))
        self.assertLess(rs["certified_lower_bound"], rs["ratio_infimum"])      # a lower bound ...
        self.assertGreater(rs["certified_lower_bound"], 1.0)                    # ... that still certifies the box
        for g in self.one:                                                      # eps = 0 is S for every gamma_-
            self.assertAlmostEqual(self.one[g]["rho_star"][0], rs["ratio_infimum"], places=12)

    def test_assumption_on_nominal_safety(self):
        self.assertAlmostEqual(self.R["min_D0_X"]["value"], 0.0, places=12)
        self.assertEqual(self.R["min_D0_X"]["state"], list(M.ASSUMPTION_STATE))
        self.assertTrue(close_to_printed(m.H(*M.ASSUMPTION_STATE), M.ASSUMPTION_H))

    def test_sharpness(self):
        s = self.R["sharpness"]
        self.assertEqual(s["state"], list(M.SHARPNESS_STATE))
        self.assertEqual(s["H"], 0.0)
        self.assertTrue(close_to_printed(s["sup_Hdot"], M.SHARPNESS_SUP_HDOT))
        self.assertLess(s["sup_Hdot"], 0.0)

    def test_sweep_monotone_and_eps_max(self):
        eps = np.array(self.R["eps"])
        for g, rows in self.one.items():
            self.assertTrue(rows["monotone_nonincreasing"], g)
            r = np.array(rows["rho_star"])
            i = int(np.argmin(np.abs(eps - rows["eps_max"])))
            self.assertGreaterEqual(r[i], 1.0)
            if i + 1 < len(r):
                self.assertLess(r[i + 1], 1.0)
        self.assertTrue(close_to_printed(self.one["0.5"]["eps_max"], M.EPS_MAX_GAMMA_MINUS_05))
        self.assertTrue(close_to_printed(self.one["0.3"]["eps_max"], M.EPS_MAX_GAMMA_MINUS_03))
        self.assertTrue(close_to_printed(self.one["0.5"]["rho_star"][-1], M.RHO_STAR_X_GAMMA_MINUS_05))
        self.assertAlmostEqual(self.one["0.1"]["rho_star"][-1], self.R["rho_star_S"]["ratio_infimum"], places=12)

    def test_certified_lower_bounds(self):
        for g, rows in self.one.items():
            r, c = np.array(rows["rho_star"]), np.array(rows["certified_lower_bound"])
            self.assertTrue(np.all(c <= r + 1e-12), g)
            self.assertTrue(np.all(np.diff(c) <= 1e-12), g)

    def test_gamma_minus_threshold(self):
        thr = self.R["gamma_minus_threshold"]
        rho_S = self.R["rho_star_S"]["ratio_infimum"]
        self.assertTrue(close_to_printed(thr["value"], M.GAMMA_MINUS_THRESHOLD))
        self.assertAlmostEqual(thr["value"], (2.5 - 1.35 * rho_S) / 5.0, places=12)
        self.assertEqual(thr["corner_state"], list(M.ASSUMPTION_STATE))
        rx = thr["rho_star_X"]
        self.assertAlmostEqual(rx["0.108"], rho_S, places=12)
        self.assertAlmostEqual(rx["0.1"], rho_S, places=12)
        self.assertLess(rx["0.11"], rho_S)
        self.assertEqual(rx["0.5"], 0.0)

    def test_single_slope_sweep(self):
        for g in ("0.5", "0.3"):                       # same binding state (on H = 0 or in X \ S) as one-sided
            np.testing.assert_allclose(self.single[g]["rho_star"], self.one[g]["rho_star"], atol=1e-12)
        for g, printed in M.SINGLE_RHO_STAR_FLAT.items():
            r = np.array(self.single[g]["rho_star"])
            self.assertTrue(close_to_printed(r[0], printed), g)
            # flat in eps: the binding state lies inside S (for gamma = 0.25 until the corner of X binds near eps = 5)
            flat_until = r.size if g != "0.25" else int(np.argmin(np.abs(np.array(self.R["eps"]) - 4.0)))
            np.testing.assert_allclose(r[:flat_until], r[0], atol=1e-12, err_msg=g)
        self.assertEqual(self.single["0.25"]["binding_state"][0], list(M.SINGLE_BINDING_STATE_025))
        self.assertTrue(close_to_printed(self.single["0.25"]["binding_H"][0], M.SINGLE_BINDING_H_025))
        self.assertTrue(close_to_printed(max(v["rho_star"][-1] for v in self.single.values()), M.SINGLE_MAX_RHO_STAR_X))

    def test_lipschitz_constants(self):
        np.testing.assert_allclose(m.lipschitz_constants(0.0), (0.74, 2.0, 2.4))
        s0, s1 = certify.lipschitz_slack(0.05, 0.05)
        self.assertAlmostEqual(s0, 0.1285)
        self.assertGreater(s1, 0.0)

    def test_pareto_front_and_bisection_equal_brute_force(self):
        hs, vs = certify.axes(0.5)
        A_all, B_all = [], []
        for h in hs:
            VA, VE = certify.slice_X(h, vs)
            H, sup, psi_d = m.residual_terms(h, VA, VE)
            keep = H >= -1e-9
            A_all.append(sup[keep] + m.alpha(H[keep]))
            B_all.append(psi_d[keep])
        A, B = np.concatenate(A_all), np.concatenate(B_all)
        fA, fB = certify.pareto_front(A, B)
        front = {"A": fA, "psi_dtheta_0": fB}
        for rho in np.linspace(0.0, 4.0, 41):
            self.assertAlmostEqual(certify.min_D_front(front, rho), float(np.min(A - rho * B)), places=12)
        pos = B > 1e-12
        self.assertAlmostEqual(np.min(A[pos] / B[pos]), certify.sweep(0.5, 0.5)["per_gamma"]["0.5"]["rho_star"][0], places=12)


if __name__ == "__main__":
    unittest.main()
