"""The structured margin is the exact worst case over the parameter box:

    eta_str(x) = - min_{theta in Theta} c(x) Phi(x) (theta - theta_hat) = sum_k |c Phi_k| dtheta_k.

Three independent evaluations of  min_{theta in Theta} q theta,  q = c Phi,  must agree: the
closed form, enumeration of the 2^p vertices, and the dual linear program of Cohen, Belta and
Tron (CDC 2022, eq. (10)):

    max_{mu <= 0} b' mu   s.t.  A' mu = q,     A = [I; -I],  b = [theta_hat + dtheta; dtheta - theta_hat].

For a box the dual has the explicit certificate  mu+ = min(q, 0),  mu- = -max(q, 0).
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import params as p                          # noqa: E402
import model as m                           # noqa: E402
import simulate as sim                      # noqa: E402

try:
    from scipy.optimize import linprog
except ImportError:                         # pragma: no cover
    linprog = None


# ---------------------------------------------------------------- evaluators
def closed_form(q, theta_hat, dtheta):
    return float(q @ theta_hat - np.abs(q) @ dtheta)


def vertex_min(q, theta_hat, dtheta):
    n = len(q)
    best, arg = np.inf, None
    for bits in range(2 ** n):
        s = np.array([1.0 if bits >> k & 1 else -1.0 for k in range(n)])
        th = theta_hat + s * dtheta
        val = float(q @ th)
        if val < best:
            best, arg = val, th
    return best, arg


def box_halfspaces(theta_hat, dtheta):
    n = len(theta_hat)
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([theta_hat + dtheta, dtheta - theta_hat])
    return A, b


def dual_certificate(q):
    """mu* = (mu+, mu-) <= 0 with A' mu* = q, attaining the dual optimum on a box."""
    return np.concatenate([np.minimum(q, 0.0), -np.maximum(q, 0.0)])


def dual_value(mu, theta_hat, dtheta):
    _, b = box_halfspaces(theta_hat, dtheta)
    return float(b @ mu)


def dual_lp(q, theta_hat, dtheta):
    A, b = box_halfspaces(theta_hat, dtheta)
    res = linprog(-b, A_eq=A.T, b_eq=q, bounds=[(None, 0.0)] * len(b), method="highs")
    assert res.status == 0, res.message
    return -float(res.fun), res.x


def primal_lp(q, theta_hat, dtheta):
    A, b = box_halfspaces(theta_hat, dtheta)
    res = linprog(q, A_ub=A, b_ub=b, bounds=[(None, None)] * len(q), method="highs")
    assert res.status == 0, res.message
    return float(res.fun), res.x


# ---------------------------------------------------------------- tests
HAND_CASES = [
    # q, theta_hat, dtheta, min, minimizer (None = free coordinate), mu+, mu-
    dict(q=(1.0, -2.0), th=(0.0, 0.0), d=(1.0, 1.0), val=-3.0,
         arg=(-1.0, 1.0), mu_plus=(0.0, -2.0), mu_minus=(-1.0, 0.0)),
    dict(q=(2.0, 0.0), th=(1.0, 1.0), d=(0.5, 0.5), val=1.0,
         arg=(0.5, None), mu_plus=(0.0, 0.0), mu_minus=(-2.0, 0.0)),
    dict(q=(-1.0, -1.0, 3.0), th=(1.0, 2.0, 3.0), d=(1.0, 1.0, 1.0), val=1.0,
         arg=(2.0, 3.0, 2.0), mu_plus=(-1.0, -1.0, 0.0), mu_minus=(0.0, 0.0, -3.0)),
]


class TestHandCases(unittest.TestCase):
    def test_minimizer_and_certificate(self):
        for c in HAND_CASES:
            q, th, d = map(lambda v: np.array(v, float), (c["q"], c["th"], c["d"]))
            A, b = box_halfspaces(th, d)
            arg = np.array([th[k] if a is None else a for k, a in enumerate(c["arg"])])
            self.assertTrue(np.all(A @ arg <= b + 1e-12))                  # primal feasible ...
            self.assertAlmostEqual(float(q @ arg), c["val"], places=12)    # ... and attains the value
            self.assertAlmostEqual(closed_form(q, th, d), c["val"], places=12)
            self.assertAlmostEqual(vertex_min(q, th, d)[0], c["val"], places=12)
            mu = np.concatenate([c["mu_plus"], c["mu_minus"]]).astype(float)
            np.testing.assert_allclose(dual_certificate(q), mu)
            self.assertTrue(np.all(mu <= 0.0))                              # dual feasible ...
            np.testing.assert_allclose(A.T @ mu, q, atol=1e-12)
            self.assertAlmostEqual(dual_value(mu, th, d), c["val"], places=12)   # ... strong duality

    def test_weak_duality_for_other_feasible_multipliers(self):
        rng = np.random.default_rng(0)
        for c in HAND_CASES:
            q, th, d = map(lambda v: np.array(v, float), (c["q"], c["th"], c["d"]))
            A, b = box_halfspaces(th, d)
            for _ in range(100):
                r = np.maximum(q, 0.0) + rng.uniform(0.0, 3.0, size=len(q))
                mu = np.concatenate([q - r, -r])                           # A' mu = q, mu <= 0
                self.assertTrue(np.all(mu <= 0.0))
                np.testing.assert_allclose(A.T @ mu, q, atol=1e-12)
                self.assertLessEqual(dual_value(mu, th, d), c["val"] + 1e-12)


class TestRandomBoxes(unittest.TestCase):
    def test_closed_form_equals_vertex_enumeration(self):
        rng = np.random.default_rng(3)
        for n in (1, 2, 3, 5):
            for _ in range(50):
                q, th, d = rng.normal(size=n), rng.normal(size=n), rng.uniform(0.0, 2.0, size=n)
                cf = closed_form(q, th, d)
                self.assertAlmostEqual(cf, vertex_min(q, th, d)[0], places=12)
                self.assertAlmostEqual(dual_value(dual_certificate(q), th, d), cf, places=12)

    @unittest.skipUnless(linprog is not None, "scipy not available")
    def test_lp_solver_agrees(self):
        rng = np.random.default_rng(4)
        for n in (1, 2, 3, 5):
            for _ in range(20):
                q, th, d = rng.normal(size=n), rng.normal(size=n), rng.uniform(0.0, 2.0, size=n)
                cf = closed_form(q, th, d)
                dv, mu = dual_lp(q, th, d)
                pv, _ = primal_lp(q, th, d)
                self.assertAlmostEqual(dv, cf, places=9)
                self.assertAlmostEqual(pv, cf, places=9)
                self.assertTrue(np.all(mu <= 1e-9))


class TestAlongTheModel(unittest.TestCase):
    """q = c Phi with c = [-tau, 0, ..., 0]; the box is centred at theta_hat."""

    def test_one_follower_trajectory(self):
        log = sim.simulate(p.theta_a, "str")
        for k in range(0, len(log["t"]), 40):                               # 50 states
            q = m.c(1) @ np.array([[log["Phi_11"][k], log["Phi_12"][k]]])
            worst = closed_form(q, p.theta_hat, p.dtheta)                   # min_theta c Phi theta
            self.assertAlmostEqual(float(q @ p.theta_hat) - worst, log["eta_str"][k], places=12)
            self.assertAlmostEqual(vertex_min(q, p.theta_hat, p.dtheta)[0], worst, places=12)
            self.assertAlmostEqual(dual_value(dual_certificate(q), p.theta_hat, p.dtheta), worst, places=12)

    def test_queue_states(self):
        rng = np.random.default_rng(5)
        n = 3
        for _ in range(50):
            gaps = rng.uniform(p.d_min, p.h_max, size=n)
            v = rng.uniform(0.0, p.v_max, size=n)
            vA = rng.uniform(0.0, p.v_max)
            Phi = m.Phi(gaps, v, np.concatenate([[vA], v[:-1]]))            # eq. (queue)
            q = m.c(n) @ Phi
            worst = closed_form(q, p.theta_hat, p.dtheta)
            self.assertAlmostEqual(float(q @ p.theta_hat) - worst, m.eta_str(Phi), places=12)
            self.assertAlmostEqual(vertex_min(q, p.theta_hat, p.dtheta)[0], worst, places=12)
            self.assertAlmostEqual(dual_value(dual_certificate(q), p.theta_hat, p.dtheta), worst, places=12)


if __name__ == "__main__":
    unittest.main()
