# Minimal Robust Safety Margins Under Parametric Environment Uncertainty: Interaction-Aware Control Barrier Functions

Code for the case study (Section V) of

> C. R. He and T. G. Molnar, "Minimal Robust Safety Margins Under Parametric Environment
> Uncertainty: Interaction-Aware Control Barrier Functions," *IEEE Control Systems Letters*, 2026.
> arXiv: (to appear)

It reproduces Fig. 1 of the paper, every number quoted in Section V, and three extended figures
that are not in the paper (below).

```bibtex
@article{He2026minimal,
  author  = {He, Chaozhe R. and Molnar, Tamas G.},
  title   = {Minimal Robust Safety Margins Under Parametric Environment Uncertainty:
             Interaction-Aware Control Barrier Functions},
  journal = {IEEE Control Systems Letters},
  year    = {2026},
}
```

## What is computed

A controlled lead vehicle A brakes in front of `n` reactive followers E_1, ..., E_n that run the
optimal-velocity car-following policy with uncertain gains `theta` in a box `Theta`.  The leader's
input passes through a robust CBF safety filter protecting the gap to E_1 with one of three
margins: the structured margin `eta_str` of the paper (the exact worst case over the box at the
current state), the state-dependent-radius margin `eta_sd`, or the classical margin `eta_cl`
built on a uniform bound.  The scripts simulate the braking maneuver, a mandatory stop with a
second (braking-distance) CBF, a steady-speed sweep and a queue of up to ten followers, and
compute the certifiable uncertainty level `rho*_Omega` on the certification sets `Omega_eps` of
the operating set.

| file | content |
|---|---|
| `params.py` | every constant of Section V, with the paper's symbols |
| `model.py` | policy `pi_E`, regressor `Phi`, barrier `H`, margins `eta_str`, `eta_sd`, `eta_cl`, bounds `u_H`, `u_B`, the two priority rules, the residual `D` and its Lipschitz constants |
| `simulate.py` | forward-Euler closed loop for `n >= 1` followers, free maneuver or mandatory stop |
| `certify.py` | `rho*_Omega` versus the depth `eps` of `Omega` for each `gamma_-`, as the ratio infimum of eq. (rhostar) on a grid of the operating set; `rho*_S` also by bisection on the sign of the grid minimum, plus the certified lower bound of the Remark on computation (Lipschitz constants times the grid spacing); sharpness state; `gamma_-` threshold |
| `generate_results.py` | runs everything, writes `results/runs.npz` and `results/results.json` |
| `stats.py` | prints every quoted number next to the value in the paper, PASS / FAIL |
| `plotter.py` | Fig. 1 (`results/fig1_case_study.png/.pdf`) and, with `--extended`, the extended figures |
| `run_all.py` | tests, results, numbers, figures in order |
| `tests/` | unit tests (`manuscript.py` holds the printed values as strings) |
| `figures/sketch.png` | the drawing above Fig. 1(a) (authored, not generated) |
| `results/` | committed outputs, so the figures can be drawn without simulating |

## Requirements

Python >= 3.9 with `numpy` and `matplotlib`; `scipy` is optional (one cross-check in the tests
is skipped without it).  Tested with Python 3.9 to 3.12, numpy 1.26 to 2.4, matplotlib 3.9.4.

The commands below use [uv](https://docs.astral.sh/uv/), which creates the environment from
`pyproject.toml` on the first `uv run`.  `uv.lock` pins matplotlib 3.9.4, the version that drew
the committed figures (later versions move the tight bounding box by a few pixels; the content
is the same).  Without uv, run `pip install .` (or `pip install numpy matplotlib scipy`) once
and drop the `uv run` prefix.

## Usage

```
uv run run_all.py                        # everything, about half a minute (--extra test adds scipy)
```
or step by step:
```
uv run --extra test -m unittest discover tests   # model-level tests (seconds; test_results needs results.json)
uv run generate_results.py                       # simulations + certification sweep -> results/ (about 15 s)
uv run stats.py                                  # the numbers of Section V, PASS / FAIL against the printed ones
uv run plotter.py --extended                     # results/fig1_case_study.*, figE1_*, figE2_*, figE3_*
```
`--extra test` adds `scipy` for the dual-LP cross-check; the tests also run without it.
`generate_results.py --step 0.1` uses a coarser grid in the certification sweep (the quoted
numbers are unchanged; the curves of Fig. 1(h) are then sampled every 0.1 m instead of 0.05 m).

## Reproducibility

Every run is deterministic: forward Euler with step 0.02 s over 40 s, and the twenty parameter
samples of Fig. 1(b) are drawn with `numpy.random.default_rng(0)` after the four vertices of the
box.  Two conventions the paper does not spell out: positions are rear bumpers with vehicle
length `L = 5` m, so the gap is `h = s_A - s_E - L`, and the origin is the leader's rear bumper
at `t = 0` (the stop line is then at `s_line = 45` m as printed); speeds are clamped at zero (the
vehicles do not reverse).  Neither changes any number: `L` cancels in every gap, and the clamp is
never active in the free maneuver.

Fonts: the figures use Times New Roman for text when it is installed (Windows, or
`/mnt/c/Windows/Fonts` under WSL) and STIX otherwise; mathematics is set in Computer Modern as in
the paper's class file.  With another text font the panels are identical up to the glyph shapes.

## Extended figures (not in the paper)

Built by `plotter.py --extended` from the same model; the numbers appear in `stats.py`.

- `results/figE1_alpha_comparison.png`: (a) Fig. 1(h), the extended class K function
  `alpha(H) = gamma H` for `H >= 0` and `gamma_- H` for `H < 0`; (b) a single slope `gamma` on all
  of the operating set.  For `gamma <= 0.25` the binding state of (b) moves into the safe set
  (to `(h, v_A, v_E) = (22, 0, 5)`, `H = 16` m) and the level is flat in `eps`: 1.07, 0.57 and 0
  for `gamma = 0.25, 0.2, 0.1`; no single slope certifies the operating set above 0.93, whereas
  the one-sided function reaches `rho*_S = 1.45` on all of it for every `gamma_- <= 0.108` 1/s.
- `results/figE2_queue.png`: Fig. 1(i), (j) as (a), (b), plus (c) the time during which the robust
  CBF constraint and the stop constraint are incompatible, `u_H > min(u_B, u_max)`.  The `eta_sd`
  filter is infeasible for 3.9 s at `n = 2` (from 13.1 s) and 31.2 s at `n = 10` (from 8.8 s to
  the end of the run; the window reaches the end of the run from `n = 5` on); the classical filter
  is infeasible from 5.7 s (`n = 1`) to 1.2 s (`n = 10`) until the end of the run; the structured
  filter is feasible throughout for every `n`.
- `results/figE3_independent_gains.png`: `eta_sd` when every follower has its own uncertain pair
  `(theta_1i, theta_2i)` in the same box (block-diagonal regressor; the true chain still shares
  `theta`), for followers starting from rest 40 m apart (as in Fig. 1) or all at the equilibrium.
  With independent gains and followers from rest the peak of `eta_sd` grows from 2.94 m/s
  (`n = 2`) to 14.46 m/s (`n = 10`) and exceeds `eta_cl` from `n = 5` on; under the mandatory stop
  the `eta_sd` filter is infeasible for 10.4 s at `n = 2`, 31.6 s at `n = 3` and the whole run from
  `n = 10`, and `H` reaches -0.45 m at `n = 3` and -3.0 m for `n >= 10`.  `eta_str` is the same in
  every case.

## Contact

Chaozhe R. He, University at Buffalo (chaozheh@buffalo.edu).
