#!/usr/bin/env python3
"""Figures from the saved data (results/runs.npz, results/results.json; no simulation here).

    python plotter.py               results/fig1_case_study.png/.pdf   Fig. 1 of the paper: the sketch
                                    figures/sketch.png on top, then panels (a)-(j) in one column
    python plotter.py --extended    also results/figE1_alpha_comparison, figE2_queue,
                                    figE3_independent_gains (.png/.pdf): the extended figures of the ReadMe

Fonts: Times New Roman for text when it is installed (Windows, or under /mnt/c on WSL), STIX
otherwise; Computer Modern for mathematics as in the paper's class file.
"""
import argparse
import json
import os

# Fixed creation date in the PDF metadata (matplotlib honours SOURCE_DATE_EPOCH), so a rerun
# reproduces the committed PDFs byte for byte.  2026-01-01 00:00 UTC.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

import params as p
import model as m

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SKETCH = os.path.join(HERE, "figures", "sketch.png")
FONT_DIRS = ("/mnt/c/Windows/Fonts", "C:/Windows/Fonts")
MATH_FONTSET = "cm"                 # the class file sets Times for text and leaves mathematics in Computer Modern
EPSILON = r"\epsilon"               # lunate epsilon under "cm", as in the paper

BLUE, RED, ORANGE, GRAY, CYAN, PURPLE = "blue", "red", "#e07b00", "0.35", "#009ccc", "#7b2d8e"
GM_STYLE = {0.5: dict(color=BLUE, lw=1.2, ls="-"), 0.3: dict(color=CYAN, lw=1.0, ls="-."),
            0.25: dict(color="green", lw=1.0, ls="-"), 0.2: dict(color=ORANGE, lw=1.0, ls="-"),
            0.1: dict(color=GRAY, lw=1.0, ls="--")}
BAND_COLORS = {"certified": "#c3d4ff", "slow_recovery": "#d4d4d4", "infeasible": "#ffe0e0"}
MARGIN_STYLE = {"cl": (RED, "s"), "sd": (ORANGE, "o"), "str": (BLUE, "^")}
GKEY = lambda g: f"{g:g}"


def rc(titlesize=9.6):
    """Plot setup shared by every figure."""
    from matplotlib import font_manager
    for d in FONT_DIRS:
        for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            path = os.path.join(d, f)
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": MATH_FONTSET,
        "font.size": 9.6, "axes.titlesize": titlesize, "axes.labelsize": 9.6,
        "legend.fontsize": 8.4, "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
        "legend.frameon": False, "axes.linewidth": 0.6,
    })


def load():
    Z = np.load(os.path.join(RESULTS, "runs.npz"))
    R = json.load(open(os.path.join(RESULTS, "results.json")))
    return Z, R


def xlabel_at(ax, text, x_data):
    """x-quantity label in the tick row at a data x (between two ticks)."""
    ax.text(x_data, -0.045, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=plt.rcParams["xtick.labelsize"])


def xlabel_between(ax, text, x=0.75):
    """x-quantity label in the tick row at an axes fraction."""
    ax.text(x, -0.045, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=plt.rcParams["xtick.labelsize"])


def _shown(Z, t_max):
    return Z["t"] <= t_max + 1e-9


# ====================================================================== Fig. 1 panels
def draw_margins(ax, Z, letter="a", t_max=p.t_show, title=False):
    """(a) margins along the structured run with theta_a: eta_cl, eta_str, the vertex envelope and
    the realized mismatch |c Phi delta theta|."""
    t, k = Z["t"], _shown(Z, t_max)
    ax.plot(t[k], Z["a/eta_cl"][k], color=RED, lw=1.2, label=r"$\eta_{\mathrm{cl}}$")
    ax.plot(t[k], Z["a/eta_str"][k], color=BLUE, lw=1.2, label=r"$\eta_{\mathrm{str}}(x)$")
    ax.plot(t[k], Z["a/envelope"][k], color=BLUE, ls=":", lw=1.0, marker="o", ms=2.0,
            markevery=int(round(2.0 / p.dt)), label="vertex env.")
    ax.plot(t[k], Z["a/realized"][k], color="0.4", ls="--", lw=1.0, label=r"$|c\,\Phi\,\Delta\theta|$ realized")
    xlabel_between(ax, r"$t$ [s]")
    ax.set_ylabel(r"$\eta$ [m/s]", labelpad=0)
    if title:
        ax.set_title(f"({letter}) margins: gap and tightness")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), handlelength=1.2, labelspacing=0.2,
              borderpad=0.1, borderaxespad=0.0)


def draw_barrier(ax, Z, R, letter="b", t_max=p.t_show, title=False):
    """(b) H(t) for the vertices and samples of Theta under each margin (minimum over theta in bold),
    the level tau delta_bar / gamma dashed, inset on the approach to H = 0."""
    t, k = Z["t"], _shown(Z, t_max)
    faint = dict(lw=0.4, alpha=0.18, zorder=1)
    H = {mg: Z[f"b/{mg}/H"] for mg in ("none", "str", "cl")}
    for mg, col in (("cl", RED), ("str", BLUE), ("none", "0.3")):
        for row in H[mg]:
            ax.plot(t[k], row[k], color=col, **faint)
    ax.plot(t[k], H["str"].min(axis=0)[k], color=BLUE, lw=1.2, label="structured")
    ax.plot(t[k], H["cl"].min(axis=0)[k], color=RED, lw=1.2, label="classical")
    ax.plot(t[k], H["none"].min(axis=0)[k], color="0.3", ls="--", lw=1.0, label="no margin")
    ax.axhline(0.0, color="k", lw=0.6)
    level = R["b"]["H_settled_classical"]
    ax.axhline(level, color=RED, lw=0.7, ls=(0, (3, 2)), zorder=1.5)
    ax.annotate(r"$\tau\bar{\delta}/\gamma$", xy=(2.4, level - 0.1), xytext=(2.4, 2.3), fontsize=8.4,
                color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=0.6, color=RED, relpos=(0.5, 1.0),
                                patchA=None, shrinkA=1, shrinkB=0))
    axi = ax.inset_axes([0.40, 0.415, 0.55, 0.235])
    for mg, col in (("str", BLUE), ("none", "0.3")):
        for row in H[mg]:
            axi.plot(t[k], row[k], color=col, **faint)
    axi.plot(t[k], H["str"].min(axis=0)[k], color=BLUE, lw=1.0)
    axi.plot(t[k], H["none"].min(axis=0)[k], color="0.3", ls="--", lw=0.9)
    axi.axhline(0.0, color="k", lw=0.5)
    axi.set_xlim(4.0, 20.0)
    axi.set_ylim(-0.45, 1.1)
    axi.tick_params(labelsize=7.0, length=2, pad=1)
    for sp in axi.spines.values():
        sp.set_linewidth(0.5)
    xlabel_between(ax, r"$t$ [s]")
    ax.set_ylabel(r"$H$ [m]", labelpad=0)
    if title:
        ax.set_title(rf"({letter}) min-over-$\theta$ barrier")
    ax.legend(loc="upper right", bbox_to_anchor=(0.965, 1.02), fontsize=7.6, handlelength=1.3,
              labelspacing=0.1, borderpad=0.1, borderaxespad=0.0)


def draw_stop_row(axs, Z, letters=("c", "d", "e", "f"), t_max=p.t_show, title=False):
    """(c)-(f) mandatory stop, vertex theta, stop prioritized: inputs under the classical and the
    structured margin (infeasibility shaded), then H and h for the three margins."""
    k = _shown(Z, t_max)
    t = Z["t"][k]
    S = {mg: {key: Z[f"stop/stop/{mg}/{key}"][k] for key in ("u", "u_H", "u_B", "H", "h", "infeasible")}
         for mg in ("none", "str", "cl")}
    ax_c, ax_d, ax_e, ax_f = axs
    for ax, mg, col, ttl in ((ax_c, "cl", RED, f"({letters[0]}) classical: QP infeasible"),
                             (ax_d, "str", BLUE, f"({letters[1]}) structured: feasible")):
        r = S[mg]
        ax.plot(t, r["u"], color=col, lw=1.2, label="applied $u$")
        ax.plot(t, r["u_H"], color=col, ls=(0, (5, 2, 1, 2)), lw=1.0, label=r"$u_{H}$")
        ax.plot(t, np.minimum(r["u_B"], 5.5), color="darkgreen", ls="--", lw=1.0, label=r"$u_{B}$")
        ax.axhline(p.u_max, color="k", ls=":", lw=0.7)
        ax.axhline(p.u_min, color="k", ls=":", lw=0.7)
        inf = r["infeasible"] > 0.5
        if inf.any():
            ax.fill_between(t, -4.3, 5.6, where=inf, color=RED, alpha=0.10, zorder=0)
        ax.set_ylim(-4.3, 5.6)
        ax.set_ylabel(r"$u$ [m/s$^2$]", labelpad=0)
        if title:
            ax.set_title(ttl)
    inf = S["cl"]["infeasible"] > 0.5
    t_inf = t[int(np.argmax(inf))]
    ax_c.text(0.5 * (t_inf + t[-1]), 5.5, "QP infeasible", color=RED, fontsize=8.4, ha="center", va="top",
              fontweight="bold")
    ax_d.text(0.60, 0.90, "QP feasible", transform=ax_d.transAxes, color="darkgreen", fontsize=8.4,
              ha="center", va="top", fontweight="bold")
    # two frameless legend groups per panel sharing one left anchor: "applied u" above the u = 0
    # curve, u_H / u_B side by side below it
    for ax in (ax_c, ax_d):
        hnd, lab = ax.get_legend_handles_labels()
        upper = ax.legend(hnd[:1], lab[:1], loc="lower left", bbox_to_anchor=(0.35, 0.52), handlelength=1.4,
                          fontsize=7.6, borderaxespad=0.0, borderpad=0.0, frameon=False)
        ax.add_artist(upper)
        ax.legend(hnd[1:], lab[1:], loc="upper left", bbox_to_anchor=(0.35, 0.30), labelspacing=0.08, ncol=2,
                  columnspacing=1.0, handlelength=1.4, fontsize=7.6, borderaxespad=0.0, borderpad=0.0,
                  frameon=False)
    for mg, lab, col, ls in (("str", "structured", BLUE, "-"), ("cl", "classical", RED, "-"),
                             ("none", "no margin", "0.3", "--")):
        lw = 1.2 if ls == "-" else 1.0
        ax_e.plot(t, S[mg]["H"], color=col, ls=ls, lw=lw, label=lab)
        ax_f.plot(t, S[mg]["h"], color=col, ls=ls, lw=lw, label=lab)
    ax_e.axhline(0.0, color="k", lw=0.6)
    ax_e.axhspan(-2.2, 0.0, color=RED, alpha=0.10, zorder=0)
    ax_e.set_ylim(-2.2, float(S["str"]["H"].max()) * 1.06)
    ax_e.set_ylabel(r"$H$ [m]", labelpad=0)
    ax_f.axhline(p.d_min, color=RED, ls="--", lw=0.9)
    ax_f.axhline(0.0, color="k", lw=0.6)
    ax_f.axhspan(-1.0, 0.0, color=RED, alpha=0.18, zorder=0)
    ax_f.axhspan(0.0, p.d_min, color=RED, alpha=0.08, zorder=0)
    ax_f.text(t[0] + 0.02 * (t[-1] - t[0]), p.d_min + 0.9, r"$d_{\min}$", fontsize=8.4, color=RED,
              ha="left", va="bottom")
    ax_f.set_ylim(-1.0, float(max(S[mg]["h"].max() for mg in S)) * 1.06)
    ax_f.set_ylabel(r"$h$ [m]", labelpad=0)
    if title:
        ax_e.set_title(f"({letters[2]}) control barrier function $H$")
        ax_f.set_title(f"({letters[3]}) rear gap $h$")
    for ax in (ax_e, ax_f):
        ax.legend(loc="upper right", handlelength=1.5, labelspacing=0.2, borderaxespad=0.15)
    for ax in axs:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([0, 10, 20])
        xlabel_between(ax, r"$t$ [s]", x=0.95)


def draw_steady_speed(ax, Z, R, letter="g", title=False):
    """(g) achieved steady speed v* versus v_ref for the structured and the classical margin."""
    refs, lim = Z["g/v_ref"], [0.0, 4.3]
    v_floor = R["g"]["v_floor"]
    ax.plot(lim, lim, color="0.6", lw=0.7)
    ax.axhline(v_floor, color="k", ls="--", lw=0.8)
    ax.plot(refs, Z["g/str"], color=BLUE, marker="o", ms=2.5, lw=1.0, label="structured")
    ax.plot(refs, Z["g/cl"], color=RED, marker="s", ms=2.5, lw=1.0, label="classical")
    ax.text(4.25, v_floor - 0.12, r"$v_{\mathrm{floor}}$", fontsize=8.4, ha="right", va="top")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 1, 2, 3, 4])
    xlabel_at(ax, r"$v_{\mathrm{ref}}$", 3.5)
    ax.set_ylabel(r"achieved $v^\star$ [m/s]", labelpad=0)
    if title:
        ax.set_title(f"({letter}) certifiable steady speeds")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.4, labelspacing=0.25,
              borderpad=0.15, borderaxespad=0.1)


def _eps_axis(ax, ylabel):
    ax.set_xlim(0.0, p.eps_max)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["0", "1", "2", "3", "", "5"])          # the x-quantity label takes the slot of "4"
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    xlabel_at(ax, f"${EPSILON}$ [m]", 4.0)
    ax.set_ylabel(ylabel, labelpad=0)


def draw_certification(ax, R, letter="h", gammas=p.gamma_minus_drawn, title=None,
                       band_gammas=(0.5, 0.1), label_x={0.5: 2.6, 0.3: 2.3, 0.1: 3.9}):
    """(h) the (eps, rho) plane with the boundaries rho*_Omega(eps) for each gamma_-: shading
    min_Omega D >= 0 for gamma_- = 0.5 (blue), only for gamma_- = 0.1 (grey), for no alpha (red)."""
    C = R["certify"]
    eps = np.asarray(C["eps"])
    one = C["one_sided"]
    rho_S = C["rho_star_S"]["ratio_infimum"]
    fs = plt.rcParams["legend.fontsize"]
    y_top = 2.1
    gs = sorted(band_gammas, reverse=True)
    lo = np.zeros_like(eps)
    for gm, col in zip(gs, (BAND_COLORS["certified"], BAND_COLORS["slow_recovery"])):
        hi = np.asarray(one[GKEY(gm)]["rho_star"])
        ax.fill_between(eps, lo, hi, color=col, lw=0, zorder=0)
        lo = hi
    ax.fill_between(eps, lo, 10.0, color=BAND_COLORS["infeasible"], lw=0, zorder=0)
    g_hi, g_lo = gs[0], gs[-1]
    ys_hi = np.asarray(one[GKEY(g_hi)]["rho_star"])
    slope = (np.interp(4.0, eps, ys_hi) - np.interp(0.0, eps, ys_hi)) / 4.0
    ax.text(0.12, 0.05, rf"$D\,(\gamma_-\!=\!{g_hi:g})>0$", fontsize=fs - 0.8, color=BLUE, ha="left", va="bottom")
    ax.text(3.8, 0.62, rf"$D\,(\gamma_-\!=\!{g_lo:g})>0$", fontsize=fs - 2.0, color="black", ha="center",
            va="center", rotation=np.degrees(np.arctan2(slope, 1.0)), rotation_mode="anchor",
            transform_rotates_text=True)
    ax.text(4.95, y_top - 0.02, rf"$D\,(\gamma_-\!=\!{g_lo:g})<0$", fontsize=fs - 0.8, color=RED, ha="right", va="top")
    for gm, x0 in label_x.items():                      # gamma_- labels next to their curves, along the curve
        if gm not in gammas:
            continue
        ys = np.asarray(one[GKEY(gm)]["rho_star"])
        dx = 0.5
        y0, y1, y2 = (np.interp(v, eps, ys) for v in (x0, x0 - dx, x0 + dx))
        ang = np.degrees(np.arctan2(y2 - y1, 2 * dx))
        below = gm == max(gammas)
        ax.text(x0, y0 + (-0.025 if below else 0.025), rf"$\gamma_-\!=\!{gm:g}$", fontsize=fs - 0.4,
                color=GM_STYLE[gm]["color"], ha="center", va="top" if below else "bottom", rotation=ang,
                rotation_mode="anchor", transform_rotates_text=True, zorder=4)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    for gm in gammas:
        ax.plot(eps, one[GKEY(gm)]["rho_star"], **GM_STYLE[gm])
    ax.plot([0.0], [rho_S], marker="*", color=RED, ms=7.0, ls="none", clip_on=False, zorder=5)
    ax.annotate(rf"$\rho^\ast_S={rho_S:.2f}$", xy=(0.0, rho_S), xytext=(0.22, rho_S + 0.1), fontsize=fs,
                ha="left", va="bottom")
    ax.text(0.08, 0.98, r"$\rho=1$", fontsize=fs, color="0.4", ha="left", va="top")
    ax.set_ylim(0.0, y_top)
    _eps_axis(ax, r"$\rho$")
    if title:
        ax.set_title(title if isinstance(title, str) else rf"({letter}) $\mathrm{{min}}_\Omega D$ vs. certification set")


def draw_single_slope(ax, R, letter="b", gammas=p.gamma_single_drawn, title=True):
    """figE1(b): rho*_Omega(eps) for alpha(H) = gamma H on all of X."""
    C = R["certify"]
    eps = np.asarray(C["eps"])
    for gm in gammas:
        st = dict(GM_STYLE[gm])
        if gm == min(gammas):                 # the gamma = 0.1 level is 0 and lies on the x axis: thicker, on top
            st["lw"] = 2.0
        ax.plot(eps, C["single"][GKEY(gm)]["rho_star"], label=rf"$\gamma={gm:g}$", zorder=3, clip_on=False, **st)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    fs = plt.rcParams["legend.fontsize"]
    ax.text(0.08, 0.98, r"$\rho=1$", fontsize=fs, color="0.4", ha="left", va="top")
    ax.plot([0.0], [C["rho_star_S"]["ratio_infimum"]], marker="*", color=RED, ms=7.0, ls="none", clip_on=False,
            zorder=5)
    ax.set_ylim(0.0, 1.85)
    _eps_axis(ax, r"$\rho^\ast_\Omega$")
    if title:
        ax.set_title(rf"({letter}) $\alpha(H)=\gamma H$ on all of $\mathcal{{X}}$")
    ax.legend(loc="upper right", fontsize=fs - 1.6, handlelength=3.2, labelspacing=0.1, borderaxespad=0.2)


def _n_axis(ax, ns):
    ax.set_xticks(ns)
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel(r"followers $n$", labelpad=0)


def draw_queue_margins(ax, R, letter="i", title=None, yticks=(0, 4, 8)):
    """(i) peak margins along the free maneuver versus the number of followers n."""
    q = R["queue"]
    ns, ms, lw = q["n"], 3.0, 1.2
    ax.plot(ns, q["eta_cl"], color=RED, marker="s", ms=ms, lw=lw)
    ax.plot(ns, q["eta_sd_peak"], color=ORANGE, marker="o", ms=ms, lw=lw)
    ax.plot(ns, q["eta_str_peak"], color=BLUE, marker="^", ms=ms, lw=lw)
    ax.set_ylabel("margin [m/s]", labelpad=0)
    ax.text(3.0, q["eta_cl"][2] + 0.9, r"$\eta_{\mathrm{cl}}$", fontsize=9.6, color=RED, ha="center", va="bottom")
    ax.text(9.4, q["eta_sd_peak"][ns.index(9)] + 0.85, r"$\eta_{\mathrm{sd}}$", fontsize=9.6, color=ORANGE,
            ha="center", va="bottom")
    ax.text(10.4, q["eta_str_peak"][0] + 0.6, r"$\eta_{\mathrm{str}}$: invariant in $n$", fontsize=8.4, color=BLUE,
            ha="right", va="bottom")
    ax.set_ylim(0.0, max(q["eta_cl"]) * 1.18)
    ax.set_yticks(list(yticks))
    if title:
        ax.set_title(title if isinstance(title, str) else f"({letter}) margins vs. $n$")
    _n_axis(ax, ns)


def draw_queue_min_H(ax, R, letter="j", title=None):
    """(j) min_t H of the forced stop (vertex theta, stop prioritized) for each margin versus n."""
    q = R["queue"]
    ns, ms, lw = q["n"], 3.0, 1.2
    for mg, (col, mk) in MARGIN_STYLE.items():
        ax.plot(ns, q["stop"][mg]["min_H"], color=col, marker=mk, ms=ms, lw=lw)
    ax.axhline(0.0, color=GRAY, lw=0.6, ls=":")
    ax.axhspan(-10.0, 0.0, color="0.92", lw=0, zorder=0)        # guarantee lost: H < 0
    ax.set_ylabel(r"$\mathrm{min}_t H$ [m]", labelpad=0)
    ax.set_ylim(-3.3, 0.9)
    ax.set_yticks([-3, -2, -1, 0])
    ax.text(4.0, 0.30, r"$\eta_{\mathrm{sd}}$", fontsize=9.6, color=ORANGE, ha="center", va="bottom")
    ax.text(8.0, 0.10, r"$\eta_{\mathrm{str}}$", fontsize=9.6, color=BLUE, ha="center", va="bottom")
    ax.text(2.0, -2.45, r"$\eta_{\mathrm{cl}}$", fontsize=9.6, color=RED, ha="center", va="top")
    ax.text(10.4, -0.80, "$H<0$: guarantee lost", fontsize=8.4, color=GRAY, ha="right", va="top")
    if title:
        ax.set_title(title if isinstance(title, str) else f"({letter}) forced stop vs. $n$")
    _n_axis(ax, ns)


def draw_queue_infeasible(ax, R, letter="c", title=True):
    """figE2(c): time during which the robust CBF constraint and the stop constraint are incompatible."""
    q = R["queue"]
    ns, ms, lw, T = q["n"], 3.0, 1.2, p.T
    for mg, (col, mk) in MARGIN_STYLE.items():
        ax.plot(ns, q["stop"][mg]["infeasible_time"], color=col, marker=mk, ms=ms, lw=lw)
    ax.axhline(T, color=GRAY, lw=0.6, ls=":")
    ax.text(0.7, T - 0.6, f"run length {T:g} s", fontsize=8.4, color=GRAY, ha="left", va="top")
    ax.text(5.5, q["stop"]["cl"]["infeasible_time"][ns.index(5)] - 1.2, r"$\eta_{\mathrm{cl}}$", fontsize=9.6,
            color=RED, ha="center", va="top")
    ax.text(7.0, q["stop"]["sd"]["infeasible_time"][ns.index(6)] - 1.2, r"$\eta_{\mathrm{sd}}$", fontsize=9.6,
            color=ORANGE, ha="center", va="top")
    ax.text(10.4, 1.0, r"$\eta_{\mathrm{str}}$: feasible for every $n$", fontsize=8.4, color=BLUE, ha="right",
            va="bottom")
    ax.set_ylim(-1.5, T * 1.1)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("infeasible time [s]", labelpad=0)
    if title:
        ax.set_title(f"({letter}) forced stop: infeasible time vs. $n$")
    _n_axis(ax, ns)


# ====================================================================== Fig. 1: the column layout
PANEL_H = 0.866            # realized height of the (a)/(b) panels [in]
ROW_SCALE_STOP = 0.75      # (c)-(f) height relative to (a)/(b)
ROW_SCALE_QUEUE = 0.72     # (i)/(j) height relative to (a)/(b)
WIDTH_MAX = 3.5            # the saved figure may not be wider than this [in]
TIME_ROW_GAP = 0.05        # minimum gap between stacked time-series panels [in]
ROW_CLEAR_PT = (1.5, 1.5, 3.0, 3.0)     # clearance above each row gap
LETTER_PAD_PT = 1.5        # clearance between a panel letter and the y label / tick labels
LETTER_INK_SHIFT_PT = 2.6  # letters start this far right of the y-label box (ink alignment)
YLABEL_OVERHANG = 0.10     # a y label pushed down by its letter may hang this far below the panel [in]
SHORT_YLABEL = {r"achieved $v^\star$ [m/s]": r"$v^\star$ [m/s]", "margin [m/s]": r"$\eta$ [m/s]",
                r"$\mathrm{min}_t H$ [m]": r"$\mathrm{min}_t H$"}


def _close_rows(fig, rows, gap_min=TIME_ROW_GAP, clear_pt=ROW_CLEAR_PT):
    """tight_layout uses one row gap for the whole grid; close every gap to gap_min, or to what the
    tick labels and annotations sticking out of the panels need, by moving the upper rows down."""
    ren = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fh = fig.get_figheight()

    def below(axs):
        return max(ax.get_position().y0 - inv.transform(ax.get_tightbbox(ren))[0, 1] for ax in axs)

    def above(axs):
        return max(inv.transform(ax.get_tightbbox(ren))[1, 1] - ax.get_position().y1 for ax in axs)

    for i in range(len(rows) - 1, 0, -1):
        upper, lower = rows[i - 1], rows[i]
        gap_now = upper[0].get_position().y0 - lower[0].get_position().y1
        gap_want = max(gap_min / fh, below(upper) + above(lower) + clear_pt[i - 1] / 72.0 / fh)
        d = gap_now - gap_want
        if d <= 0:
            continue
        for ax in (a for row in rows[:i] for a in row):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - d, pos.width, pos.height])


def _letters_top_left(fig, panels, letters, fs=9.6, pad_pt=LETTER_PAD_PT):
    """Panel letters outside each panel at its top-left corner, left-aligned with the column's y labels;
    a y label is pushed down to clear its letter or replaced by its short form."""
    ren = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fh = fig.get_figheight()
    pad = pad_pt / 72.0
    cols = {}
    for ax in panels:
        cols.setdefault(round(ax.get_position().x0, 3), []).append(ax)
    for axs in cols.values():
        fig.align_ylabels(axs)
    fig.canvas.draw()
    x_left = {key: min(inv.transform(a.yaxis.label.get_window_extent(ren))[0, 0] for a in axs)
              for key, axs in cols.items()}
    notes = []
    for ax, L in zip(panels, letters):
        pos = ax.get_position()
        t = fig.text(x_left[round(pos.x0, 3)] + LETTER_INK_SHIFT_PT / 72.0 / fig.get_figwidth(), pos.y1,
                     f"({L})", ha="left", va="top", fontsize=fs)
        bb = inv.transform(t.get_window_extent(ren))
        letter_bottom = bb[0, 1] - pad / fh
        for tk in ax.yaxis.get_ticklabels():
            if not (tk.get_visible() and tk.get_text()):
                continue
            tb = inv.transform(tk.get_window_extent(ren))
            if tb[1, 1] > bb[0, 1] and tb[0, 1] < bb[1, 1] and tb[0, 0] < bb[1, 0]:
                notes.append(f"letter ({L}) overlaps the y tick label '{tk.get_text()}'")

        def overlap():
            yl = inv.transform(ax.yaxis.label.get_window_extent(ren))
            return yl[1, 1] - letter_bottom, yl[0, 1] - (pos.y0 - YLABEL_OVERHANG / fh)

        over, room = overlap()
        if over > 0 and over > room and ax.get_ylabel() in SHORT_YLABEL:
            ax.set_ylabel(SHORT_YLABEL[ax.get_ylabel()], labelpad=0)
            ax.get_tightbbox(ren)
            over, room = overlap()
        if over > 0:
            if over > room + 1e-6:
                notes.append(f"y label of ({L}) hangs {(over - room) * fh:.2f} in below the allowed overhang")
            x, y = ax.yaxis.label.get_position()
            ax.yaxis.label.set_position((x, y - over / pos.height))
    return notes


def fig1(Z, R, path, dpi=300, panel_h=PANEL_H, width=WIDTH_MAX, sketch=SKETCH):
    """Fig. 1: the sketch on top, then (a)(b) / (c)(d) / (e)(f) on a shared time axis, (g)(h), (i)(j);
    letters outside the panels; the figure height is iterated until the (a)/(b) panels are panel_h tall."""
    rc()
    img = mpimg.imread(sketch)
    aspect = img.shape[0] / img.shape[1]
    heights = [(width - 0.50) * aspect, panel_h, panel_h * ROW_SCALE_STOP, panel_h * ROW_SCALE_STOP, panel_h,
               panel_h * ROW_SCALE_QUEUE]

    def layout(fig_h, fig_w):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(6, 2, height_ratios=heights)
        ax_sketch = fig.add_subplot(gs[0, :])
        ab = [fig.add_subplot(gs[1, c]) for c in (0, 1)]
        cd = [fig.add_subplot(gs[2, c]) for c in (0, 1)]
        ef = [fig.add_subplot(gs[3, c]) for c in (0, 1)]
        gh = [fig.add_subplot(gs[4, c]) for c in (0, 1)]
        ij = [fig.add_subplot(gs[5, c]) for c in (0, 1)]
        draw_margins(ab[0], Z)
        draw_barrier(ab[1], Z, R)
        draw_stop_row(cd + ef, Z)
        draw_steady_speed(gh[0], Z, R)
        draw_certification(gh[1], R)
        draw_queue_margins(ij[0], R)
        draw_queue_min_H(ij[1], R)
        for ax in ab:                        # (a), (b) share the time axis of the stop rows
            ax.set_xlim(cd[0].get_xlim())
            ax.set_xticks(cd[0].get_xticks())
        ax_sketch.imshow(img, interpolation="bilinear")
        ax_sketch.set_axis_off()
        panels = ab + cd + ef + gh + ij
        for ax in ab + cd:                   # tick labels and t [s] only on the third time row
            ax.tick_params(axis="x", labelbottom=False, direction="in")
            for t in list(ax.texts):
                if "[s]" in t.get_text():
                    t.remove()
        for ax in ef + gh + ij:
            ax.tick_params(axis="x", length=2.5, pad=2.0)
        fig.tight_layout(pad=0.35, w_pad=0.5, h_pad=0.12)
        rows = (ab, cd, ef, gh, ij)
        _close_rows(fig, rows)
        notes = _letters_top_left(fig, panels, "abcdefghij")
        b0, b1 = ab[0].get_position(), ab[1].get_position()
        w_frac = b1.x1 - b0.x0
        h_frac = w_frac * fig_w * aspect / fig_h
        x_ink = fig.get_tightbbox(fig.canvas.get_renderer()).x0 / fig_w
        ax_sketch.set_position([0.5 * (x_ink + b1.x1) - 0.5 * w_frac, b0.y1 + 0.06 / fig_h, w_frac, h_frac])
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        return fig, b0.height * fig_h, (tb.width, tb.height), notes

    fig_h, fig_w = sum(heights) + 1.0, width
    fig, h_a, tight, notes = layout(fig_h, fig_w)
    for _ in range(10):
        if abs(h_a - panel_h) < 0.002 and tight[0] <= WIDTH_MAX + 0.005:
            break
        plt.close(fig)
        fig_h *= panel_h / h_a
        fig_w -= max(0.0, tight[0] - WIDTH_MAX)
        fig, h_a, tight, notes = layout(fig_h, fig_w)
    for n in notes:
        print("  warning:", n)
    print(f"Fig. 1: panel (a) height {h_a:.3f} in, figure {tight[0]:.2f} x {tight[1]:.2f} in")
    _save(fig, path, dpi)


def _save(fig, path, dpi):
    for ext in (".png", ".pdf"):
        fig.savefig(path + ext, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        print("wrote", path + ext)
    plt.close(fig)


# ====================================================================== extended figures
def figE1(R, path, dpi=300):
    """E1: (a) Fig. 1(h), the one-sided alpha; (b) a single slope gamma on all of X."""
    rc(titlesize=8.8)
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1))
    draw_certification(axs[0], R, letter="a",
                       title=r"(a) $\alpha(H)=\gamma H$ ($H\geq0$), $\gamma_-H$ ($H<0$), as Fig. 1(h)")
    draw_single_slope(axs[1], R, letter="b")
    axs[1].set_ylim(*axs[0].get_ylim())
    fig.tight_layout(pad=0.35, w_pad=0.8)
    _save(fig, path, dpi)


def figE2(R, path, dpi=300):
    """E2: Fig. 1(i), (j) as (a), (b) plus (c) the infeasible time of the forced stop versus n."""
    rc(titlesize=8.8)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
    draw_queue_margins(axs[0], R, title="(a) peak margins vs. $n$", yticks=(0, 4, 8, 12))
    draw_queue_min_H(axs[1], R, title=r"(b) forced stop: $\mathrm{min}_t H$ vs. $n$")
    draw_queue_infeasible(axs[2], R, letter="c")
    fig.tight_layout(pad=0.35, w_pad=0.9)
    _save(fig, path, dpi)


def figE3(R, path, dpi=300):
    """E3: eta_sd with shared versus independent follower gains, followers from rest versus at the
    equilibrium: (a) peak margins, (b) min_t H and (c) infeasible time of the forced stop, versus n
    (unit spacing up to n = 10, then 15, 20, 30)."""
    rc(titlesize=8.8)
    G = R["gains"]
    ns = G["n"]
    pos = lambda n: n if n <= 10 else 10 + 0.4 * (n - 10)
    xs = [pos(n) for n in ns]
    x_at = dict(zip(ns, xs))
    style = {"rest/shared": (ORANGE, "-", "o"), "rest/independent": (ORANGE, "--", "s"),
             "equilibrium/shared": (PURPLE, "-", "o"), "equilibrium/independent": (PURPLE, "--", "s")}
    label = {"rest/shared": r"$\max_t\eta_{\mathrm{sd}}$: rest, shared (Fig. 1)", "rest/independent": "rest, independent",
             "equilibrium/shared": "equilibrium, shared", "equilibrium/independent": "equilibrium, independent"}
    ms, lw, T = 3.0, 1.2, p.T
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45))
    ax = axs[0]
    ax.plot(xs, G["eta_cl"], color=RED, marker="s", ms=ms, lw=lw)
    handles = []
    for case, (col, ls, mk) in style.items():
        handles += ax.plot(xs, G["cases"][case]["eta_sd_peak"], color=col, ls=ls, marker=mk, ms=ms, lw=lw,
                           label=label[case])
    ax.plot(xs, G["eta_str_peak"], color=BLUE, marker="^", ms=ms, lw=lw)
    ax.text(x_at[20] - 0.3, G["eta_cl"][ns.index(20)] + 2.8, r"$\eta_{\mathrm{cl}}$", fontsize=9.6, color=RED,
            ha="right", va="bottom")
    ax.text(x_at[30] + 0.4, G["eta_str_peak"][-1] + 2.0, r"$\eta_{\mathrm{str}}$ (all cases)", fontsize=8.4,
            color=BLUE, ha="right", va="bottom")
    ax.set_ylabel("margin [m/s]", labelpad=0)
    ax.set_ylim(0.0, 46.0)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_title("(a) peak margins vs. $n$")
    ax = axs[1]
    for case, (col, ls, mk) in style.items():
        ax.plot(xs, G["cases"][case]["min_H"], color=col, ls=ls, marker=mk, ms=ms, lw=lw)
    ax.axhline(0.0, color=GRAY, lw=0.6, ls=":")
    ax.axhspan(-10.0, 0.0, color="0.92", lw=0, zorder=0)
    ax.text(x_at[30] + 0.4, -2.15, "$H<0$: guarantee lost", fontsize=8.4, color=GRAY, ha="right", va="top")
    ax.set_ylabel(r"$\mathrm{min}_t H$ [m]", labelpad=0)
    ax.set_ylim(-3.4, 1.2)
    ax.set_yticks([-3, -2, -1, 0, 1])
    ax.set_title(r"(b) forced stop: $\mathrm{min}_t H$ vs. $n$")
    ax = axs[2]
    for case, (col, ls, mk) in style.items():
        ax.plot(xs, G["cases"][case]["infeasible_time"], color=col, ls=ls, marker=mk, ms=ms, lw=lw)
    ax.axhline(T, color=GRAY, lw=0.6, ls=":")
    ax.text(0.7, T + 0.5, f"run length {T:g} s", fontsize=8.4, color=GRAY, ha="left", va="bottom")
    ax.set_ylim(-1.5, T * 1.12)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("infeasible time [s]", labelpad=0)
    ax.set_title("(c) forced stop: infeasible time vs. $n$")
    for ax in axs:
        ax.set_xlim(0.5, pos(30) + 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel(r"followers $n$", labelpad=0)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=4, handlelength=2.8,
               columnspacing=1.6, borderaxespad=0.0)
    fig.tight_layout(pad=0.35, w_pad=0.9, rect=(0, 0.11, 1, 1))
    _save(fig, path, dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--extended", action="store_true", help="also build the extended figures E1-E3")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    Z, R = load()
    fig1(Z, R, os.path.join(RESULTS, "fig1_case_study"), dpi=args.dpi)
    if args.extended:
        figE1(R, os.path.join(RESULTS, "figE1_alpha_comparison"), dpi=args.dpi)
        figE2(R, os.path.join(RESULTS, "figE2_queue"), dpi=args.dpi)
        figE3(R, os.path.join(RESULTS, "figE3_independent_gains"), dpi=args.dpi)


if __name__ == "__main__":
    main()
