"""Second group of paper figures: parameter-free scaling, the finite size
location term, tails by symmetry class, the maximum laws, the multigraph
model, the universal cover densities, and drawings of the base graphs and
of the leading defects.

Called from build_figures.main() (so that images/README.md lists every
figure), after analysis/finite_size.py has written finite_size_fits.csv.
It can also be run on its own from the repo root:
    python3 analysis/build_figures_extended.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_figures as bf  # noqa: E402
import finite_size as fs  # noqa: E402
import tw_fredholm as T  # noqa: E402
import universal_cover as U  # noqa: E402
from dataset_utils import (  # noqa: E402
    ATOMS_K5_MINUS_E, MAX_LAW, REPO_ROOT, RHO_K4_MINUS_E, RHO_K5_MINUS_E,
    TW_SHAPE_MOMENTS, discover_datasets, size_scaling_prefactor,
)

INK, INK_2, MUTED, GRID = bf.INK, bf.INK_2, bf.MUTED, bf.GRID
C4 = size_scaling_prefactor()
RHO4 = 2 * np.sqrt(3)
LAW_COLOR = {"tw1": "#2a78d6", "tw2": "#1baf7a", "tw4": "#eda100"}
LAW_LS = {"tw1": "-", "tw2": "--", "tw4": "-."}
LAW_NAME = {"tw1": r"$TW_1$", "tw2": r"$TW_2$", "tw4": r"$TW_4$",
            "max12": r"$\max(TW_1,TW_2)$", "max22": r"$\max(TW_2,TW_2')$",
            "normal": "normal"}

# Multigraph (permutation model) defect eigenvalues of cycle rank two:
# a triple edge, or two adjacent looped vertices, bind at exactly 7/2; a
# looped vertex joined to a neighbor by a double edge binds at the real
# root of lambda^3 = 8 lambda + 16.
ATOM_TRIPLE = 3.5
ATOM_LOOP_DOUBLE = float(np.real(np.roots([1, 0, -8, -16])[0]))
# a triangle with a loop at one vertex binds just above the threshold
ATOM_TRIANGLE_LOOP = 3.4675038571
# Cycle rank two defects of K4 - e (generalized thetas with branches).
ATOMS_K4E = {"theta(2,2,4)": 2.5118521, "theta(1,2,5)": 2.5136918}

# Edge coefficients C_B of the universal cover densities, and the
# predicted amplitudes C_B^(-2/3) in the V^(-2/3) normalization.
EDGE_C = {
    "k5_cover": U.edge_constant(U.K5, RHO4),
    "k5_minus_edge": U.edge_constant(U.K5_MINUS_E, RHO_K5_MINUS_E),
    "k4_minus_edge": U.edge_constant(U.K4_MINUS_E, RHO_K4_MINUS_E),
}
PRED_C = {k: v ** (-2 / 3) for k, v in EDGE_C.items()}


def color_of(family, k=None):
    for fam, kk, label, color, marker in bf.SERIES:
        if fam == family and (kk is None or kk == k):
            return color, marker, label
    raise KeyError(family)


def load_fits():
    rows = []
    with open(fs.CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            for key, value in row.items():
                if key in ("file", "family", "law"):
                    continue
                row[key] = float(value) if value != "" else None
            rows.append(row)
    return rows


def fit_series(fits, family, k=None):
    out = [r for r in fits if r["family"] == family
           and (k is None or r["cover_deg"] == k)]
    return sorted(out, key=lambda r: r["n"])


def record_for(family, n, k=None):
    for r in discover_datasets():
        size = r["base_size"] if fs.SPEC[fs.key_of(r)][2] == "base" else r["V"]
        if r["family"] == family and size == n and (k is None or r["cover_deg"] == k):
            return r
    raise KeyError((family, n, k))


def load_s(record):
    x = np.load(record["path"])
    s, law, c, n = fs.rescale(record, x)
    return s, law, c, n


def ecdf_on(s_sorted, t):
    return np.searchsorted(s_sorted, t, side="right") / s_sorted.size


# ---------------------------------------------------------------------------
# A. Drawings of the base graphs and a cover
# ---------------------------------------------------------------------------

def _draw_graph(ax, pos, edges, loops=(), node_color=INK, lw=1.2,
                edge_color=INK_2, ms=5, curved=()):
    for (u, v) in edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, lw=lw, zorder=1,
                solid_capstyle="round")
    for (u, v, rad) in curved:
        arrow = FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="-", color=edge_color, lw=lw, zorder=1)
        ax.add_patch(arrow)
    for (v, direction, size) in loops:
        x, y = pos[v]
        cx, cy = x + size * np.cos(direction), y + size * np.sin(direction)
        circ = plt.Circle((cx, cy), size, fill=False, color=edge_color, lw=lw,
                          zorder=1)
        ax.add_patch(circ)
    for v, (x, y) in pos.items():
        ax.plot(x, y, "o", color="white", ms=ms + 2.5, zorder=2)
        ax.plot(x, y, "o", color=node_color, ms=ms, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_base_graphs():
    fig, axes = plt.subplots(1, 5, figsize=(6.8, 1.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})
    # bouquet of two loops
    ax = axes[0]
    _draw_graph(ax, {0: (0, 0)}, [], loops=[(0, np.pi / 2, 0.45),
                                             (0, -np.pi / 2, 0.45)])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("bouquet $B_2$\n" + r"$\rho=3.4641$", fontsize=8)
    pent = {i: (np.sin(2 * np.pi * i / 5), np.cos(2 * np.pi * i / 5))
            for i in range(5)}
    ax = axes[1]
    _draw_graph(ax, pent, U.K5)
    ax.set_title("$K_5$\n" + r"$\rho=3.4641$", fontsize=8)
    ax = axes[2]
    _draw_graph(ax, pent, U.K5_MINUS_E)
    ax.plot(*zip(pent[0], pent[1]), ls=(0, (1.5, 2)), color=MUTED, lw=0.9)
    ax.set_title("$K_5-e$\n" + r"$\rho=3.2629$", fontsize=8)
    sq = {0: (-1, 0), 1: (1, 0), 2: (0, 0.85), 3: (0, -0.85)}
    ax = axes[3]
    _draw_graph(ax, sq, U.K4_MINUS_E)
    ax.plot(*zip(sq[0], sq[1]), ls=(0, (1.5, 2)), color=MUTED, lw=0.9)
    ax.set_title("$K_4-e$\n" + r"$\rho=2.5083$", fontsize=8)
    # a random 3-cover of K4 - e, drawn fiber by fiber
    ax = axes[4]
    rng = np.random.default_rng(3)
    k = 3
    fiber_x = {0: -1.5, 2: -0.5, 3: 0.5, 1: 1.5}
    pos = {(v, i): (fiber_x[v], 0.9 - 0.9 * i) for v in range(4) for i in range(k)}
    edges = []
    for (u, v) in U.K4_MINUS_E:
        perm = rng.permutation(k)
        edges += [((u, i), (v, perm[i])) for i in range(k)]
    # vertical order: 0, 2, 3, 1 so that all base edges join nearby fibers
    for fx in fiber_x.values():
        ax.add_patch(plt.Rectangle((fx - 0.18, -1.05), 0.36, 2.1, color=GRID,
                                   zorder=0, lw=0))
    _draw_graph(ax, pos, edges, ms=4, lw=0.9,
                curved=[])
    ax.set_title("a random $3$-cover\nof $K_4-e$ (fibers shaded)", fontsize=8)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.2, 1.2)
    fig.subplots_adjust(wspace=0.15)
    return fig


# ---------------------------------------------------------------------------
# B. The limit laws: Tracy-Widom densities and the Ramanujan functionals
# ---------------------------------------------------------------------------

def fig_tw_laws():
    s = np.linspace(-6, 4, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    ax = axes[0]
    for beta, law in ((1, "tw1"), (2, "tw2"), (4, "tw4")):
        f = T.pdf(beta, s)
        ax.plot(s, f, color=LAW_COLOR[law], ls=LAW_LS[law], lw=1.4,
                label=f"{LAW_NAME[law]}:  $F_{beta}(0)={float(T.cdf(beta, 0)):.5f}$")
        ax.fill_between(s[s >= 0], f[s >= 0], color=LAW_COLOR[law], alpha=0.25,
                        lw=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("$s$ (original Tracy–Widom normalization)")
    ax.set_ylabel("density")
    ax.set_title("Pure laws (shaded: $1-F_\\beta(0)$)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(0, 0.68)
    ax = axes[1]
    laws = [("tw1", "#2a78d6", "-"), ("max12", "#4a3aa7", "--"),
            ("max22", "#e34948", "-.")]
    for law, color, ls in laws:
        f = fs.law_pdf(law, s)
        ax.plot(s, f, color=color, ls=ls, lw=1.4,
                label=f"{LAW_NAME[law]}:  $P(0)={float(fs.law_cdf(law, 0)):.5f}$")
        ax.fill_between(s[s >= 0], f[s >= 0], color=color, alpha=0.2, lw=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("$s$")
    ax.set_title(r"Maximum laws of the $\mathbb{Z}_4$ and $\mathbb{Z}_5$ covers", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.subplots_adjust(wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# C. Parameter-free comparison of the regular graph data with the proven law
# ---------------------------------------------------------------------------

def fig_paramfree_simple(fits):
    t = np.linspace(-5, 3, 801)
    F1 = T.cdf(1, t)
    a_star = np.mean([r["a"] for r in fit_series(fits, "simple")
                      if r["n"] >= 5000])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    sizes = [(("simple", 200), "#9ec5f4"), (("simple", 1000), "#5598e7"),
             (("simple", 5000), "#256abf"), (("simple", 20000), "#0d366b"),
             (("matlab_simple", 500000), "#1baf7a")]
    for (fam, V), color in sizes:
        s, _, c, n = load_s(record_for(fam, V))
        s.sort()
        lab = f"$V={V:,}$".replace(",", "{,}")
        if fam == "matlab_simple":
            lab += " (legacy)"
        axes[0].plot(t, ecdf_on(s, t) - F1, color=color, lw=1.2, label=lab)
        shift = a_star / (c * n ** (1 / 3))
        axes[1].plot(t, ecdf_on(s + shift, t) - F1, color=color, lw=1.2,
                     label=lab)
        del s
    for ax in axes:
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel(r"$s=(\lambda_2-2\sqrt{3})\,V^{2/3}/c(4)$")
    axes[0].set_ylabel(r"$F_V(s)-F_1(s)$")
    axes[0].set_title("Nothing fitted", fontsize=8.5)
    axes[1].set_title(f"Shifted by $a/V$, one constant $a={a_star:.2f}$",
                      fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.subplots_adjust(wspace=0.08)
    return fig, a_star


# ---------------------------------------------------------------------------
# D. The two finite size parameters: scale ratio and location coefficient
# ---------------------------------------------------------------------------

REG_GROUP = [("simple", None), ("matlab_simple", None), ("matlab_loop_cover", None)]
COVER_GROUP = [("abelian_cover", 3), ("abelian_cover", 4), ("abelian_cover", 5),
               ("quaternion", 4), ("k5_cover", None), ("k5_minus_edge", None),
               ("k4_minus_edge", None)]


def _predicted_ratio(r):
    """True scale / conjectured scale; for the irregular bases the reference
    is the universal cover prediction C_B^(-2/3) instead of c(4)."""
    if r["family"] in ("k5_minus_edge", "k4_minus_edge"):
        return r["c_fit"] / PRED_C[r["family"]]
    return r["scale_ratio"]


def fig_scale_location(fits):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for fam, k in REG_GROUP + COVER_GROUP:
        rows = fit_series(fits, fam, k)
        color, marker, label = color_of(fam, k)
        n = np.array([r["n"] for r in rows])
        ratio = np.array([_predicted_ratio(r) for r in rows])
        axes[0].semilogx(n, ratio, marker=marker, ms=4, lw=1.1, color=color,
                         label=label)
    axes[0].axhline(1, color=INK, lw=0.9)
    axes[0].set_xlabel("size $n$ ($V$, or base size for character blocks)")
    axes[0].set_ylabel("fitted scale / predicted scale")
    axes[0].set_title("Scale, relative to the predicted constant", fontsize=8.5)
    axes[0].set_ylim(0.96, 1.42)
    for fam, k in REG_GROUP + COVER_GROUP:
        rows = fit_series(fits, fam, k)
        color, marker, label = color_of(fam, k)
        n = np.array([r["n"] for r in rows])
        a = np.array([r["a"] for r in rows])
        se = np.array([r["a_se"] for r in rows])
        axes[1].errorbar(n, a, yerr=2 * se, marker=marker, ms=4, lw=1.1,
                         color=color, capsize=0, elinewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].axhline(0, color=INK, lw=0.9)
    axes[1].set_xlabel("size $n$")
    axes[1].set_ylabel(r"location coefficient $a$")
    axes[1].set_title(r"Location coefficient $a$ (bars: $\pm2$ s.e.)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.13))
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    return fig


# ---------------------------------------------------------------------------
# E. The Ramanujan fraction: the one parameter model and its slow approach
# ---------------------------------------------------------------------------

def fig_ramanujan_rate(fits):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    Vgrid = np.logspace(3, 9, 300)
    F0 = float(T.cdf(1, 0))
    models = [("simple", None, 4.45), ("matlab_loop_cover", None, -1.4),
              ("k5_cover", None, 0.03)]
    for fam, k, a in models:
        color, marker, label = color_of(fam, k)
        rows = fit_series(fits, fam, k)
        V = np.array([r["n"] for r in rows])
        P = np.array([r["frac_ramanujan"] for r in rows])
        axes[0].semilogx(V, P, ls="", marker=marker, ms=4.5, color=color,
                         label=label)
        axes[0].semilogx(Vgrid, T.cdf(1, a / (C4 * Vgrid ** (1 / 3))),
                         color=color, lw=1.0)
    rows = fit_series(fits, "matlab_simple")
    color, marker, label = color_of("matlab_simple")
    axes[0].semilogx([r["n"] for r in rows], [r["frac_ramanujan"] for r in rows],
                     ls="", marker=marker, ms=4.5, color=color, label=label)
    axes[0].axhline(F0, color=INK, lw=0.9, ls="--")
    axes[0].annotate(r"$F_1(0)=0.83191$", xy=(3e8, F0), xytext=(3e8, F0 - 0.012),
                     fontsize=7, color=INK, ha="right")
    axes[0].set_xlabel("$V$")
    axes[0].set_ylabel(r"$P(\lambda_{\mathrm{new}}\leq 2\sqrt{3})$")
    axes[0].set_title("Data (markers) and $F_1(a/(c(4)V^{1/3}))$ (lines)", fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="upper right")
    axes[0].set_ylim(0.80, 0.98)
    # right: how large V must be
    ax = axes[1]
    f0 = float(T.pdf(1, 0))
    for fam, k, a in models[:2]:
        color, marker, label = color_of(fam, k)
        dev = np.abs(T.cdf(1, a / (C4 * Vgrid ** (1 / 3))) - F0)
        ax.loglog(Vgrid, dev, color=color, lw=1.2, label=label)
    for fam, k in (("simple", None), ("matlab_simple", None),
                   ("matlab_loop_cover", None), ("k5_cover", None)):
        color, marker, _ = color_of(fam, k)
        rows = fit_series(fits, fam, k)
        ax.loglog([r["n"] for r in rows],
                  [abs(r["frac_ramanujan"] - F0) for r in rows],
                  ls="", marker=marker, ms=4, color=color)
    ax.axhline(1e-3, color=MUTED, lw=0.8)
    ax.axhline(1e-2, color=MUTED, lw=0.8)
    ax.set_xlabel("$V$")
    ax.set_ylabel(r"$|P(\mathrm{Ramanujan})-F_1(0)|$")
    ax.set_title(r"Distance to $F_1(0)$, decaying like $V^{-1/3}$", fontsize=8.5)
    ax.set_ylim(1e-5, 0.3)
    fig.subplots_adjust(wspace=0.32)
    return fig


# ---------------------------------------------------------------------------
# F. The origin of the 52% conjecture, quantitatively
# ---------------------------------------------------------------------------

def fig_mns_model(rows_summary, a_star):
    A = -bf.TW_MEAN[1] * C4
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axes[0]
    Vg = np.logspace(2, 6.5, 300)
    gap_model = A * Vg ** (-2 / 3) + a_star / Vg
    local = -(2 / 3 * A * Vg ** (-2 / 3) + a_star / Vg) / gap_model
    ax.semilogx(Vg, local, color=INK, lw=1.1,
                label=rf"model $|\mu_{{TW_1}}|c(4)V^{{-2/3}}+{a_star:.2f}/V$")
    for fam in ("simple", "matlab_simple"):
        data = bf.series_rows(rows_summary, fam, None)
        V = np.array([r["V"] for r in data])
        gap = np.array([r["gap"] for r in data])
        color, marker, label = color_of(fam)
        slope = np.diff(np.log(gap)) / np.diff(np.log(V))
        mid = np.sqrt(V[1:] * V[:-1])
        ax.semilogx(mid, slope, ls="", marker=marker, ms=4.5, color=color,
                    label=label)
    ax.axhline(-2 / 3, color=MUTED, lw=0.9, ls="--")
    ax.set_ylim(-0.92, -0.65)
    ax.set_xlabel("$V$ (geometric midpoint of consecutive sizes)")
    ax.set_ylabel("local exponent of the mean gap")
    ax.set_title("Local exponent of the mean gap", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    Vmax = np.logspace(3, 7, 60)
    data = bf.series_rows(rows_summary, "simple", None)
    V = np.array([r["V"] for r in data])
    gap = np.array([r["gap"] for r in data])
    glob_data = np.polyfit(np.log(V), np.log(gap), 1)[0]
    slopes = []
    for vm in Vmax:
        grid = np.logspace(2, np.log10(vm), 12)
        slopes.append(np.polyfit(np.log(grid),
                                 np.log(A * grid ** (-2 / 3) + a_star / grid), 1)[0])
    ax.semilogx(Vmax, slopes, color=INK, lw=1.1,
                label="single power law fitted to the model\nover $100\\leq V\\leq V_{\\max}$")
    ax.semilogx([20000], [glob_data], marker="o", color=color_of("simple")[0],
                ls="", ms=6, label=f"our data, $V\\leq 20{{,}}000$: {glob_data:.3f}")
    ax.axvspan(100, 20000, color=GRID, lw=0, zorder=0)
    ax.annotate("sizes of MNS", xy=(1400, -0.815), fontsize=7, color=INK_2,
                ha="center")
    ax.axhline(-2 / 3, color=MUTED, lw=0.9, ls="--")
    ax.set_xlabel(r"largest size in the fit, $V_{\max}$")
    ax.set_ylabel("fitted exponent of the mean gap")
    ax.set_ylim(-0.83, -0.65)
    ax.set_title("Exponent of a single fitted power law", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.subplots_adjust(wspace=0.35)
    return fig


# ---------------------------------------------------------------------------
# G. Discriminating power of the mass left of the mean, against N
# ---------------------------------------------------------------------------

def fig_power():
    N = np.logspace(2, 8, 200)
    from dataset_utils import TW_MASS_LEFT_OF_MEAN as M
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    se = np.sqrt(M[1] * (1 - M[1]) / N)
    pairs = [(M[1] - M[2], r"$TW_1$ against $TW_2$", LAW_COLOR["tw2"], "--"),
             (M[1] - M[4], r"$TW_1$ against $TW_4$", LAW_COLOR["tw4"], "-."),
             (M[1] - 0.5, r"$TW_1$ against the normal", MUTED, ":")]
    for delta, label, color, ls in pairs:
        ax.loglog(N, delta / se, color=color, ls=ls, lw=1.4, label=label)
    ax.axhline(3, color=INK, lw=0.8)
    ax.annotate("three standard errors", xy=(1.3e2, 3.3), fontsize=7, color=INK)
    for n0, text in ((1000, "MNS, $10^3$\nper size"),
                     (1e5, "MNS at three\nsizes, $10^5$"),
                     (5e6, "this paper,\n$5\\times10^6$")):
        ax.axvline(n0, color=MUTED, lw=0.8)
        ax.annotate(text, xy=(n0 * 1.1, 0.08), fontsize=6.5, color=INK_2)
    ax.set_xlabel("samples per size $N$")
    ax.set_ylabel("separation in binomial standard errors")
    ax.set_title("How many samples separate the candidate laws\n"
                 "by their mass left of the mean")
    ax.set_ylim(0.05, 300)
    ax.legend(fontsize=7, loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# H. The multigraph model: same law, rank two defects
# ---------------------------------------------------------------------------

def fig_multigraph(fits):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    t = np.linspace(-5, 3, 801)
    F1 = T.cdf(1, t)
    ax = axes[0]
    for fam in ("matlab_simple", "matlab_loop_cover"):
        s, _, c, n = load_s(record_for(fam, 500000))
        s.sort()
        color, marker, label = color_of(fam)
        ax.plot(t, ecdf_on(s, t) - F1, color=color, lw=1.2, label=label)
        del s
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhspan(-1.36 / np.sqrt(1e5), 1.36 / np.sqrt(1e5), color=GRID, lw=0,
               zorder=0)
    ax.set_xlabel(r"$s=(\lambda_2-2\sqrt{3})\,V^{2/3}/c(4)$,  $V=500{,}000$")
    ax.set_ylabel(r"$F_V(s)-F_1(s)$, nothing fitted")
    ax.set_title("Bulk, nothing fitted (shaded: 95% KS band)", fontsize=8.5)
    ax.set_ylim(-0.011, 0.034)
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    rng = np.random.default_rng(11)
    color, marker, label = color_of("matlab_loop_cover")
    for V in (50000, 100000, 200000, 500000):
        x = np.load(record_for("matlab_loop_cover", V)["path"])
        out = x[x > RHO4 + 8 * 1.0558 * V ** (-2 / 3)]
        ax.semilogx(V * rng.uniform(0.9, 1.1, out.size), out, ls="",
                    marker=marker, ms=5, color=color)
        del x
    ax.axhline(ATOM_TRIPLE, color=INK_2, lw=0.9, ls=(0, (1, 2)))
    ax.axhline(ATOM_TRIANGLE_LOOP, color=INK_2, lw=0.9, ls=(0, (1, 2)))
    ax.axhline(ATOM_LOOP_DOUBLE, color=INK_2, lw=0.9, ls=(0, (1, 2)))
    ax.axhline(RHO4, color=INK, lw=0.9, ls="--")
    ax.annotate(r"$7/2$: triple edge, or adjacent loops", xy=(4.2e4, 3.503),
                fontsize=6.5, color=INK_2)
    ax.annotate(r"$\lambda^3=8\lambda+16$: loop on a double edge",
                xy=(4.2e4, ATOM_LOOP_DOUBLE + 0.004), fontsize=6.5, color=INK_2)
    ax.annotate("triangle with a loop", xy=(4.2e4, ATOM_TRIANGLE_LOOP + 0.003),
                fontsize=6.5, color=INK_2)
    ax.annotate(r"$2\sqrt{3}$", xy=(6.5e5, RHO4 - 0.006), fontsize=6.5, color=INK,
                ha="right")
    ax.set_xticks([5e4, 1e5, 2e5, 5e5])
    ax.set_xticklabels(["$5\\cdot10^4$", "$10^5$", "$2\\cdot10^5$", "$5\\cdot10^5$"])
    ax.minorticks_off()
    ax.set_ylim(3.455, 3.56)
    ax.set_xlabel("$V$")
    ax.set_ylabel(r"samples $8$ std above $2\sqrt{3}$")
    ax.set_title("Far outliers and the rank two defects", fontsize=8.5)
    fig.subplots_adjust(wspace=0.32)
    return fig


# ---------------------------------------------------------------------------
# I. Tails by symmetry class
# ---------------------------------------------------------------------------

def fig_tails_by_class(fits):
    """Tails of the data and of the three laws, all standardized to mean 0
    and variance 1, so that the comparison is between shapes. The data is
    first mapped onto its own law by the affine quantile fit (a robust
    standardization that the thin far tails cannot move)."""
    picks = [("simple", 20000, None, "tw1", r"simple $4$-regular, $V=20{,}000$"),
             ("abelian_cover", 5000, 3, "tw2", r"$\mathbb{Z}_3$ covers, $n=5000$"),
             ("quaternion", 5000, 4, "tw4", r"quaternion covers, $n=5000$")]
    mom = {b: T.moments(b)[:2] for b in (1, 2, 4)}
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4), sharey="row")
    lo = np.linspace(-5.5, -1.5, 200)
    hi = np.linspace(1.0, 7.5, 200)
    for col, (fam, n, k, law, title) in enumerate(picks):
        s, _, _, _ = load_s(record_for(fam, n, k))
        fit = next(r for r in fit_series(fits, fam, k) if r["n"] == n)
        b_own = int(law[-1])
        mu, sd = mom[b_own]
        z = np.sort(((s - fit["alpha"]) / fit["scale_ratio"] - mu) / sd)
        del s
        N = z.size
        color = color_of(fam, k)[0]
        emp_lo = ecdf_on(z, lo)
        emp_hi = 1 - ecdf_on(z, hi)
        axes[0, col].semilogy(lo, np.where(emp_lo > 0, emp_lo, np.nan),
                              color=color, lw=2.6, alpha=0.5, label="data")
        axes[1, col].semilogy(hi, np.where(emp_hi > 0, emp_hi, np.nan),
                              color=color, lw=2.6, alpha=0.5, label="data")
        for b in (1, 2, 4):
            key = f"tw{b}"
            m, d = mom[b]
            axes[0, col].semilogy(lo, np.exp(T.log_cdf(b, m + d * lo)),
                                  color=LAW_COLOR[key], ls=LAW_LS[key], lw=1.0,
                                  label=LAW_NAME[key])
            axes[1, col].semilogy(hi, np.exp(T.log_sf(b, m + d * hi)),
                                  color=LAW_COLOR[key], ls=LAW_LS[key], lw=1.0,
                                  label=LAW_NAME[key])
        for row in (0, 1):
            axes[row, col].axhline(1 / N, color=MUTED, lw=0.7)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].set_ylim(1e-7, 0.3)
        axes[1, col].set_ylim(1e-7, 0.3)
        axes[1, col].set_xlabel("standardized $t$")
    axes[0, 0].set_ylabel(r"left tail $P(Z\leq t)$")
    axes[1, 0].set_ylabel(r"right tail $P(Z>t)$")
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Standardized tails select the symmetry class"
                 " (gray line: $1/N$)", y=0.995, fontsize=9)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# J. The KS matrix: every family against every candidate law
# ---------------------------------------------------------------------------

def _affine_ks(s, law):
    """KS distance after the affine quantile fit of s to the law."""
    if law == "normal":
        from scipy import stats as sps
        xq = sps.norm.ppf(fs.PROBS)
        beta, alpha = np.polyfit(xq, np.quantile(s, fs.PROBS), 1)
        u = np.sort((s - alpha) / beta)
        F = sps.norm.cdf(u)
        n = u.size
        return max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    xq = fs.law_quantile(law, fs.PROBS)
    beta, alpha = np.polyfit(xq, np.quantile(s, fs.PROBS), 1)
    return fs.ks_distance(np.sort((s - alpha) / beta), law)


def fig_ks_matrix():
    families = [("simple", 20000, None, "simple $4$-regular"),
                ("k5_cover", 10000, None, "$K_5$ covers"),
                ("k5_minus_edge", 10000, None, "$K_5-e$ covers"),
                ("abelian_cover", 5000, 3, r"$\mathbb{Z}_3$ covers"),
                ("quaternion", 5000, 4, "quaternion covers"),
                ("abelian_cover", 5000, 4, r"$\mathbb{Z}_4$ covers"),
                ("abelian_cover", 5000, 5, r"$\mathbb{Z}_5$ covers")]
    laws = ["tw1", "tw2", "tw4", "max12", "max22", "normal"]
    M = np.zeros((len(families), len(laws)))
    for i, (fam, n, k, _) in enumerate(families):
        s, _, _, _ = load_s(record_for(fam, n, k))
        for j, law in enumerate(laws):
            M[i, j] = _affine_ks(s, law)
        del s
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    logM = np.log10(M)
    im = ax.imshow(logM, cmap="Blues_r", vmin=-3.4, vmax=-1.5, aspect="auto")
    for i in range(M.shape[0]):
        best = np.argmin(M[i])
        for j in range(M.shape[1]):
            txt = f"{M[i, j] * 1e3:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if logM[i, j] < -2.6 else INK,
                    fontweight="bold" if j == best else "normal")
    ax.set_xticks(range(len(laws)))
    ax.set_xticklabels([LAW_NAME[l] for l in laws], fontsize=7, rotation=20,
                       ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f[3] for f in families], fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log_{10}$ KS distance", fontsize=7)
    ax.set_title("KS distance ($\\times10^{3}$) after the best affine fit to"
                 " each law,\nat the largest size; the minimum in each row is"
                 " bold", fontsize=8.5)
    return fig, M, families, laws


# ---------------------------------------------------------------------------
# K. The moment plane
# ---------------------------------------------------------------------------

def fig_moment_plane(rows_summary):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    targets = {
        "tw1": TW_SHAPE_MOMENTS[1], "tw2": TW_SHAPE_MOMENTS[2],
        "tw4": TW_SHAPE_MOMENTS[4],
        "max12": (MAX_LAW["max12"][3], MAX_LAW["max12"][4]),
        "max22": (MAX_LAW["max22"][3], MAX_LAW["max22"][4]),
    }
    for law, (sk, ku) in targets.items():
        ax.plot(sk, ku, marker="*", ms=12, color=INK, ls="", zorder=5)
        ax.annotate(LAW_NAME[law], xy=(sk, ku), xytext=(6, -3),
                    textcoords="offset points", fontsize=7, color=INK)
    fams = [("simple", None), ("abelian_cover", 3), ("abelian_cover", 4),
            ("abelian_cover", 5), ("quaternion", 4)]
    for fam, k in fams:
        data = bf.series_rows(rows_summary, fam, k)
        sk = np.array([r["skewness"] for r in data])
        ku = np.array([r["ex_kurtosis"] for r in data])
        color, marker, label = color_of(fam, k)
        ax.plot(sk, ku, marker=marker, ms=3.5, lw=1.0, color=color, label=label)
        ax.annotate("", xy=(sk[-1], ku[-1]), xytext=(sk[-2], ku[-2]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
    ax.set_xlabel("sample skewness")
    ax.set_ylabel("sample excess kurtosis")
    ax.set_title("Each family moves, as $n$ grows, to the point of its limit law")
    ax.legend(fontsize=6.5, loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# L. Parameter-free comparison for the covers
# ---------------------------------------------------------------------------

def fig_paramfree_covers():
    panels = [("abelian_cover", 3, "tw2", r"$\mathbb{Z}_3$ against $TW_2$"),
              ("quaternion", 4, "tw4", r"quaternion against $2^{-1/6}TW_4$"),
              ("abelian_cover", 5, "max22", r"$\mathbb{Z}_5$ against $\max(TW_2,TW_2')$"),
              ("abelian_cover", 4, "max12", r"$\mathbb{Z}_4$ against $\max(TW_1,TW_2)$"),
              ("k5_cover", None, "tw1", r"$K_5$ against $TW_1$"),
              ("k5_minus_edge", None, "tw1",
               r"$K_5-e$ against $TW_1$, predicted $c_B$")]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.2), sharey=True)
    t = np.linspace(-5, 3, 801)
    shades = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
    for ax, (fam, k, law, title) in zip(axes.flat, panels):
        recs = [r for r in discover_datasets() if r["family"] == fam
                and (k is None or r["cover_deg"] == k or fam in ("k5_cover", "k5_minus_edge"))]
        recs = sorted(recs, key=lambda r: r["V"])
        recs = [recs[i] for i in np.linspace(0, len(recs) - 1, 5).round().astype(int)]
        Flaw = fs.law_cdf(law, t)
        for rec, color in zip(recs, shades):
            x = np.load(rec["path"])
            s, _, c, n = fs.rescale(rec, x)
            if fam == "k5_minus_edge":
                s = s * c / PRED_C[fam]
            del x
            s.sort()
            ax.plot(t, ecdf_on(s, t) - Flaw, color=color, lw=1.1,
                    label=f"$n={n:,}$".replace(",", "{,}"))
            del s
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=7.5)
        ax.legend(fontsize=5.5, loc="upper right", ncol=1, handlelength=1.2,
                  borderpad=0.3, labelspacing=0.2)
    for ax in axes[1]:
        ax.set_xlabel("$s$, nothing fitted")
    axes[0, 0].set_ylabel("$F_n(s)-F(s)$")
    axes[1, 0].set_ylabel("$F_n(s)-F(s)$")
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# M. The maximum laws: KS against the pure laws and the maximum law
# ---------------------------------------------------------------------------

def fig_maxlaw_ks():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, (k, own, title) in zip(axes, ((4, "max12", r"$\mathbb{Z}_4$ covers"),
                                          (5, "max22", r"$\mathbb{Z}_5$ covers"))):
        recs = sorted([r for r in discover_datasets()
                       if r["family"] == "abelian_cover" and r["cover_deg"] == k],
                      key=lambda r: r["base_size"])
        ns = [r["base_size"] for r in recs]
        res = {law: [] for law in ("tw1", "tw2", "tw4", own)}
        for rec in recs:
            s, _, _, _ = load_s(rec)
            for law in res:
                res[law].append(_affine_ks(s, law))
            del s
        for law in ("tw1", "tw2", "tw4"):
            ax.loglog(ns, res[law], color=LAW_COLOR[law], ls=LAW_LS[law],
                      marker="o", ms=3, lw=1.1, label=LAW_NAME[law])
        ax.loglog(ns, res[own], color=INK, lw=1.6, marker="s", ms=3.5,
                  label="its maximum law")
        ax.axhline(0.87 / np.sqrt(1e6), color=MUTED, lw=0.8)
        ax.set_title(f"{title} (gray: typical KS at $N=10^6$)", fontsize=8.5)
        ax.set_xlabel("base size $n$")
    axes[0].set_ylabel("KS distance after the best affine fit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(wspace=0.08, bottom=0.22)
    return fig


# ---------------------------------------------------------------------------
# N. The universal cover densities and their square root edges
# ---------------------------------------------------------------------------

def fig_universal_cover_density():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    bases = [("k5_cover", U.K5, RHO4, "$K_5$ (Kesten–McKay, $d=4$)"),
             ("k5_minus_edge", U.K5_MINUS_E, RHO_K5_MINUS_E, "$K_5-e$"),
             ("k4_minus_edge", U.K4_MINUS_E, RHO_K4_MINUS_E, "$K_4-e$")]
    for fam, E, rho, label in bases:
        color, marker, _ = color_of(fam)
        x = np.linspace(-rho + 1e-3, rho - 1e-4, 700)
        dens = U.density(E, x)
        axes[0].plot(x, dens, color=color, lw=1.2, label=label)
        axes[0].axvline(rho, color=color, lw=0.7, ls=":")
        delta = np.logspace(-6, -0.3, 60)
        d2 = U.density(E, rho - delta)
        axes[1].loglog(delta, np.pi * d2 / np.sqrt(delta), color=color, lw=1.2)
        axes[1].annotate(f"{label.split(' (')[0]}: $C_B={EDGE_C[fam]:.4f}$",
                         xy=(1.3e-6, EDGE_C[fam] * 1.12), fontsize=6.5,
                         color=INK_2)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel(r"density of $\mu_B$")
    axes[0].set_title("Limiting density $\\mu_B$ (dotted: $\\rho(B)$)", fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[1].set_xlabel(r"$\rho(B)-x$")
    axes[1].set_ylabel(r"$\pi\,\mu_B'(x)/\sqrt{\rho(B)-x}$")
    axes[1].set_title("The square root edge coefficient $C_B$", fontsize=8.5)
    axes[1].set_ylim(0.5, 25)
    fig.subplots_adjust(wspace=0.3)
    return fig


def fig_irregular_amplitude(fits):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for fam in ("k5_cover", "k5_minus_edge", "k4_minus_edge"):
        rows = fit_series(fits, fam)
        color, marker, label = color_of(fam)
        V = np.array([r["n"] for r in rows])
        cf = np.array([r["c_fit"] for r in rows])
        ax.semilogx(V, cf / PRED_C[fam], marker=marker, ms=5, lw=1.1,
                    color=color, label=f"{label}: prediction {PRED_C[fam]:.4f}")
    ax.axhline(1, color=INK, lw=0.9)
    ax.set_xlabel("$V$")
    ax.set_ylabel(r"fitted $c_B$ / $C_B^{-2/3}$")
    ax.set_title("The fixed base amplitudes approach the universal\n"
                 "cover prediction, with nothing fitted to the data")
    ax.legend(fontsize=6.5, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# O. The leading defects, drawn
# ---------------------------------------------------------------------------

def _stub(ax, p, angle, length=0.28):
    x, y = p
    ax.plot([x, x + length * np.cos(angle)], [y, y + length * np.sin(angle)],
            color=MUTED, lw=0.9, ls=(0, (2, 1.5)), zorder=0)


def fig_defect_shapes():
    fig, axes = plt.subplots(1, 6, figsize=(6.8, 1.9))
    # bouquet: triple edge
    ax = axes[0]
    pos = {0: (-0.5, 0), 1: (0.5, 0)}
    _draw_graph(ax, pos, [(0, 1)], curved=[(0, 1, 0.5), (0, 1, -0.5)], ms=4)
    _stub(ax, pos[0], np.pi)
    _stub(ax, pos[1], 0)
    ax.set_title("triple\nedge\n$r=2$, $7/2$", fontsize=7)
    # adjacent loops
    ax = axes[1]
    _draw_graph(ax, pos, [(0, 1)], loops=[(0, np.pi / 2, 0.22), (1, np.pi / 2, 0.22)],
                ms=4)
    _stub(ax, pos[0], np.pi)
    _stub(ax, pos[1], 0)
    ax.set_title("adjacent\nloops\n$r=2$, $7/2$", fontsize=7)
    # loop on a double edge
    ax = axes[2]
    _draw_graph(ax, pos, [], curved=[(0, 1, 0.4), (0, 1, -0.4)],
                loops=[(0, np.pi, 0.22)], ms=4)
    _stub(ax, pos[1], np.pi / 4)
    _stub(ax, pos[1], -np.pi / 4)
    ax.set_title("loop on a\ndouble edge\n$r=2$, $3.53858$", fontsize=7)
    # K4 in K5 - e (and in K5)
    ax = axes[3]
    sq = {0: (-0.5, -0.5), 1: (0.5, -0.5), 2: (0.5, 0.5), 3: (-0.5, 0.5)}
    _draw_graph(ax, sq, U.K4, ms=4)
    for v, ang in zip(range(1, 4), (-np.pi / 4, np.pi / 4, 3 * np.pi / 4)):
        _stub(ax, sq[v], ang)
    ax.set_title("$K_4$ in\n$K_5-e$\n$r=3$, $3.33913$", fontsize=7)
    # thetas in K4 - e
    for ax, (a, b, c), val in ((axes[4], (2, 2, 4), 2.51185),
                               (axes[5], (1, 2, 5), 2.51369)):
        L, R = (-0.8, 0), (0.8, 0)
        pos = {"L": L, "R": R}
        edges = []
        for route, (length, height) in enumerate(zip((a, b, c), (0.0, 0.6, -0.6))):
            prev = "L"
            for i in range(1, length):
                name = f"{route}_{i}"
                xx = L[0] + (R[0] - L[0]) * i / length
                yy = height * np.sin(np.pi * i / length) * 1.3
                pos[name] = (xx, yy)
                edges.append((prev, name))
                prev = name
            edges.append((prev, "R"))
        _draw_graph(ax, pos, edges, ms=3)
        ax.set_title(rf"$\theta({a},{b},{c})$" "\n" r"in $K_4-e$" "\n" rf"$r=2$, ${val}$",
                     fontsize=7)
    for ax in axes:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.0, 1.0)
    fig.subplots_adjust(wspace=0.1)
    return fig


# ---------------------------------------------------------------------------

def make_all(rows_summary=None, save=None):
    """save: the caller's save function, so that the figures are listed in
    the images/README.md that build_figures.main() writes."""
    if save is not None:
        bf.save = save
    bf.setup_style()
    if rows_summary is None:
        rows_summary = bf.load_summary()
    fits = load_fits()
    bf.save(fig_base_graphs(), "base_graphs_and_a_cover.png",
            "The four base graphs of the dataset with the spectral radii of "
            "their universal covers, and a random 3-cover of K4-e drawn fiber "
            "by fiber.")
    bf.save(fig_tw_laws(), "tracy_widom_laws_and_ramanujan_functionals.png",
            "Left: the Tracy-Widom densities (original normalization) with "
            "the mass right of 0 shaded, 1 - F_beta(0). Right: the maximum "
            "laws of the Z4 and Z5 covers with their values at 0.")
    fig, a_star = fig_paramfree_simple(fits)
    bf.save(fig, "paramfree_simple_regular.png",
            "Parameter-free comparison of the simple 4-regular data with the "
            "proven law: F_V(s) - F_1(s) for s = (lambda_2 - 2sqrt3)V^(2/3)/c(4). "
            "Left: raw, offset by a shift decaying with V. Right: after one "
            "shift a/V with a single constant a for every size.")
    bf.save(fig_scale_location(fits), "finite_size_scale_and_location.png",
            "Quantile regression of every dataset on its conjectured limit "
            "law: left, the fitted scale over the predicted scale (c(4), "
            "c(4)2^(-1/6) for quaternion covers, and the universal cover "
            "prediction for K5-e and K4-e); right, the location coefficient "
            "a with lambda ~ rho - a/n + c n^(-2/3) X.")
    bf.save(fig_ramanujan_rate(fits), "ramanujan_probability_rate.png",
            "Empirical Ramanujan probability against V with the one "
            "parameter model F_1(a/(c(4) V^(1/3))) (left), and the distance "
            "to F_1(0) on log-log axes (right), extrapolated to V = 1e9.")
    bf.save(fig_mns_model(rows_summary, a_star), "mns_exponent_model.png",
            "Left: local exponents of the mean gap for the simple families "
            "with the two term model. Right: the exponent a single power law "
            "reports when fitted over 100 <= V <= Vmax, against Vmax.")
    bf.save(fig_power(), "discriminating_power_vs_samples.png",
            "Separation in binomial standard errors of the TW1 mass left of "
            "the mean from TW2, TW4 and the normal, against the number of "
            "samples N, with the sample sizes of MNS and of this paper.")
    bf.save(fig_multigraph(fits), "multigraph_model_bulk_and_atoms.png",
            "The permutation multigraph model at V=500,000: parameter-free "
            "CDF difference to TW1 (left), and every sample above "
            "2sqrt3+0.01 against V with the rank two defect eigenvalues 7/2 "
            "and the root of lambda^3 = 8 lambda + 16 (right).")
    bf.save(fig_tails_by_class(fits), "tails_by_symmetry_class.png",
            "Left and right tails, on a log scale, of the simple 4-regular, "
            "Z3 cover and quaternion cover data after the affine quantile "
            "fit, against TW1, TW2 and TW4.")
    fig, M, families, laws = fig_ks_matrix()
    bf.save(fig, "ks_matrix_families_vs_laws.png",
            "KS distance after the best affine fit, every family at its "
            "largest size against every candidate law.")
    with open(os.path.join(REPO_ROOT, "images", "ks_matrix_values.csv"), "w") as fh:
        fh.write("family," + ",".join(laws) + "\n")
        for (fam, n, k, label), row in zip(families, M):
            fh.write(f"{fam}{'' if k is None else k}_{n}," +
                     ",".join(f"{v:.6f}" for v in row) + "\n")
    bf.save(fig_moment_plane(rows_summary), "moment_plane_trajectories.png",
            "Skewness against excess kurtosis for the simple, cyclic and "
            "quaternion families, moving with n to the points of their limit "
            "laws.")
    bf.save(fig_paramfree_covers(), "paramfree_covers.png",
            "Parameter-free CDF differences for six cover families against "
            "their conjectured limit laws, at five sizes each.")
    bf.save(fig_maxlaw_ks(), "maximum_law_ks.png",
            "KS distance after the best affine fit of the Z4 and Z5 families "
            "to the pure Tracy-Widom laws and to their maximum laws.")
    bf.save(fig_universal_cover_density(), "universal_cover_densities.png",
            "The limiting spectral densities of the new eigenvalues for "
            "covers of K5, K5-e and K4-e, and their square root edge "
            "coefficients C_B.")
    bf.save(fig_irregular_amplitude(fits), "irregular_amplitude_vs_prediction.png",
            "Fitted amplitudes of the fixed base families over the universal "
            "cover prediction C_B^(-2/3).")
    bf.save(fig_defect_shapes(), "defect_shapes.png",
            "The leading localized defects: the three cycle rank two defects "
            "of the multigraph model, K4 in covers of K5 and K5-e, and the two "
            "shortest thetas in covers of K4-e, with their cycle ranks and "
            "defect eigenvalues. Dashed stubs are attached branches of the "
            "universal cover.")


if __name__ == "__main__":
    make_all()
