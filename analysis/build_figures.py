"""Builds all paper figures from data_summary.csv and the raw .npy files,
saving them into images/ with self-describing filenames, plus an
images/README.md explaining every figure.

Run from the repo root, after analysis/build_summary.py:
    python3 analysis/build_figures.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import (  # noqa: E402
    ATOMS_K5, ATOMS_K5_MINUS_E, F_BETA_AT_0, MAX_LAW, RAMANUJAN_THRESHOLD,
    REPO_ROOT, RHO_K4_MINUS_E, RHO_K5_MINUS_E, TW_MEAN, TW_STD, TW1_MEAN_OVER_STD,
    size_scaling_prefactor, tw, tw_cdf_standardized, tw_pdf_standardized,
    tw_quantile_standardized, tw_shape_moments,
)

IMAGES_DIR = os.path.join(REPO_ROOT, "images")
CSV_PATH = os.path.join(REPO_ROOT, "data_summary.csv")

# ---------------------------------------------------------------------------
# Style: validated categorical palette (fixed slot order), recessive chrome.
# Identity is never color-alone: every series also has a fixed marker.
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# (family, cover_deg, label, color, marker) — color follows the entity in
# every figure; slots are never reassigned when a figure shows fewer series.
SERIES = [
    ("simple", None, "4-regular simple (Python)", "#2a78d6", "o"),
    ("matlab_simple", None, "4-regular simple (MATLAB)", "#1baf7a", "s"),
    ("matlab_loop_cover", None, "4-regular perm. model (MATLAB)", "#eda100", "^"),
    ("abelian_cover", 3, "Abelian cover, $k=3$", "#008300", "D"),
    ("abelian_cover", 4, "Abelian cover, $k=4$", "#4a3aa7", "v"),
    ("abelian_cover", 5, "Abelian cover, $k=5$", "#e34948", "P"),
    ("quaternion", 4, "Quaternion cover", "#e87ba4", "X"),
    ("k5_cover", None, "$K_5$ cover", "#0d8ba3", "h"),
    ("k5_minus_edge", None, "$K_5-e$ cover", "#8a5a2b", "*"),
    ("k4_minus_edge", None, "$K_4-e$ cover", "#5e548e", "d"),
]

# Distribution-curve palette (used in PDF/CDF/KS figures): same fixed slots.
LAW_STYLE = {
    "tw1": ("Tracy–Widom $\\beta=1$ (GOE)", "#2a78d6", "-"),
    "tw2": ("Tracy–Widom $\\beta=2$ (GUE)", "#1baf7a", "--"),
    "tw4": ("Tracy–Widom $\\beta=4$ (GSE)", "#eda100", "-."),
    "normal": ("standard normal", MUTED, ":"),
}

EXCEPTIONAL = [
    ("data/simple/simple_deg4_V20000_N5000000.npy",
     "simple_V20000_N5000000",
     "random simple 4-regular graphs, $V=20{,}000$, $N=5{,}000{,}000$"),
    ("data/matlab/simple_deg4_V500000_N100000.npy",
     "matlab_simple_V500000_N100000",
     "random simple 4-regular graphs, $V=500{,}000$, $N=100{,}000$ (MATLAB)"),
    ("data/irreg_covers/irreg_covers_k5_minus_edge_cover_V2000x5_N5000000.npy",
     "k5_minus_edge_V10000_N5000000",
     "random covers of $K_5-e$, $V=10{,}000$, $N=5{,}000{,}000$"),
    ("data/k5_covers_complete_cover/k5_covers_complete_cover_V2000x5_N5000000.npy",
     "k5_cover_V10000_N5000000",
     "random covers of $K_5$, $V=10{,}000$, $N=5{,}000{,}000$"),
]

FIGURES = {}  # filename -> description, for images/README.md


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        # opaque legend so curves never strike through legend text
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": GRID,
        # publication typography: Computer-Modern-style math and serif text
        # to match a LaTeX paper, no TeX installation required
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
        # figures are drawn at their final physical size (full text width
        # ~6.5in or single-plot ~5in), so fonts print at true point size
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_summary():
    rows = []
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            for key, value in row.items():
                if key in ("file", "family", "source"):
                    continue
                row[key] = float(value) if value != "" else None
            rows.append(row)
    return rows


def series_rows(rows, family, cover_deg):
    out = [r for r in rows if r["family"] == family
           and (cover_deg is None or r["cover_deg"] == cover_deg)]
    return sorted(out, key=lambda r: r["V"])


def loglog_slope(V, y):
    return np.polyfit(np.log(V), np.log(y), 1)[0]


def save(fig, fname, description):
    """Saves both a vector PDF (for the LaTeX paper) and a PNG (for
    browsing/GitHub) under the same basename."""
    path = os.path.join(IMAGES_DIR, fname)
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    FIGURES[fname] = description
    print(f"wrote images/{fname} (+.pdf)", flush=True)


# ---------------------------------------------------------------------------
# 1. Log-log scaling figures: mean gap and standard deviation vs V
# ---------------------------------------------------------------------------

def fig_scaling(rows, subset, title_suffix,
                stat_phrase="largest non-trivial eigenvalue",
                amplitude_conjectured=True, shared_legend=False):
    """shared_legend=True: no per-series slope labels; one legend for all
    series placed below both panels (used for the busy all-family figure,
    where in-panel 8-entry legends would cover the data)."""
    prefactor = size_scaling_prefactor()
    amp_gap = -TW_MEAN[1] * prefactor
    amp_std = TW_STD[1] * prefactor

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    all_V = []
    for family, k, label, color, marker in subset:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = np.array([r["V"] for r in data])
        gap = np.array([r["gap"] for r in data])
        std = np.array([r["std"] for r in data])
        all_V.extend(V)
        if shared_legend:
            label_gap_series = label_std_series = label
        else:
            label_gap_series = (
                f"{label}  (slope {loglog_slope(V, gap):.3f})")
            label_std_series = (
                f"{label}  (slope {loglog_slope(V, std):.3f})")
        axes[0].loglog(V, gap, marker=marker, ms=4, lw=1.2, color=color,
                       label=label_gap_series)
        axes[1].loglog(V, std, marker=marker, ms=4, lw=1.2, color=color,
                       label=label_std_series)

    # The amplitude |mu_TW1| c(4), sigma_TW1 c(4) is conjectured for plain
    # 4-regular graphs only; for cover families the dashed line is just a
    # V^(-2/3) slope guide (the data runs parallel, offset by a constant).
    if shared_legend:
        label_gap = label_std = (r"conjectured 4-regular TW$_1$ law"
                                 r" $\propto V^{-2/3}$")
    elif amplitude_conjectured:
        label_gap = r"conjecture $|\mu_{TW_1}|\,c(4)\,V^{-2/3}$"
        label_std = r"conjecture $\sigma_{TW_1}\,c(4)\,V^{-2/3}$"
    else:
        label_gap = label_std = (r"$V^{-2/3}$ slope guide"
                                 " (regular-graph amplitude)")
    Vline = np.array([min(all_V) / 1.3, max(all_V) * 1.3])
    axes[0].loglog(Vline, amp_gap * Vline ** (-2 / 3), "--", color=INK, lw=1,
                   label=label_gap)
    axes[1].loglog(Vline, amp_std * Vline ** (-2 / 3), "--", color=INK, lw=1,
                   label=label_std)

    axes[0].set_xlabel("number of vertices $V$")
    axes[0].set_ylabel(r"$\rho - \mathrm{mean}(\lambda)$")
    axes[0].set_title("Gap of the mean below the Ramanujan threshold"
                      r" $\rho$")
    axes[1].set_xlabel("number of vertices $V$")
    axes[1].set_ylabel(r"standard deviation of $\lambda$")
    axes[1].set_title("Standard deviation")
    if shared_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6.5)
    else:
        # legends (with per-panel fitted slopes) go below their panels:
        # at final column width any in-panel placement occludes data
        for ax in axes:
            ax.legend(fontsize=6.5, loc="upper center",
                      bbox_to_anchor=(0.5, -0.26))
    fig.suptitle(f"$V^{{-2/3}}$ scaling of the {stat_phrase}"
                 f" — {title_suffix}", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2. Compensated scaling: y * V^{2/3} should approach a constant
# ---------------------------------------------------------------------------

def fig_compensated(rows):
    prefactor = size_scaling_prefactor()
    amp_gap = -TW_MEAN[1] * prefactor
    amp_std = TW_STD[1] * prefactor

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = np.array([r["V"] for r in data])
        gap = np.array([r["gap"] for r in data])
        std = np.array([r["std"] for r in data])
        axes[0].semilogx(V, gap * V ** (2 / 3), marker=marker, ms=4, lw=1.2,
                         color=color, label=label)
        axes[1].semilogx(V, std * V ** (2 / 3), marker=marker, ms=4, lw=1.2,
                         color=color, label=label)

    # the dashed conjecture line gets a small in-panel legend of its own;
    # the seven series share one legend below the panels
    line_gap = axes[0].axhline(
        amp_gap, ls="--", color=INK, lw=1,
        label=rf"conjecture $|\mu_{{TW_1}}|\,c(4) = {amp_gap:.4f}$")
    line_std = axes[1].axhline(
        amp_std, ls="--", color=INK, lw=1,
        label=rf"conjecture $\sigma_{{TW_1}}\,c(4) = {amp_std:.4f}$")
    axes[0].legend(handles=[line_gap], fontsize=6.5, loc="upper right")
    axes[1].legend(handles=[line_std], fontsize=6.5, loc="upper right")
    handles = [ax_line for ax_line in axes[0].get_lines()
               if ax_line.get_label().startswith(
                   ("4-", "Abelian", "Quat", "$K_5"))]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               fontsize=6.5)
    axes[0].set_ylabel(r"$(\rho - \mathrm{mean}(\lambda))\cdot V^{2/3}$")
    axes[0].set_title("Compensated mean gap")
    axes[1].set_ylabel(r"$\mathrm{std}(\lambda)\cdot V^{2/3}$")
    axes[1].set_title("Compensated standard deviation")
    for ax in axes:
        ax.set_xlabel("number of vertices $V$")
    fig.suptitle("Compensated $V^{2/3}$ amplitudes — dashed: constants"
                 " conjectured for random 4-regular graphs", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2b. Local scaling exponents: the figure that explains the MNS discrepancy.
# MNS fitted a single power law over small V and got an exponent steeper
# than -2/3 for the mean gap, concluding it outpaces the std. The local
# (consecutive-size) exponent shows that steepness is a finite-size
# transient decaying toward -2/3.
# ---------------------------------------------------------------------------

def fig_local_exponent(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    fig.subplots_adjust(wspace=0.3)  # room for the right panel's y-label
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if len(data) < 2:
            continue
        V = np.array([r["V"] for r in data])
        for ax, key in ((axes[0], "gap"), (axes[1], "std")):
            y = np.array([r[key] for r in data])
            slopes = np.log(y[1:] / y[:-1]) / np.log(V[1:] / V[:-1])
            mid = np.sqrt(V[1:] * V[:-1])
            ax.semilogx(mid, slopes, marker=marker, ms=4, lw=1.2,
                        color=color, label=label)
    for ax in axes:
        ax.axhline(-2 / 3, ls="--", color=INK, lw=0.9,
                   label="conjectured exponent $-2/3$")
        ax.set_xlabel("number of vertices $V$ (geometric midpoint)")
    axes[0].set_ylabel("local exponent of $\\rho - \\mathrm{mean}$")
    axes[0].set_title("Mean gap")
    axes[1].set_ylabel("local exponent of $\\mathrm{std}$")
    axes[1].set_title("Standard deviation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6.5)
    fig.suptitle("Local log–log slopes between consecutive sizes — a single"
                 " power-law fit over small $V$ overestimates the gap"
                 " exponent", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2c. Moment-level convergence: skewness and excess kurtosis vs V
# ---------------------------------------------------------------------------

def fig_moments(rows):
    shape = {beta: tw_shape_moments(beta) for beta in (1, 2, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for family, k, label, color, marker in SERIES:
        if family in ("matlab_loop_cover", "k5_cover", "k5_minus_edge",
                      "k4_minus_edge"):
            # third/fourth moments contaminated by far-outlier tails
            # (multigraph: kurtosis ~10^4; fixed-base covers: see the
            # standardized-tail figure) -- the bulk statistics are in the
            # other figures
            continue
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = [r["V"] for r in data]
        axes[0].semilogx(V, [r["skewness"] for r in data], marker=marker,
                         ms=4, lw=1.2, color=color, label=label)
        axes[1].semilogx(V, [r["ex_kurtosis"] for r in data], marker=marker,
                         ms=4, lw=1.2, color=color, label=label)
    for beta, ls in ((1, "--"), (2, "-."), (4, ":")):
        skew_tw, kurt_tw = shape[beta]
        axes[0].axhline(skew_tw, ls=ls, color=INK, lw=0.9,
                        label=rf"TW$_{beta}$ skew. $\approx {skew_tw:.3f}$")
        axes[1].axhline(kurt_tw, ls=ls, color=INK, lw=0.9,
                        label=rf"TW$_{beta}$ ex. kurt. $\approx {kurt_tw:.3f}$")
    axes[0].set_ylabel("sample skewness")
    axes[0].set_title("Skewness")
    axes[1].set_ylabel("sample excess kurtosis")
    axes[1].set_title("Excess kurtosis")
    for ax in axes:
        ax.set_xlabel("number of vertices $V$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6.5)
    fig.suptitle("Third and fourth moments converge to the Tracy–Widom"
                 " values of each family's symmetry class (permutation"
                 " model omitted: kurtosis up to $\\sim 10^4$)", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2d. Right tail: simple vs permutation (multigraph) model at V = 500,000
# ---------------------------------------------------------------------------

def fig_tail_comparison():
    picks = [
        ("data/matlab/simple_deg4_V500000_N100000.npy",
         "4-regular simple (MATLAB)", "#1baf7a", "-"),
        ("data/matlab/loop_cover_deg4_V500000_N100000.npy",
         "4-regular perm. model (MATLAB)", "#eda100", "-"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for path, label, color, ls in picks:
        x = np.sort(np.load(os.path.join(REPO_ROOT, path)))
        n = x.size
        surv = 1.0 - np.arange(1, n + 1) / n
        # decimate the bulk but keep the full tail resolution
        keep = np.unique(np.concatenate([
            np.linspace(0, n - 2, 1500).astype(int),
            np.arange(max(0, n - 2000), n - 1),
        ]))
        ax.semilogy(x[keep], surv[keep], ls=ls, color=color, lw=1.3,
                    label=label)
        del x
    ax.axvline(RAMANUJAN_THRESHOLD, ls="--", color=INK, lw=0.9,
               label=r"Ramanujan threshold $2\sqrt{3}$")
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"empirical $P(\lambda > x)$")
    ax.set_ylim(5e-6, 1.1)
    ax.set_title("Right tail at $V = 500{,}000$: the multigraph model has"
                 "\nfar outliers past the Ramanujan threshold")
    ax.legend(fontsize=7, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# 2d'. Standardized right tail of the fixed-base cover families at V=10,000:
# the bulk collapses onto TW1 while K5-e (and faintly K5) carries a thin
# far-outlier tail at frequency ~1e-5, which dominates the third and fourth
# moments without moving KS-level statistics.
# ---------------------------------------------------------------------------

def fig_tail_k5():
    picks = [
        ("data/simple/simple_deg4_V10000_N5000000.npy",
         "4-regular simple (Python)", "#2a78d6"),
        ("data/k5_covers_complete_cover/"
         "k5_covers_complete_cover_V2000x5_N5000000.npy",
         "$K_5$ cover", "#0d8ba3"),
        ("data/irreg_covers/"
         "irreg_covers_k5_minus_edge_cover_V2000x5_N5000000.npy",
         "$K_5-e$ cover", "#8a5a2b"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    # TW1 reference where the package cdf is trustworthy (survival >= 1e-4)
    t_ref = np.linspace(-1.0, 4.2, 400)
    surv_ref = 1.0 - tw_cdf_standardized(1)(t_ref)
    ax.semilogy(t_ref, surv_ref, ls="--", color=INK, lw=1.0,
                label="Tracy\u2013Widom $\\beta=1$")
    for path, label, color in picks:
        x = np.load(os.path.join(REPO_ROOT, path))
        z = np.sort((x - x.mean()) / x.std(ddof=1))
        del x
        n = z.size
        surv = 1.0 - np.arange(1, n + 1) / n
        keep = np.unique(np.concatenate([
            np.linspace(0, n - 2, 1500).astype(int),
            np.arange(max(0, n - 2000), n - 1),
        ]))
        ax.semilogy(z[keep], surv[keep], color=color, lw=1.2, label=label)
        del z
    ax.set_xlabel(r"standardized eigenvalue $(\lambda - \mathrm{mean})/\mathrm{std}$")
    ax.set_ylabel(r"empirical $P(\lambda > x)$")
    ax.set_ylim(1e-7, 1.1)
    ax.set_title("Standardized right tail at $V = 10{,}000$: thin far-outlier"
                 "\ntails for the fixed-base cover families")
    ax.legend(fontsize=7, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# 2f. Outlier atoms: every far-outlier sample of the fixed-base families
# against V, with the computed defect eigenvalues overlaid. The outliers
# lock onto the atoms; frequencies decay like k^(1-r) by cycle rank.
# ---------------------------------------------------------------------------

def fig_outlier_atoms():
    panels = [
        ("irreg_covers", "irreg_covers_k5_minus_edge_cover_V{k}x5_N5000000.npy",
         RHO_K5_MINUS_E, r"$\rho(K_5-e)$", ATOMS_K5_MINUS_E, "#8a5a2b",
         "covers of $K_5-e$"),
        ("k5_covers_complete_cover",
         "k5_covers_complete_cover_V{k}x5_N5000000.npy",
         RAMANUJAN_THRESHOLD, r"$2\sqrt{3}$", ATOMS_K5, "#0d8ba3",
         "covers of $K_5$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    rng = np.random.default_rng(7)
    for ax, (subdir, tmpl, rho, rho_label, atoms, color, title) in zip(axes, panels):
        for k in (100, 200, 400, 1000, 2000):
            V = 5 * k
            x = np.load(os.path.join(REPO_ROOT, "data", subdir,
                                     tmpl.format(k=k)))
            cut = x.mean() + 8 * x.std(ddof=1)
            out = x[x > cut]
            del x
            jitter = rng.uniform(0.88, 1.12, out.size)
            ax.semilogx(V * jitter, out, ls="", marker="o", ms=2.4,
                        color=color, alpha=0.6, markeredgewidth=0)
        ax.axhline(rho, ls="--", color=INK, lw=0.9)
        ax.annotate(rho_label, xy=(1.01, rho),
                    xycoords=("axes fraction", "data"), fontsize=6,
                    color=INK, va="center")
        for name, value in atoms.items():
            ax.axhline(value, ls=(0, (1, 3)), color=INK_2, lw=0.9)
            label = (r"$\lambda_{K_4}$" if name.startswith("K4")
                     else "rank $4$")
            if not name.endswith("4a"):
                ax.annotate(f"{label}: {value:.5f}", xy=(1.01, value),
                            xycoords=("axes fraction", "data"), fontsize=6,
                            color=INK_2, va="center")
        ax.set_xlabel("number of vertices $V$")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(
        r"outlier eigenvalues $\lambda > \mathrm{mean} + 8\,\mathrm{std}$")
    fig.suptitle("Far outliers lock onto the localized defect eigenvalues"
                 " (dotted); frequencies decay like $k^{1-r}$", y=1.02)
    fig.subplots_adjust(wspace=0.55)
    return fig


# ---------------------------------------------------------------------------
# 2g. The approach to the limit: the deviation of the standardized ECDF
# from TW1 is proportional to a fixed Edgeworth-type profile (t^2-1)f(t)
# (a skewness deficit), with amplitude decaying like a power of V.
# ---------------------------------------------------------------------------

def fig_edgeworth():
    t = np.linspace(-4.5, 5.5, 2001)
    F = tw_cdf_standardized(1)(t)
    f = tw_pdf_standardized(1)(t)
    psi = (t ** 2 - 1) * f
    norm2 = np.trapz(psi * psi, t)

    series = [
        ("simple", "data/simple/simple_deg4_V{V}_N5000000.npy",
         (100, 200, 500, 1000, 2000, 5000, 10000, 20000), "#2a78d6", "o",
         "4-regular simple (Python)"),
        ("k5", "data/k5_covers_complete_cover/"
         "k5_covers_complete_cover_V{k}x5_N5000000.npy",
         (100, 200, 500, 1000, 2000, 5000, 10000), "#0d8ba3", "h",
         "$K_5$ cover"),
    ]
    collapse_sizes = (200, 1000, 5000, 20000)
    collapse = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for name, tmpl, sizes, color, marker, label in series:
        chats = []
        for V in sizes:
            path = tmpl.format(V=V, k=V // 5)
            x = np.load(os.path.join(REPO_ROOT, path))
            z = np.sort((x - x.mean()) / x.std(ddof=1))
            del x
            ecdf = np.searchsorted(z, t, side="right") / z.size
            del z
            delta = ecdf - F
            c = np.trapz(delta * psi, t) / norm2
            chats.append(abs(c))
            if name == "simple" and V in collapse_sizes:
                collapse[V] = delta / c
        axes[0].loglog(sizes, chats, marker=marker, ms=4, lw=1.2,
                       color=color, label=label)
        if name == "simple":
            slope, intercept = np.polyfit(np.log(sizes), np.log(chats), 1)
            Vline = np.array([70.0, 30000.0])
            axes[0].loglog(Vline, np.exp(intercept) * Vline ** slope, "--",
                           color=INK, lw=0.9,
                           label=rf"fit $\propto V^{{{slope:.2f}}}$")
    axes[0].set_xlabel("number of vertices $V$")
    axes[0].set_ylabel(r"amplitude $|\hat{c}(V)|$")
    axes[0].set_title("Edgeworth amplitude of the deviation")
    axes[0].legend(fontsize=6.5, loc="lower left")

    shades = ["#c5d9f2", "#8db8e8", "#4f8fd9", "#1c5fae"]
    for shade, V in zip(shades, collapse_sizes):
        axes[1].plot(t, collapse[V], color=shade, lw=1.1,
                     label=f"$V={V}$")
    axes[1].plot(t, psi, "--", color=INK, lw=1.2,
                 label=r"profile $(t^2-1)\,\tilde{f}_1(t)$")
    axes[1].set_xlabel(r"standardized eigenvalue $t$")
    axes[1].set_ylabel(r"$(F_V(t) - \tilde{F}_1(t))\,/\,\hat{c}(V)$")
    axes[1].set_title("Rescaled deviations collapse")
    axes[1].legend(fontsize=6.5, loc="lower right")
    axes[1].set_xlim(-4.5, 5.5)
    fig.subplots_adjust(wspace=0.34)
    fig.suptitle("The deviation of the standardized ECDF from $TW_1$ is a"
                 " single skewness profile with power-law amplitude", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2h. K4-e: the defect band and the trimmed bulk. The base has cycle rank 2,
# so proper rank-2 defects (long thetas with branches) bind above rho at
# rate ~1/k; computed values 2.51185 and 2.51369 cap the leading band, with
# rank-3 shapes reaching ~2.530 at rate ~1/k^2. Trimming the band (|z|>6)
# recovers a TW1-consistent bulk.
# ---------------------------------------------------------------------------

K4E_DEFECTS = {
    r"$\theta(2,2,4)$": 2.5118521208,
    r"$\theta(1,2,5)$": 2.5136917846,
}


def fig_k4e():
    from dataset_utils import tw_cdf_standardized as _cdf
    sizes = (25, 50, 125, 250, 500, 1250, 2500)
    raw_D, trim_D, outliers = [], [], {}
    cdf1 = _cdf(1)
    for k in sizes:
        V = 4 * k
        x = np.load(os.path.join(
            REPO_ROOT, "data", "k4_minus_edge",
            f"k4_minus_edge_cover_V{k}x4_N1000000.npy"))
        m, s0 = x.mean(), x.std(ddof=1)
        z = np.sort((x - m) / s0)
        raw_D.append(sps.kstest(z, cdf1).statistic)
        outliers[V] = x[x > m + 8 * s0]
        y = x
        for _ in range(2):
            my, sy = y.mean(), y.std(ddof=1)
            y = y[np.abs(y - my) <= 6 * sy]
        my, sy = y.mean(), y.std(ddof=1)
        zt = np.sort((y - my) / sy)
        trim_D.append(sps.kstest(zt, cdf1).statistic)
        del x, y, z, zt
    V = [4 * k for k in sizes]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    axes[0].loglog(V, raw_D, marker="d", ms=4, lw=1.2, color="#5e548e",
                   label="all samples")
    axes[0].loglog(V, trim_D, marker="d", ms=4, lw=1.2, ls="--",
                   color="#2a78d6", label=r"defect band removed ($|z|\leq6$)")
    axes[0].set_xlabel("number of vertices $V$")
    axes[0].set_ylabel("KS statistic $D$ vs Tracy–Widom $\\beta=1$")
    axes[0].set_title("Bulk distance to $TW_1$")
    axes[0].legend(fontsize=6.5, loc="lower left")

    rng = np.random.default_rng(11)
    ax = axes[1]
    for VV, out in outliers.items():
        if out.size == 0:
            continue
        jitter = rng.uniform(0.9, 1.1, out.size)
        ax.semilogx(VV * jitter, out, ls="", marker="o", ms=2.2,
                    color="#5e548e", alpha=0.5, markeredgewidth=0)
    ax.axhline(RHO_K4_MINUS_E, ls="--", color=INK, lw=0.9)
    ax.annotate(r"$\rho(K_4-e)$", xy=(1.01, RHO_K4_MINUS_E),
                xycoords=("axes fraction", "data"), fontsize=6, color=INK,
                va="center")
    for name, value in K4E_DEFECTS.items():
        ax.axhline(value, ls=(0, (1, 3)), color=INK_2, lw=0.9)
    ax.annotate("rank $2$ defects", xy=(1.01, 2.5128),
                xycoords=("axes fraction", "data"), fontsize=6, color=INK_2,
                va="center")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel(r"outlier eigenvalues $\lambda > \mathrm{mean}+8\,\mathrm{std}$")
    ax.set_title("The defect band above $\\rho$")
    fig.subplots_adjust(wspace=0.42)
    fig.suptitle("Covers of $K_4-e$: a dense rank-2 defect band at rate"
                 " $k^{-1}$ over a $TW_1$ bulk", y=1.03)
    return fig


# ---------------------------------------------------------------------------
# 2e. MNS's decisive statistic at 50x their sample size: the observed mass
# to the left of the sample mean, vs the values predicted by each law
# ---------------------------------------------------------------------------

def fig_mass_left(rows):
    from dataset_utils import tw_mass_left_of_mean
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = np.array([r["V"] for r in data])
        mlm = np.array([r["mass_left_of_mean"] for r in data])
        N = np.array([r["N"] for r in data])
        ax.errorbar(V, mlm, yerr=np.sqrt(mlm * (1 - mlm) / N),
                    marker=marker, ms=4, lw=1.2, color=color, label=label,
                    capsize=1.5, elinewidth=0.7)
    ax.set_xscale("log")
    for beta, ls in ((1, "--"), (2, "-."), (4, ":")):
        theta = tw_mass_left_of_mean(beta)
        ax.axhline(theta, ls=ls, color=INK, lw=0.9,
                   label=rf"TW $\beta={beta}$: {theta:.6f}")
    ax.axhline(0.5, ls=":", color=MUTED, lw=0.9, label="normal: 0.5")
    for key, name in (("max12", "$\\max(TW_1,TW_2)$"),
                      ("max22", "$\\max(TW_2,TW_2')$")):
        theta = MAX_LAW[key][1]
        ax.axhline(theta, ls=(0, (1, 3)), color=INK_2, lw=0.9)
        ax.annotate(f"{name}: {theta:.5f}", xy=(1.01, theta),
                    xycoords=("axes fraction", "data"), fontsize=5.5,
                    color=INK_2, va="center")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel("observed mass left of the sample mean")
    ax.set_title("MNS's discriminating statistic: mass left of the mean\n"
                 "(binomial error bars; smaller than markers at $N \\geq$ 1M)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3)
    return fig


# ---------------------------------------------------------------------------
# 3. Ratio gap/std and empirical Ramanujan probability
# ---------------------------------------------------------------------------

def fig_ratio(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = [r["V"] for r in data]
        ratio = [r["gap_over_std"] for r in data]
        ax.semilogx(V, ratio, marker=marker, ms=4, lw=1.2, color=color,
                    label=label)
    ax.axhline(TW1_MEAN_OVER_STD, ls="--", color=INK, lw=0.9,
               label=rf"$|\mu_{{TW_1}}|/\sigma_{{TW_1}} = {TW1_MEAN_OVER_STD:.6f}$")
    ratio_refs = ((-TW_MEAN[2] / TW_STD[2], "$TW_2$"),
                  (-TW_MEAN[4] / TW_STD[4], "$TW_4$"),
                  (MAX_LAW["max12"][0], "$\\max(TW_1,TW_2)$"),
                  (MAX_LAW["max22"][0], "$\\max(TW_2,TW_2')$"))
    for value, name in ratio_refs:
        ax.axhline(value, ls=(0, (1, 3)), color=INK_2, lw=0.9)
        ax.annotate(f"{name}: {value:.4f}", xy=(1.01, value),
                    xycoords=("axes fraction", "data"), fontsize=5.5,
                    color=INK_2, va="center")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel(r"$(\rho - \mathrm{mean}(\lambda))\,/\,\mathrm{std}(\lambda)$")
    ax.set_title("Mean gap below the threshold $\\rho$ in units of the"
                 " standard deviation")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


def fig_prob_ramanujan(rows):
    # Painleve II value of F_1(0); the TracyWidom package's cdf(0)
    # returns 0.831913, off in the 5th decimal.
    F1_0 = 0.8319081
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = np.array([r["V"] for r in data])
        frac = np.array([r["frac_ramanujan"] for r in data])
        N = np.array([r["N"] for r in data])
        # binomial standard error; smaller than the markers for N >= 1M
        ax.errorbar(V, frac, yerr=np.sqrt(frac * (1 - frac) / N),
                    marker=marker, ms=4, lw=1.2, color=color, label=label,
                    capsize=1.5, elinewidth=0.7)
    ax.set_xscale("log")
    ax.axhline(F1_0, ls="--", color=INK, lw=0.9,
               label=rf"$F_1(0) = {F1_0:.6f}$")
    for value, name in ((F_BETA_AT_0[2], "$F_2(0)$"),
                        (F_BETA_AT_0[4], "$F_4(0)$"),
                        (MAX_LAW["max12"][2], "$F_1(0)F_2(0)$"),
                        (MAX_LAW["max22"][2], "$F_2(0)^2$")):
        ax.axhline(value, ls=(0, (1, 3)), color=INK_2, lw=0.9)
        ax.annotate(f"{name}: {value:.5f}", xy=(1.01, value),
                    xycoords=("axes fraction", "data"), fontsize=5.5,
                    color=INK_2, va="center")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel(r"empirical $P(\lambda \leq \rho)$")
    ax.set_title("Empirical probability of being (one-sided) Ramanujan")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


# ---------------------------------------------------------------------------
# 4. PDF / CDF / QQ figures for the exceptional datasets
# ---------------------------------------------------------------------------

def fig_pdf_vs_tw(z_sorted, dataset_label):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    bins = 400 if z_sorted.size > 1_000_000 else 160
    # range extends further right than left: the standardized laws are
    # right-skewed and the data reaches ~+7 sd but only ~-4 sd
    ax.hist(z_sorted, bins=bins, range=(-4.5, 5.5), density=True,
            color=BASELINE, alpha=0.55, label="data (standardized)",
            edgecolor="none", zorder=1)
    t = np.linspace(-4.5, 5.5, 800)
    for law in ("tw1", "tw2", "tw4"):
        label, color, ls = LAW_STYLE[law]
        beta = int(law[2])
        ax.plot(t, tw_pdf_standardized(beta)(t), ls=ls, color=color, lw=1.4,
                label=label, zorder=2)
    label, color, ls = LAW_STYLE["normal"]
    ax.plot(t, sps.norm.pdf(t), ls=ls, color=color, lw=1.1, label=label,
            zorder=2)
    ax.set_xlabel(r"standardized eigenvalue $(\lambda - \mathrm{mean})/\mathrm{std}$")
    ax.set_ylabel("probability density")
    ax.set_title("Density of $\\lambda$ vs Tracy–Widom laws\n"
                 f"{dataset_label}")
    ax.legend(fontsize=7)
    ax.set_xlim(-4.5, 5.5)
    return fig


def fig_cdf_vs_tw1(z_sorted, dataset_label, ks_D, ks_p):
    n = z_sorted.size
    # extend the grid to the largest sample so the CDF curve spans the
    # shared x-axis (the difference panel reaches the extreme sample)
    t = np.linspace(-5.0, float(z_sorted[-1]) + 0.3, 1601)
    ecdf = np.searchsorted(z_sorted, t, side="right") / n
    tw1_cdf = tw_cdf_standardized(1)(t)

    # difference evaluated at the sample points themselves (decimated for
    # plotting, keeping the extrema) so the curve attains +-D exactly
    F_at_z = tw_cdf_standardized(1)(z_sorted)
    diff_hi = np.arange(1, n + 1) / n - F_at_z
    diff_lo = np.arange(0, n) / n - F_at_z
    keep = np.unique(np.concatenate([
        np.linspace(0, n - 1, 4001).astype(int),
        [np.argmax(diff_hi), np.argmin(diff_lo)],
    ]))

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(5.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    ax0.plot(t, ecdf, color="#2a78d6", lw=1.5, label="empirical CDF")
    ax0.plot(t, tw1_cdf, ls="--", color=INK, lw=1.0,
             label="Tracy–Widom $\\beta=1$ CDF")
    ax0.set_ylabel("cumulative probability")
    ax0.set_title("Empirical CDF vs Tracy–Widom $\\beta=1$\n"
                  f"{dataset_label}")
    ax0.legend(fontsize=7, loc="upper left")
    ax0.text(0.97, 0.10,
             f"Kolmogorov–Smirnov: $D = {ks_D:.2e}$,  $p = {ks_p:.3g}$\n"
             f"$\\sqrt{{N}}\\,D = {np.sqrt(n) * ks_D:.2f}$",
             transform=ax0.transAxes, ha="right", fontsize=7, color=INK_2,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    ax1.plot(z_sorted[keep], diff_hi[keep], color="#2a78d6", lw=1.0)
    ax1.axhline(0, color=BASELINE, lw=0.7)
    for sign in (1, -1):
        ax1.axhline(sign * ks_D, ls=":", color=MUTED, lw=0.8)
    ax1.set_xlabel(r"standardized eigenvalue $(\lambda - \mathrm{mean})/\mathrm{std}$")
    ax1.set_ylabel("ECDF $-$ TW$_1$ CDF")
    ax1.set_title("Difference (dotted lines: $\\pm$ KS statistic)",
                  fontsize=8.5)
    return fig


def fig_qq_vs_tw1(z_sorted, dataset_label, beta=1):
    # upper limit 0.999: the TracyWidom package's cdfinv returns NaN in the
    # extreme right tail (p > ~0.9994)
    p = np.linspace(1e-4, 0.999, 1999)
    theory = tw_quantile_standardized(beta)(p)
    empirical = np.quantile(z_sorted, p)
    fig, ax = plt.subplots(figsize=(3.7, 3.7))
    lims = [min(theory.min(), empirical.min()) - 0.2,
            max(theory.max(), empirical.max()) + 0.2]
    ax.plot(lims, lims, ls="--", color=BASELINE, lw=0.9, label="$y = x$")
    ax.plot(theory, empirical, color="#2a78d6", lw=1.3,
            label="quantile pairs")
    ax.set_xlim(lims), ax.set_ylim(lims)
    ax.set_xlabel(f"Tracy–Widom $\\beta={beta}$ quantiles (standardized)")
    ax.set_ylabel("data quantiles (standardized)")
    ax.set_title(f"QQ plot vs Tracy–Widom $\\beta={beta}$\n"
                 f"{dataset_label}", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    return fig


# ---------------------------------------------------------------------------
# 5. KS statistics vs size
# ---------------------------------------------------------------------------

def fig_ks_simple(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    python = series_rows(rows, "simple", None)
    matlab = series_rows(rows, "matlab_simple", None)
    for law in ("tw1", "tw2", "tw4", "normal"):
        label, color, ls = LAW_STYLE[law]
        for data, mfc, tag in [(python, None, "Python"),
                               (matlab, "white", "MATLAB")]:
            V = [r["V"] for r in data]
            D = [r[f"ks_D_{law}"] for r in data]
            ax.loglog(V, D, marker="o", ms=4, lw=1.2, ls=ls, color=color,
                      markerfacecolor=mfc or color,
                      label=f"{label} ({tag})" if tag == "Python" else None)
    ax.loglog([], [], marker="o", ms=4, color=MUTED, lw=0,
              markerfacecolor="white", label="open markers: MATLAB data")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel("Kolmogorov–Smirnov statistic $D$")
    ax.set_title("KS distance of standardized $\\lambda$ to candidate laws\n"
                 "random simple 4-regular graphs")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


def fig_ks_symmetry_classes(rows):
    """The headline structural finding: each 'pure' cover family converges
    to the Tracy-Widom law of its own random-matrix symmetry class."""
    combos = [
        ("simple", None, "4-regular simple (Python) vs TW $\\beta=1$ (GOE)",
         "#2a78d6", "o", "tw1"),
        ("abelian_cover", 3, "Abelian cover $k=3$ vs TW $\\beta=2$ (GUE)",
         "#008300", "D", "tw2"),
        ("quaternion", 4, "Quaternion cover vs TW $\\beta=4$ (GSE)",
         "#e87ba4", "X", "tw4"),
        ("k5_cover", None, "$K_5$ cover vs TW $\\beta=1$ (GOE)",
         "#0d8ba3", "h", "tw1"),
        ("k5_minus_edge", None, "$K_5-e$ cover vs TW $\\beta=1$ (GOE)",
         "#8a5a2b", "*", "tw1"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for family, k, label, color, marker, law in combos:
        data = series_rows(rows, family, k)
        V = [r["V"] for r in data]
        ax.loglog(V, [r[f"ks_D_{law}"] for r in data], marker=marker,
                  ms=4, lw=1.2, color=color, label=label)
        if law != "tw1":
            ax.loglog(V, [r["ks_D_tw1"] for r in data], ls=":", lw=0.9,
                      color=color)
    ax.plot([], [], ls=":", lw=0.9, color=MUTED,
            label="dotted: same family vs TW $\\beta=1$")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel("Kolmogorov–Smirnov statistic $D$")
    ax.set_title("Each cover family converges to the Tracy–Widom law\n"
                 "of its own symmetry class")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


def fig_ks_k5(rows):
    """KS distances for the two fixed-base cover families: both should
    select beta=1, with the K5-e family standardized around its own
    (irregular) threshold."""
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    k5e = series_rows(rows, "k5_minus_edge", None)
    k5 = series_rows(rows, "k5_cover", None)
    for law in ("tw1", "tw2", "tw4", "normal"):
        label, color, ls = LAW_STYLE[law]
        for data, mfc, tag in [(k5e, None, "k5e"), (k5, "white", "k5")]:
            V = [r["V"] for r in data]
            D = [r[f"ks_D_{law}"] for r in data]
            ax.loglog(V, D, marker="o", ms=4, lw=1.2, ls=ls, color=color,
                      markerfacecolor=mfc or color,
                      label=f"{label} ($K_5-e$)" if tag == "k5e" else None)
    ax.loglog([], [], marker="o", ms=4, color=MUTED, lw=0,
              markerfacecolor="white", label="open markers: $K_5$ covers")
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel("Kolmogorov\u2013Smirnov statistic $D$")
    ax.set_title("KS distance of standardized $\\lambda_{new}$ to candidate"
                 " laws\ncovers of $K_5-e$ (filled) and $K_5$ (open)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


def fig_ks_tw1_all(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for family, k, label, color, marker in SERIES:
        data = series_rows(rows, family, k)
        if not data:
            continue
        V = [r["V"] for r in data]
        D = [r["ks_D_tw1"] for r in data]
        ax.loglog(V, D, marker=marker, ms=4, lw=1.2, color=color, label=label)
    ax.set_xlabel("number of vertices $V$")
    ax.set_ylabel("KS statistic $D$ vs Tracy–Widom $\\beta=1$")
    ax.set_title("Convergence to Tracy–Widom $\\beta=1$ across all families")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)
    return fig


# ---------------------------------------------------------------------------
# 6. Universality overlay across families
# ---------------------------------------------------------------------------

def fig_universality(rows):
    picks = [
        ("data/simple/simple_deg4_V20000_N5000000.npy",
         "4-regular simple (Python), $V=20{,}000$", "#2a78d6"),
        ("data/matlab/simple_deg4_V500000_N100000.npy",
         "4-regular simple (MATLAB), $V=500{,}000$", "#1baf7a"),
        ("data/abelian_cover/abelian_cover_V5000x5_N100000x10_bdeg4.npy",
         "Abelian cover $k=5$, $V=25{,}000$", "#e34948"),
        ("data/quaternion_rep/quaternion_deg4_V5000x4_N1000000.npy",
         "Quaternion cover, $V=20{,}000$", "#e87ba4"),
        ("data/irreg_covers/irreg_covers_k5_minus_edge_cover_V2000x5_N5000000.npy",
         "$K_5-e$ cover, $V=10{,}000$", "#8a5a2b"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    # TW1 reference drawn first (under the data); the huge-N Python curve
    # drawn last so it is not buried; bin count scaled to sample size so
    # the N=100k series is not dominated by bin noise
    t = np.linspace(-4.5, 5.5, 800)
    ax.plot(t, tw_pdf_standardized(1)(t), ls="--", color=INK, lw=1.0,
            label="Tracy–Widom $\\beta=1$", zorder=1)
    for zorder, (path, label, color) in enumerate(reversed(picks), start=2):
        x = np.load(os.path.join(REPO_ROOT, path))
        nbins = 300 if x.size > 200_000 else 150
        bins = np.linspace(-4.5, 5.5, nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        z = (x - x.mean()) / x.std(ddof=1)
        density, _ = np.histogram(z, bins=bins, density=True)
        ax.plot(centers, density, color=color, lw=1.0, label=label,
                zorder=zorder)
        del x, z
    handles, labels = ax.get_legend_handles_labels()
    # the density is centered and the legend is wide, so no in-panel spot
    # is collision-free at final column width: legend goes below the axis
    ax.legend(handles[::-1], labels[::-1], fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_xlabel(r"standardized eigenvalue $(\lambda - \mathrm{mean})/\mathrm{std}$")
    ax.set_ylabel("probability density")
    ax.set_title("Standardized densities collapse onto Tracy–Widom $\\beta=1$")
    ax.set_xlim(-4.5, 5.5)
    return fig


# ---------------------------------------------------------------------------

def main():
    setup_style()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rows = load_summary()

    save(fig_scaling(rows, SERIES[:3], "random 4-regular graphs"),
         "scaling_loglog_regular_graphs.png",
         "Log-log plots of (2sqrt3 - mean) and std vs V for the three "
         "4-regular-graph datasets (Python simple, MATLAB simple, MATLAB "
         "permutation model), with fitted slopes and the conjectured "
         "Tracy-Widom V^(-2/3) line.")
    save(fig_scaling(rows, SERIES[3:6],
                     "abelian covers ($V = $ base $\\times k$)",
                     stat_phrase="largest new eigenvalue",
                     amplitude_conjectured=False),
         "scaling_loglog_abelian_cover.png",
         "Log-log plots of (2sqrt3 - mean) and std of the largest NEW "
         "eigenvalue vs V for abelian (cyclic) covers of degree k = 3, 4, "
         "5 over random simple 4-regular base graphs; V is the total cover "
         "size base*k. The dashed line is a V^(-2/3) slope guide at the "
         "regular-graph amplitude (no amplitude is conjectured for "
         "covers); the data runs parallel to it at a family-dependent "
         "offset.")
    save(fig_scaling(rows, SERIES[6:7],
                     "quaternion covers ($V = 4\\times$ base)",
                     stat_phrase="largest new eigenvalue",
                     amplitude_conjectured=False),
         "scaling_loglog_quaternion_cover.png",
         "Log-log plots of (2sqrt3 - mean) and std of the largest new "
         "eigenvalue vs V for quaternion-representation covers; V is the "
         "matrix size 4*base. Dashed line: V^(-2/3) slope guide at the "
         "regular-graph amplitude.")
    save(fig_scaling(rows, SERIES[7:],
                     "covers of the fixed bases $K_5$, $K_5-e$, $K_4-e$",
                     stat_phrase="largest new eigenvalue",
                     amplitude_conjectured=False),
         "scaling_loglog_k5_covers.png",
         "Log-log plots of (rho - mean) and std of the largest new "
         "eigenvalue vs V for random covers of the fixed base graphs K5 "
         "(rho = 2sqrt3) and K5 minus an edge (rho = 3.2628764659...); "
         "V = 5x cover degree. Dashed line: V^(-2/3) slope guide at the "
         "regular-graph amplitude.")
    save(fig_scaling(rows, SERIES, "all families",
                     stat_phrase="largest non-trivial (new) eigenvalue",
                     shared_legend=True),
         "scaling_loglog_all_families.png",
         "Log-log scaling of mean gap (rho - mean, with rho the family's "
         "own Ramanujan threshold) and std vs V for all nine dataset "
         "series on one pair of axes, spanning V = 100 to 500,000.")
    save(fig_compensated(rows),
         "compensated_scaling_all_families.png",
         "Compensated plots: (rho - mean)*V^(2/3) and std*V^(2/3) vs V "
         "(log x). The dashed lines are the amplitudes |mu_TW1|*c(4) and "
         "sigma_TW1*c(4) conjectured for plain 4-regular graphs, which "
         "those series approach; each cover family plateaus at its own "
         "amplitude, i.e. the V^(-2/3) rate is universal but the constant "
         "is family-dependent.")
    save(fig_local_exponent(rows),
         "scaling_local_exponent_vs_size_all_families.png",
         "Local log-log slopes (consecutive-size exponents) of the mean "
         "gap and std vs V for all series, with the conjectured -2/3 "
         "line. The mean-gap effective exponent drifts toward -2/3 from "
         "below as V grows, so a single power-law fit over small V (as "
         "in MNS) overestimates the exponent and wrongly suggests the "
         "gap outpaces the std - the origin of their 52% conjecture.")
    save(fig_moments(rows),
         "moments_skewness_kurtosis_vs_size.png",
         "Sample skewness and excess kurtosis vs V with the Tracy-Widom "
         "beta = 1, 2, 4 reference values (computed numerically from the "
         "TW pdfs; affine-invariant, so independent of the mean/std "
         "normalization). The simple families converge to the beta=1 "
         "values, abelian k=3 to beta=2, quaternion to beta=4 - the "
         "moment-level confirmation of the symmetry-class picture. The "
         "MATLAB permutation model (excess kurtosis up to ~10^4) and the "
         "fixed-base K5/K5-e families (moments contaminated by thin "
         "far-outlier tails; see the standardized-tail figure) are "
         "omitted.")
    save(fig_k4e(),
         "k4_minus_edge_bulk_and_defect_band.png",
         "Covers of K4-e: left, the KS distance of the standardized sample "
         "to TW1 with and without the defect band (|z|>6 removed) - the "
         "raw distance grows at mid sizes because the base has cycle rank "
         "2, so proper rank-2 defects (long thetas with branches; computed "
         "values 2.51185 and 2.51369) bind above rho = sqrt(1+2 sqrt7) = "
         "2.5082868 at rate ~1/k, with rank-3 shapes reaching ~2.530 at "
         "~1/k^2; right, every sample above mean+8 std against V with the "
         "computed rank-2 defect values overlaid.")
    save(fig_outlier_atoms(),
         "outlier_atoms_k5_families.png",
         "Every sample more than 8 standard deviations above the mean, "
         "against V, for the K5-e and K5 cover families, with the computed "
         "localized defect eigenvalues overlaid: rho, lambda(K4) = "
         "3.3391340785 for K5-e (exactly 7/2 for K5), and the largest "
         "cycle-rank-4 values. The outliers lock onto the atoms; their "
         "frequencies decay like k^(1-r) where r is the cycle rank of the "
         "seeding subgraph (measured exponents ~2 in the K4 window, ~3 "
         "beyond it).")
    save(fig_edgeworth(),
         "edgeworth_amplitude_vs_size.png",
         "The approach to the Tracy-Widom limit: the deviation of the "
         "standardized empirical CDF from TW1 is proportional to the fixed "
         "profile (t^2-1)*f(t) (an Edgeworth-type skewness deficit); the "
         "left panel shows the fitted amplitude decaying like a power of V "
         "(exponent about -0.45) for the simple 4-regular and K5-cover "
         "families, the right panel the rescaled deviations collapsing "
         "onto the profile.")
    save(fig_tail_k5(),
         "tail_survival_standardized_k5_families_V10000.png",
         "Standardized right tail P(lambda > x) at V=10,000 for the simple "
         "4-regular, K5-cover, and K5-e-cover families with the Tracy-Widom "
         "beta=1 reference. The bulk of all three collapses onto TW1; the "
         "K5-e family carries a thin far-outlier tail (frequency ~1e-5, "
         "reaching rho+41 std at V=10,000, and growing in std units with "
         "V), K5 a much thinner one, and the simple family none. These "
         "tails dominate the third/fourth moments of the fixed-base "
         "families while leaving KS-level statistics at the sampling "
         "floor.")
    save(fig_tail_comparison(),
         "tail_survival_simple_vs_permutation_V500000.png",
         "Log-scale right tail P(lambda > x) at V=500,000: the simple "
         "model's tail dies out within ~1e-4 of the Ramanujan threshold "
         "2sqrt(3), while the permutation (multigraph) model has "
         "outliers reaching lambda ~ 3.54, i.e. localized eigenvalues "
         "from multi-edges/loops - the reason that model deviates from "
         "Tracy-Widom.")
    save(fig_mass_left(rows),
         "mass_left_of_mean_vs_size_all_families.png",
         "Replication of MNS's decisive experiment at ~50x their sample "
         "size: the observed fraction of samples below the sample mean, "
         "vs the values predicted by Tracy-Widom beta=1 (0.519652), "
         "beta=2 (0.515016), beta=4 (0.511069), the normal (0.5), and "
         "the two maximum laws (0.53895 and 0.52193, dotted). "
         "With N = 1M-5M samples the binomial standard error is 2-5e-4, "
         "so the four candidate laws are separated by tens of standard "
         "errors.")
    save(fig_ratio(rows),
         "ratio_mean_gap_over_std_vs_size_all_families.png",
         "The ratio (rho - mean)/std vs V for all families, with the "
         "Tracy-Widom GOE reference |mean|/std = 0.951538 (dashed) - the "
         "constant behind the 83% Ramanujan conjecture for regular "
         "graphs, which the 4-regular simple series approach. The cover "
         "families plateau at the |mean|/std ratio of their own limit "
         "laws, marked by the dotted reference lines (TW2 1.9640, TW4 "
         "3.2061, max(TW1,TW2) 0.8099, max(TW2,TW2') 1.6219), so each "
         "family has its own limiting Ramanujan probability. (MNS "
         "conjectured this ratio tends to 0.)")
    save(fig_prob_ramanujan(rows),
         "prob_ramanujan_vs_size_all_families.png",
         "Empirical P(lambda <= rho) vs V for all families (rho = 2sqrt3 "
         "except for K5-e covers, where rho = 3.2628764659...), with the "
         "conjectured limit F_1(0) = 0.8319081 (un-recentered Tracy-Widom "
         "GOE CDF at 0) for the beta=1 families; the other cover families "
         "approach the dotted product-of-F_beta(0) limits (F_2(0), "
         "F_4(0), F_1(0)F_2(0), F_2(0)^2).")
    save(fig_ks_simple(rows),
         "ks_statistic_vs_size_simple_regular.png",
         "KS distance of the standardized samples to standardized "
         "Tracy-Widom beta = 1, 2, 4 and the standard normal, vs V, for "
         "random simple 4-regular graphs (filled markers: Python 5M-sample "
         "data; open: MATLAB 100k-sample data). TW beta=1 is the only law "
         "the distance to which keeps shrinking.")
    save(fig_ks_symmetry_classes(rows),
         "ks_statistic_symmetry_classes.png",
         "The structural finding: KS distance of each 'pure' cover family "
         "to the Tracy-Widom law of its own random-matrix symmetry class "
         "- plain 4-regular graphs to beta=1 (GOE), abelian k=3 covers "
         "(two conjugate complex characters -> complex Hermitian blocks) "
         "to beta=2 (GUE), quaternion covers to beta=4 (GSE). Dotted "
         "lines show the same families' distance to beta=1 for contrast; "
         "the matching-class distances fall to the sampling noise floor. "
         "Mixed-character covers (k=4, k=5) match no single TW law.")
    save(fig_ks_k5(rows),
         "ks_statistic_vs_size_k5_covers.png",
         "KS distance of the standardized samples to standardized "
         "Tracy-Widom beta = 1, 2, 4 and the standard normal, vs V, for "
         "random covers of the fixed bases K5-e (filled markers) and K5 "
         "(open markers), 5M samples per size. Both families select "
         "beta=1, the K5-e family around its own algebraic threshold.")
    save(fig_ks_tw1_all(rows),
         "ks_statistic_tw1_vs_size_all_families.png",
         "KS distance to Tracy-Widom beta=1 vs V for all nine series. "
         "Note the sample-size noise floors differ: ~4e-4 at N=5M, ~9e-4 "
         "at N=1M, ~2.7e-3 at N=100k.")
    save(fig_universality(rows),
         "pdf_universality_across_families.png",
         "Standardized empirical densities of the largest datasets from "
         "five families (Python simple V=20,000, MATLAB simple V=500,000, "
         "abelian k=5 cover V=25,000, quaternion cover V=20,000, K5-e "
         "cover V=10,000) overlaid "
         "on the standardized Tracy-Widom beta=1 density: the "
         "universality picture. The MATLAB permutation-model family is "
         "excluded because of its extreme-outlier tail.")

    for relpath, tag, label in EXCEPTIONAL:
        x = np.load(os.path.join(REPO_ROOT, relpath))
        z = np.sort((x - x.mean()) / x.std(ddof=1))
        del x
        row = next(r for r in rows if r["file"] == relpath)
        save(fig_pdf_vs_tw(z, label),
             f"pdf_vs_tracywidom_{tag}.png",
             f"Standardized histogram of {relpath} against the "
             "standardized Tracy-Widom beta = 1, 2, 4 densities and the "
             "standard normal.")
        save(fig_cdf_vs_tw1(z, label, row["ks_D_tw1"], row["ks_p_tw1"]),
             f"cdf_vs_tracywidom1_{tag}.png",
             f"Empirical CDF of {relpath} vs the Tracy-Widom beta=1 CDF "
             "(both standardized), with the pointwise difference and the "
             "KS statistic.")
        save(fig_qq_vs_tw1(z, label),
             f"qq_vs_tracywidom1_{tag}.png",
             f"QQ plot of {relpath} against Tracy-Widom beta=1 "
             "(both standardized), probability range 1e-4 to 0.999.")
        del z

    for relpath, beta, tag, label in [
        ("data/abelian_cover/abelian_cover_V5000x3_N100000x10_bdeg4.npy", 2,
         "abelian3_V15000_N1000000",
         "abelian $k=3$ covers, $V=15{,}000$, $N=10^{6}$"),
        ("data/quaternion_rep/quaternion_deg4_V5000x4_N1000000.npy", 4,
         "quaternion_V20000_N1000000",
         "quaternion covers, $V=20{,}000$, $N=10^{6}$"),
    ]:
        x = np.load(os.path.join(REPO_ROOT, relpath))
        z = np.sort((x - x.mean()) / x.std(ddof=1))
        del x
        save(fig_qq_vs_tw1(z, label, beta=beta),
             f"qq_vs_tracywidom{beta}_{tag}.png",
             f"QQ plot of {relpath} against Tracy-Widom beta={beta} "
             "(both standardized), probability range 1e-4 to 0.999: "
             "quantile-level evidence for the symmetry-class selection.")
        del z

    lines = [
        "# Figures",
        "",
        "All figures are generated by `python3 analysis/build_figures.py`"
        " (run from the repo root) from `data_summary.csv` and the raw"
        " `.npy` files; `data_summary.csv` itself is generated by"
        " `python3 analysis/build_summary.py`. Each figure is saved twice:"
        " a vector `.pdf` (use this in the LaTeX paper — crisp at any"
        " scale, Computer-Modern-style fonts, drawn at final column"
        " width) and a `.png` preview for browsing. Descriptions below"
        " are keyed by the `.png` name.",
        "",
        "Everywhere below, samples are standardized to mean 0 and std 1,"
        " and the Tracy-Widom laws are likewise standardized from their"
        " original normalization (TW beta=1: mean -1.2065335746,"
        " std 1.2679830577).",
        "",
    ]
    for fname in sorted(FIGURES):
        lines.append(f"- **`{fname}`** — {FIGURES[fname]}")
    with open(os.path.join(IMAGES_DIR, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote images/README.md ({len(FIGURES)} figures)")


if __name__ == "__main__":
    main()
