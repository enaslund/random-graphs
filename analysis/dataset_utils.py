"""Shared utilities for analyzing the eigenvalue datasets in data/.

Every data file is a flat 1-D float64 .npy array; each entry is the largest
positive non-trivial (for covers: new) adjacency eigenvalue of one sampled
graph. The Ramanujan threshold is the spectral radius of the universal cover
of the base graph: 2*sqrt(3) for every 4-regular family (the 4-regular tree),
and rho(K5 - e) = 3.2628764659... for the K5-minus-an-edge covers. Each
dataset record carries its own "threshold" field.
"""
import os
import re

import numpy as np
from TracyWidom import TracyWidom

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

DEG = 4
RAMANUJAN_THRESHOLD = 2 * np.sqrt(DEG - 1)

# Spectral radius of the universal cover of K5 minus an edge: the largest
# zero of x^14 - 20x^12 + 122x^10 - 152x^8 - 1295x^6 + 4540x^4 - 5948x^2
# - 2000 (McKay, https://mathoverflow.net/a/440155), independently
# re-derived and verified. This is the Ramanujan threshold for the new
# eigenvalues of K5-e covers; the base spectrum is {1 +- sqrt(7), -1, -1, 0}.
RHO_K5_MINUS_E = 3.26287646593635862827

# Tracy-Widom moments in the original (un-recentered) normalization,
# matching random_graphs/stats.py and Bornemann (2010).
TW_MEAN = {1: -1.2065335745820, 2: -1.771086807411, 4: -2.306884893241}
TW_VAR = {1: 1.607781034581, 2: 0.8131947928329, 4: 0.5177237207726}
TW_STD = {beta: np.sqrt(var) for beta, var in TW_VAR.items()}

# |mean|/std of TW GOE: the conjectured limit of (2*sqrt(3) - mu_V)/sigma_V.
TW1_MEAN_OVER_STD = -TW_MEAN[1] / TW_STD[1]

_TW_CACHE = {}


def tw(beta):
    if beta not in _TW_CACHE:
        _TW_CACHE[beta] = TracyWidom(beta=beta)
    return _TW_CACHE[beta]


def tw_cdf_standardized(beta):
    """CDF of TW(beta) rescaled to mean 0, std 1."""
    dist, mu, sigma = tw(beta), TW_MEAN[beta], TW_STD[beta]
    return lambda t: dist.cdf(mu + sigma * np.asarray(t))


def tw_pdf_standardized(beta):
    """PDF of TW(beta) rescaled to mean 0, std 1."""
    dist, mu, sigma = tw(beta), TW_MEAN[beta], TW_STD[beta]
    return lambda t: sigma * dist.pdf(mu + sigma * np.asarray(t))


def tw_quantile_standardized(beta):
    """Quantile function of TW(beta) rescaled to mean 0, std 1."""
    dist, mu, sigma = tw(beta), TW_MEAN[beta], TW_STD[beta]
    return lambda p: (dist.cdfinv(np.asarray(p)) - mu) / sigma


# F_beta(mu_beta) from a direct Painleve II (Hastings-McLeod) integration
# (DOP853, rtol 1e-12), which reproduces the Bornemann means/variances above
# to all printed digits. MNS's published values (0.519652, 0.515016,
# 0.511072) agree except for beta=4, where theirs is off by 3e-6; the
# TracyWidom package's cdf is only ~1e-5 accurate here.
TW_MASS_LEFT_OF_MEAN = {1: 0.5196521, 2: 0.5150156, 4: 0.5110689}


def tw_mass_left_of_mean(beta):
    """F_beta(mu_beta): the TW(beta) mass to the left of its own mean,
    MNS's decisive discriminating statistic."""
    return TW_MASS_LEFT_OF_MEAN[beta]


# (skewness, excess kurtosis) of TW(beta) from the same Painleve II
# integration. The package's interpolated pdf is too crude for these: it
# returns (0.171, 0.033) for beta=4 against the true (0.1655, 0.0492).
TW_SHAPE_MOMENTS = {
    1: (0.2934645, 0.1652429),
    2: (0.2240842, 0.0934481),
    4: (0.1655095, 0.0491952),
}


def tw_shape_moments(beta):
    """(skewness, excess kurtosis) of TW(beta). Affine-invariant, so
    directly comparable to the same statistics of standardized samples."""
    return TW_SHAPE_MOMENTS[beta]


def size_scaling_prefactor(deg=DEG):
    """Conjectured d-dependent prefactor c(d) with

        sigma_V ~ TW_STD[1] * c(d) * V^(-2/3)
        2*sqrt(d-1) - mu_V ~ |TW_MEAN[1]| * c(d) * V^(-2/3)

    (see rough_result_summary.md; constants match Sodin arXiv:0903.4295)."""
    return ((deg - 2) ** 2 / (deg * (deg - 1))) ** (2.0 / 3.0) * np.sqrt(deg - 1)


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------

_PATTERNS = [
    # (subdir, regex, family, source)
    ("simple", r"simple_deg(?P<deg>\d+)_V(?P<V>\d+)_N(?P<N>\d+)\.npy",
     "simple", "python"),
    ("matlab", r"simple_deg(?P<deg>\d+)_V(?P<V>\d+)_N(?P<N>\d+)\.npy",
     "matlab_simple", "matlab"),
    ("matlab", r"loop_cover_deg(?P<deg>\d+)_V(?P<V>\d+)_N(?P<N>\d+)\.npy",
     "matlab_loop_cover", "matlab"),
    ("abelian_cover",
     r"abelian_cover_V(?P<base>\d+)x(?P<k>\d+)_N(?P<N1>\d+)x(?P<N2>\d+)_bdeg(?P<deg>\d+)\.npy",
     "abelian_cover", "python"),
    ("quaternion_rep", r"quaternion_deg(?P<deg>\d+)_V(?P<base>\d+)x4_N(?P<N>\d+)\.npy",
     "quaternion", "python"),
    # V{k}x{base}: cover degree k over the 5-vertex base (V = 5k vertices)
    ("k5_covers_complete_cover",
     r"k5_covers_complete_cover_V(?P<k>\d+)x(?P<base>\d+)_N(?P<N>\d+)\.npy",
     "k5_cover", "python"),
    ("irreg_covers",
     r"irreg_covers_k5_minus_edge_cover_V(?P<k>\d+)x(?P<base>\d+)_N(?P<N>\d+)\.npy",
     "k5_minus_edge", "python"),
]


def discover_datasets(data_dir=DATA_DIR):
    """Returns a list of dataset records sorted by (family, cover_deg, V).

    Each record: dict with path, file, family, source, base_degree,
    base_size, cover_deg, V (total number of vertices / matrix size),
    N (number of samples, from the filename).
    """
    records = []
    for subdir, pattern, family, source in _PATTERNS:
        directory = os.path.join(data_dir, subdir)
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            m = re.fullmatch(pattern, fname)
            if not m:
                continue
            g = m.groupdict()
            if family in ("simple", "matlab_simple", "matlab_loop_cover"):
                base_size, cover_deg = int(g["V"]), None
                V, N = int(g["V"]), int(g["N"])
            elif family == "abelian_cover":
                base_size, cover_deg = int(g["base"]), int(g["k"])
                V, N = base_size * cover_deg, int(g["N1"]) * int(g["N2"])
            elif family == "quaternion":
                base_size, cover_deg = int(g["base"]), 4
                V, N = 4 * base_size, int(g["N"])
            elif family in ("k5_cover", "k5_minus_edge"):
                base_size, cover_deg = int(g["base"]), int(g["k"])
                V, N = base_size * cover_deg, int(g["N"])
            # K5 is 4-regular; K5 - e is irregular (degrees 3, 4, 4, 4, 3)
            base_degree = None if family == "k5_minus_edge" else int(
                g.get("deg", DEG))
            threshold = (RHO_K5_MINUS_E if family == "k5_minus_edge"
                         else RAMANUJAN_THRESHOLD)
            records.append({
                "path": os.path.join(directory, fname),
                "file": os.path.join("data", subdir, fname),
                "family": family,
                "source": source,
                "base_degree": base_degree,
                "base_size": base_size,
                "cover_deg": cover_deg,
                "V": V,
                "N": N,
                "threshold": threshold,
            })
    records.sort(key=lambda r: (r["family"], r["cover_deg"] or 0, r["V"]))
    return records


def load_eigenvalues(record):
    return np.load(record["path"])
