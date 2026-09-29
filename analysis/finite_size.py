"""Parameter-free comparison of every dataset with its conjectured limit law.

For a dataset of size parameter n (the base size for the abelian and
quaternion covers, whose blocks are n x n over C or H; the number of
vertices V otherwise), the conjectured law is

    lambda = rho + c * n^(-2/3) * X,

with rho the family's Ramanujan threshold, c = c(4) = 3^(-1/6) for every
4-regular family (times 2^(-1/6) for the quaternionic block, which is the
normalization of the GSE, see the paper), and X distributed as TW_1, TW_2,
TW_4, or the maximum of independent Tracy-Widom variables for the mixed
cyclic covers. We form s = (lambda - rho) n^(2/3) / c, with nothing fitted,
and regress the empirical quantiles of s on the quantiles of X over the
probability range [0.05, 0.95]:

    s_p = alpha + beta * X_p.

beta is the ratio of the true scale to the conjectured one, and alpha is a
location offset in units of c n^(-2/3), reported as the coefficient

    a = -alpha * c * n^(1/3),

so that the law sits at rho - a/n + c n^(-2/3) X. For the irregular bases
K5-e and K4-e no constant is conjectured, and c(4) is used as the reference
scale, so that beta * c(4) is the fitted constant c_B.

Run from the repo root:  python3 analysis/finite_size.py
It writes finite_size_fits.csv, which build_figures.py reads.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import REPO_ROOT, discover_datasets, size_scaling_prefactor  # noqa: E402
import tw_fredholm as T  # noqa: E402

CSV_PATH = os.path.join(REPO_ROOT, "finite_size_fits.csv")
C4 = size_scaling_prefactor()
QUAT_FACTOR = 2 ** (-1 / 6)
PROBS = np.linspace(0.05, 0.95, 91)

# (family, cover degree or None) -> (law, scale factor, size variable)
SPEC = {
    ("simple", None): ("tw1", 1.0, "V"),
    ("matlab_simple", None): ("tw1", 1.0, "V"),
    ("matlab_loop_cover", None): ("tw1", 1.0, "V"),
    ("abelian_cover", 3): ("tw2", 1.0, "base"),
    ("abelian_cover", 4): ("max12", 1.0, "base"),
    ("abelian_cover", 5): ("max22", 1.0, "base"),
    ("quaternion", 4): ("tw4", QUAT_FACTOR, "base"),
    ("k5_cover", None): ("tw1", 1.0, "V"),
    ("k5_minus_edge", None): ("tw1", 1.0, "V"),
    ("k4_minus_edge", None): ("tw1", 1.0, "V"),
}

_GRID = np.linspace(-14, 10, 240001)


def law_cdf(law, s):
    s = np.asarray(s, float)
    if law == "tw1":
        return T.cdf(1, s)
    if law == "tw2":
        return T.cdf(2, s)
    if law == "tw4":
        return T.cdf(4, s)
    if law == "max12":
        return T.cdf(1, s) * T.cdf(2, s)
    if law == "max22":
        return T.cdf(2, s) ** 2
    raise ValueError(law)


def law_pdf(law, s):
    s = np.asarray(s, float)
    if law in ("tw1", "tw2", "tw4"):
        return T.pdf(int(law[-1]), s)
    if law == "max12":
        return T.pdf(1, s) * T.cdf(2, s) + T.cdf(1, s) * T.pdf(2, s)
    if law == "max22":
        return 2 * T.pdf(2, s) * T.cdf(2, s)
    raise ValueError(law)


def law_quantile(law, p):
    F = law_cdf(law, _GRID)
    keep = np.concatenate([[True], np.diff(F) > 0])
    return np.interp(np.asarray(p, float), F[keep], _GRID[keep])


def key_of(record):
    fam = record["family"]
    k = record["cover_deg"] if fam in ("abelian_cover", "quaternion") else None
    return fam, k


def rescale(record, x):
    """The parameter-free variable s, and the (law, c, n) used."""
    law, factor, size = SPEC[key_of(record)]
    n = record["base_size"] if size == "base" else record["V"]
    c = C4 * factor
    return (x - record["threshold"]) * n ** (2 / 3) / c, law, c, n


def ks_distance(sorted_s, law):
    n = sorted_s.size
    F = law_cdf(law, sorted_s)
    return max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))


def qq_fit(s, law, batches=10):
    """(beta, alpha) and batch-means standard errors."""
    xq = law_quantile(law, PROBS)
    beta, alpha = np.polyfit(xq, np.quantile(s, PROBS), 1)
    parts = [np.polyfit(xq, np.quantile(b, PROBS), 1)
             for b in np.array_split(s, batches)]
    se = np.std(parts, axis=0, ddof=1) / np.sqrt(batches)
    return beta, alpha, se[0], se[1]


def fit_record(record):
    x = np.load(record["path"])
    s, law, c, n = rescale(record, x)
    del x
    # batch standard errors need the samples in generation order
    beta, alpha, se_beta, se_alpha = qq_fit(s, law)
    s.sort()
    shift = -alpha / beta
    return {
        "file": record["file"], "family": record["family"],
        "cover_deg": record["cover_deg"] or "", "n": n, "V": record["V"],
        "N": s.size, "law": law, "c": c,
        "scale_ratio": beta, "scale_ratio_se": se_beta,
        "c_fit": beta * c,
        "alpha": alpha, "alpha_se": se_alpha,
        "a": -alpha * c * n ** (1 / 3), "a_se": se_alpha * c * n ** (1 / 3),
        "frac_ramanujan": np.mean(s <= 0),
        "F_at_shift": float(law_cdf(law, shift)),
        "F_at_0": float(law_cdf(law, 0.0)),
        "median_s": float(np.median(s)),
        "ks_free": ks_distance(s, law),
        "ks_shift": ks_distance(s - alpha, law),
        "ks_affine": ks_distance((s - alpha) / beta, law),
    }


def main():
    rows = []
    for record in discover_datasets():
        row = fit_record(record)
        rows.append(row)
        print(f"{row['file']}: scale {row['scale_ratio']:.4f} "
              f"a {row['a']:+.3f} ks_free {row['ks_free']:.4f} "
              f"ks_shift {row['ks_shift']:.4f}", flush=True)
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
