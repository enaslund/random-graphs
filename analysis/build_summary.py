"""Builds data_summary.csv (and a readable data_summary.md) with one row per
.npy file in data/: sample statistics, the gap to the Ramanujan threshold,
the empirical Ramanujan probability, and Kolmogorov-Smirnov statistics of the
standardized sample against the standardized Tracy-Widom beta = 1, 2, 4
distributions and the standard normal.

Run from the repo root:  python3 analysis/build_summary.py
"""
import csv
import os
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import (  # noqa: E402
    REPO_ROOT, RHO_K5_MINUS_E, TW1_MEAN_OVER_STD,
    discover_datasets, load_eigenvalues, tw_cdf_standardized,
    tw_mass_left_of_mean,
)

# MNS's decisive statistic: expected mass left of the sample mean under
# each candidate law (MNS Table 1: 0.519652 / 0.515016 / 0.511072 / 0.5)
MLM_REF = {
    "tw1": tw_mass_left_of_mean(1),
    "tw2": tw_mass_left_of_mean(2),
    "tw4": tw_mass_left_of_mean(4),
    "normal": 0.5,
}

CSV_PATH = os.path.join(REPO_ROOT, "data_summary.csv")
MD_PATH = os.path.join(REPO_ROOT, "data_summary.md")

COLUMNS = [
    "file", "family", "source", "base_degree", "base_size", "cover_deg",
    "V", "N", "threshold",
    "mean", "variance", "std", "median", "min", "max",
    "skewness", "ex_kurtosis",
    "gap", "gap_over_std", "median_standardized",
    "frac_ramanujan",
    "mass_left_of_mean",
    "z_mlm_tw1", "z_mlm_tw2", "z_mlm_tw4", "z_mlm_normal",
    "ks_D_tw1", "ks_p_tw1",
    "ks_D_tw2", "ks_p_tw2",
    "ks_D_tw4", "ks_p_tw4",
    "ks_D_normal", "ks_p_normal",
]


def summarize(record):
    x = load_eigenvalues(record)
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    row = dict(record)
    del row["path"]
    row.update({
        "N": n,  # actual sample count (matches the filename for all files)
        "mean": mean,
        "variance": std ** 2,
        "std": std,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skewness": float(sps.skew(x)),
        "ex_kurtosis": float(sps.kurtosis(x)),
        "gap": record["threshold"] - mean,
        "gap_over_std": (record["threshold"] - mean) / std,
        "frac_ramanujan": float(np.mean(x <= record["threshold"])),
    })
    row["median_standardized"] = (row["median"] - mean) / std

    # MNS's mass-left-of-the-sample-mean z-test, at ~50x their sample size
    mlm = float(np.mean(x <= mean))
    row["mass_left_of_mean"] = mlm
    for name, theta in MLM_REF.items():
        row[f"z_mlm_{name}"] = (mlm - theta) / np.sqrt(
            theta * (1 - theta) / n)

    z = np.sort((x - mean) / std)
    for name, cdf in [
        ("tw1", tw_cdf_standardized(1)),
        ("tw2", tw_cdf_standardized(2)),
        ("tw4", tw_cdf_standardized(4)),
        ("normal", sps.norm.cdf),
    ]:
        ks = sps.kstest(z, cdf)
        row[f"ks_D_{name}"] = float(ks.statistic)
        row[f"ks_p_{name}"] = float(ks.pvalue)
    return row


def write_markdown(rows):
    md_cols = [
        ("file", "{}"), ("V", "{}"), ("N", "{}"),
        ("mean", "{:.6f}"), ("std", "{:.3e}"), ("median", "{:.6f}"),
        ("gap", "{:.3e}"), ("gap_over_std", "{:.4f}"),
        ("frac_ramanujan", "{:.4f}"),
        ("mass_left_of_mean", "{:.4f}"), ("z_mlm_tw1", "{:.2f}"),
        ("ks_D_tw1", "{:.2e}"), ("ks_D_tw2", "{:.2e}"),
        ("ks_D_tw4", "{:.2e}"), ("ks_D_normal", "{:.2e}"),
    ]
    lines = [
        "# Data summary",
        "",
        "One row per `.npy` file in `data/` (full precision in"
        " `data_summary.csv`). `gap` = ρ − mean, where ρ is the family's"
        " Ramanujan threshold: 2√3 for every 4-regular family and"
        f" ρ(K₅−e) = {RHO_K5_MINUS_E:.9f} for the K₅−e covers; for the"
        " plain 4-regular families `gap_over_std` should converge to"
        " |TW1 mean|/TW1 std = "
        f"{TW1_MEAN_OVER_STD:.9f} (multi-sheeted cover families, whose"
        " largest new eigenvalue is a max over several blocks, plateau at"
        " family-dependent constants); `frac_ramanujan` = P(λ ≤ ρ),"
        " conjectured limit F₁(0) = 0.8319081 for the β=1 (GOE) families. KS columns are Kolmogorov-Smirnov statistics of the"
        " standardized sample vs the standardized Tracy-Widom β = 1, 2, 4"
        " laws and the standard normal.",
        "",
        "| " + " | ".join(name for name, _ in md_cols) + " |",
        "|" + "|".join("---" for _ in md_cols) + "|",
    ]
    for row in rows:
        cells = [fmt.format(row[name]) for name, fmt in md_cols]
        lines.append("| " + " | ".join(cells) + " |")
    with open(MD_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = []
    for record in discover_datasets():
        t0 = time.time()
        row = summarize(record)
        rows.append(row)
        print(f"{record['file']}: N={row['N']:,} mean={row['mean']:.6f} "
              f"D_tw1={row['ks_D_tw1']:.2e} ({time.time() - t0:.1f}s)",
              flush=True)

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows)
    print(f"\nwrote {CSV_PATH} and {MD_PATH} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
