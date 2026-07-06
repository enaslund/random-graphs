# Covers of Random Graphs

Data, code, and paper for a numerical study of the fluctuations of the largest
new eigenvalue of random covers of graphs. The headline findings: the largest
new eigenvalue of a random cover follows the Tracy-Widom law of the random
matrix symmetry class (β = 1, 2, 4) selected by the Frobenius-Schur type of the
irreducible representations of the covering group, the limiting probability
that a random cover is Ramanujan is an explicit product of Tracy-Widom
functionals (F₁(0) = 0.8319081… for random 4-regular graphs, not the ≈52%
conjectured by Miller-Novikoff-Sabelli), and the same β = 1 law holds for
covers of a fixed irregular base graph around the spectral radius of its
universal cover. The companion paper is `main.tex`.

## Contents

| Path | Description |
|---|---|
| `main.tex`, `references.bib` | The paper. Figures are included from `images/`. |
| `images/` | All figures: vector `.pdf` (used by the paper) plus `.png` previews. Generated, see [Reproducing the analysis](#reproducing-the-analysis). Described in `images/README.md`. |
| `data/` | Raw eigenvalue samples: 1.35×10⁸ samples across nine series (~1.1 GB of `.npy` files, committed deliberately). Naming conventions in `data/README.md`. |
| `data_summary.csv`, `data_summary.md` | One row per data file: moments, gap to the Ramanujan threshold, empirical Ramanujan probability, and Kolmogorov-Smirnov distances to the candidate laws. Generated. |
| `analysis/` | Analysis pipeline: `build_summary.py` writes the summary tables, `build_figures.py` writes every figure, `dataset_utils.py` holds the filename parsers, per-family thresholds, and Tracy-Widom reference constants. |
| `random_graphs/` | The library that generated the data: cover constructors (`covers.py`), sparse eigensolver drivers (`eigenvalues.py`), permutation and matrix-representation samplers, and basic statistics (`stats.py`). |
| `scripts/`, `run.py` | Generation drivers, dispatched via `run.py`. Kept for provenance; nothing in the analysis requires running them. |

## The dataset

Every data file is a flat one-dimensional `float64` NumPy array. Each entry is
the largest positive non-trivial (for covers: *new*, that is, not inherited
from the base graph) adjacency eigenvalue of one independently sampled graph.
There is no metadata inside the files: all parameters are in the filename.
Load with `np.load(path)`, or `np.load(path, mmap_mode="r")` for the 40 MB
files. All families have base degree 4.

| Series | Sizes | Samples per size | Model |
|---|---|---|---|
| simple 4-regular | V = 100 … 20,000 (8) | 5,000,000 | two uniform permutations, conditioned simple, connected |
| simple 4-regular (legacy) | V = 50,000 … 500,000 (4) | 100,000 | same model, older MATLAB pipeline |
| permutation multigraph (legacy) | V = 50,000 … 500,000 (4) | 100,000 | loops and multiple edges kept (a loop adds 2 to the diagonal) |
| ℤₖ covers, k = 3, 4, 5 | base n = 100 … 5000 (6 each) | 1,000,000 | uniform ℤₖ voltages on a fresh random simple 4-regular base; 10 covers per base |
| quaternion covers | base n = 100 … 5000 (6) | 1,000,000 | uniform Q₈ voltages through the 4-dimensional real irreducible representation |
| K₅ covers | cover degree k = 20 … 2000 (7) | 5,000,000 | uniform Sₖ voltages on the fixed base K₅ (V = 5k) |
| K₅−e covers | cover degree k = 20 … 2000 (7) | 5,000,000 | uniform Sₖ voltages on the fixed base K₅ minus an edge (V = 5k) |

The abelian files are stored in base-graph-major order: `reshape(100000, 10)`
groups the 10 covers sharing a base graph.

The Ramanujan threshold (the spectral radius of the universal cover of the
base) is 2√3 = 3.4641016… for every 4-regular family, and for the K₅−e covers

ρ(K₅−e) = 3.26287646593635862827…,

the largest zero of x¹⁴ − 20x¹² + 122x¹⁰ − 152x⁸ − 1295x⁶ + 4540x⁴ − 5948x² − 2000
(see [McKay's MathOverflow answer](https://mathoverflow.net/a/440155);
independently re-derived and verified). Each dataset record in
`analysis/dataset_utils.py` carries its own `threshold` field, and
`data_summary.csv` reports gaps and Ramanujan fractions relative to the
family's own threshold.

## Provenance and precision

- The Python-generated series are computed to machine precision with ARPACK
  (`scipy.sparse.linalg.eigsh`) applied to the adjacency matrix shifted by a
  small multiple of the identity. Old (lifted base) eigenvalues are removed by
  exact comparison against the base spectrum with a tolerance of 10⁻¹⁰;
  disconnected samples in the simple family are discarded.
- The two legacy series were generated years earlier by a separate MATLAB
  pipeline whose source is not included. Their values sit on a 10⁻¹² grid,
  coarser than machine precision. Do not pool them with the Python series
  without accounting for this.
- A per-process seeding collision duplicated worker batches within the
  abelian series: 452,508 of the 1.8 million base-graph batches (25%) are
  redundant copies, concentrated at the smallest sizes (55-81% of batches at
  n = 100-200, below 0.1% at n >= 1000). Duplicated batches agree to better
  than 3×10⁻¹³; dropping them moves summary statistics by at most about two
  units in their last displayed digit.
- Quirks to be aware of when reading `scripts/`: the `trivial_eig` arguments
  in `complete_cover.py` and `irreg_cover.py` only steer the eigensolver's
  identity shift (base eigenvalues are filtered exactly, so the stale values
  there are harmless); the generators write to the current working directory
  and the committed files were moved and in some cases renamed afterward; and
  `loop_graphs.py`'s `simple` flag affects only the output filename (the
  worker always generates simple graphs).

## Reproducing the analysis

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib`, and the
[`TracyWidom`](https://pypi.org/project/TracyWidom/) package. Run from the
repository root, summary first:

```
python3 analysis/build_summary.py    # rewrites data_summary.csv / data_summary.md
python3 analysis/build_figures.py    # rewrites images/ (30 figures) and images/README.md
```

The Tracy-Widom reference constants in `analysis/dataset_utils.py` (means,
variances, shape moments, mass left of the mean, F_β(0)) were computed by
direct Painlevé II (Hastings-McLeod) integration and cross-checked against
Bornemann's tables. The `TracyWidom` package is used only for CDF and quantile
evaluation: its interpolated pdf is too crude for moment-level references
(for β = 4 it gives an excess kurtosis of 0.033 against the true 0.0492), and
its CDF is accurate only to about 10⁻⁵ (it returns F₁(0) = 0.831913 against
the true 0.8319081).

## Building the paper

```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Regenerating the data

Not required for anything above, since all analysis runs from the committed `.npy`
files. For provenance: the generators are dispatched through `run.py`
(multiprocessing, one worker per core), and the committed dataset represents
thousands of core-hours, generated primarily on Google Cloud, with the most
recent series computed in 2023.
