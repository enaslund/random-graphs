"""High-accuracy Tracy-Widom distribution functions by Bornemann's method.

The distribution functions are Fredholm determinants evaluated by
Gauss-Legendre quadrature (Bornemann, Math. Comp. 79 (2010) 871-915):

    F_2(s) = det(I - K_Ai)            on L^2(s, oo),
    F_1(s) = det(I - A)               on L^2(s, oo),  A(x,y) = Ai((x+y)/2)/2,
    F_4(s) = (det(I - A) + det(I + A))/2   on L^2(sqrt(2) s, oo),

all in the original Tracy-Widom normalization (the F_4 argument scaling is
the one whose mean is -2.3068848932, matching Tracy-Widom 1996 and the
constants in dataset_utils.py). The absolute error is near machine
precision, so unlike the TracyWidom package (about 1e-5) the functions are
usable on a logarithmic scale deep into both tails.

Everything is tabulated once on a fine grid and cached to
analysis/tw_table.npz; the public functions interpolate the table.
"""
import os

import numpy as np
from scipy import special
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "tw_table.npz")

S_MIN, S_MAX, S_STEP = -14.0, 10.0, 0.002
QUAD_NODES = 80


def _nodes(s, m=QUAD_NODES):
    """Gauss-Legendre nodes and weights for [s, oo), via x = s + 10 tan(pi t/2)."""
    t, w = np.polynomial.legendre.leggauss(m)
    t = (t + 1) / 2
    w = w / 2
    x = s + 10 * np.tan(np.pi * t / 2)
    wx = w * 10 * (np.pi / 2) / np.cos(np.pi * t / 2) ** 2
    return x, wx


def _det_airy(s):
    x, w = _nodes(s)
    ai, aip, _, _ = special.airy(x)
    X, Y = np.meshgrid(x, x, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (np.outer(ai, aip) - np.outer(aip, ai)) / (X - Y)
    K[np.diag_indices_from(K)] = aip ** 2 - x * ai ** 2
    sw = np.sqrt(w)
    return np.linalg.det(np.eye(len(x)) - sw[:, None] * K * sw[None, :])


def _det_goe(s, sign=-1.0):
    x, w = _nodes(s)
    X, Y = np.meshgrid(x, x, indexing="ij")
    A = 0.5 * special.airy((X + Y) / 2)[0]
    sw = np.sqrt(w)
    return np.linalg.det(np.eye(len(x)) + sign * sw[:, None] * A * sw[None, :])


def _cdf_exact(beta, s):
    if beta == 1:
        return _det_goe(s, -1.0)
    if beta == 2:
        return _det_airy(s)
    if beta == 4:
        t = np.sqrt(2) * s
        return 0.5 * (_det_goe(t, -1.0) + _det_goe(t, 1.0))
    raise ValueError(beta)


def build_table():
    grid = np.arange(S_MIN, S_MAX + S_STEP / 2, S_STEP)
    table = {"s": grid}
    for beta in (1, 2, 4):
        F = np.array([_cdf_exact(beta, s) for s in grid])
        table[f"F{beta}"] = np.clip(F, 0.0, 1.0)
    np.savez_compressed(TABLE_PATH, **table)
    return table


_CACHE = {}


def _table():
    if "t" not in _CACHE:
        if os.path.exists(TABLE_PATH):
            data = np.load(TABLE_PATH)
            _CACHE["t"] = {k: data[k] for k in data.files}
        else:
            _CACHE["t"] = build_table()
        t = _CACHE["t"]
        for beta in (1, 2, 4):
            F = t[f"F{beta}"]
            spline = CubicSpline(t["s"], F)
            _CACHE[("cdf", beta)] = spline
            _CACHE[("pdf", beta)] = spline.derivative()
            # log-CDF spline for the left tail, where F underflows linearly
            logF = np.log(np.maximum(F, 1e-300))
            _CACHE[("logcdf", beta)] = CubicSpline(t["s"], logF)
            logS = np.log(np.maximum(1 - F, 1e-300))
            _CACHE[("logsf", beta)] = CubicSpline(t["s"], logS)
    return _CACHE["t"]


def cdf(beta, s):
    _table()
    s = np.asarray(s, dtype=float)
    out = _CACHE[("cdf", beta)](np.clip(s, S_MIN, S_MAX))
    out = np.where(s < S_MIN, 0.0, np.where(s > S_MAX, 1.0, out))
    return np.clip(out, 0.0, 1.0)


def pdf(beta, s):
    _table()
    s = np.asarray(s, dtype=float)
    out = _CACHE[("pdf", beta)](np.clip(s, S_MIN, S_MAX))
    return np.where((s < S_MIN) | (s > S_MAX), 0.0, np.maximum(out, 0.0))


def log_cdf(beta, s):
    """log F_beta(s), accurate in the left tail."""
    _table()
    return _CACHE[("logcdf", beta)](np.clip(np.asarray(s, float), S_MIN, S_MAX))


def log_sf(beta, s):
    """log(1 - F_beta(s)), accurate in the right tail up to about s = 5."""
    _table()
    return _CACHE[("logsf", beta)](np.clip(np.asarray(s, float), S_MIN, S_MAX))


def quantile(beta, p):
    """Inverse CDF by monotone interpolation of the table."""
    t = _table()
    F = t[f"F{beta}"]
    keep = np.concatenate([[True], np.diff(F) > 0])
    return np.interp(np.asarray(p, float), F[keep], t["s"][keep])


def moments(beta):
    """(mean, std, skewness, excess kurtosis) by quadrature of the table."""
    t = _table()
    s = t["s"]
    f = pdf(beta, s)
    m1 = np.trapezoid(s * f, s)
    c = s - m1
    m2 = np.trapezoid(c ** 2 * f, s)
    m3 = np.trapezoid(c ** 3 * f, s)
    m4 = np.trapezoid(c ** 4 * f, s)
    return m1, np.sqrt(m2), m3 / m2 ** 1.5, m4 / m2 ** 2 - 3


def standardized(beta):
    """(cdf, pdf, quantile) of TW_beta rescaled to mean 0 and variance 1."""
    mu, sd, _, _ = moments(beta)
    return (lambda t: cdf(beta, mu + sd * np.asarray(t)),
            lambda t: sd * pdf(beta, mu + sd * np.asarray(t)),
            lambda p: (quantile(beta, p) - mu) / sd)


if __name__ == "__main__":
    build_table()
    for beta in (1, 2, 4):
        mu, sd, sk, ku = moments(beta)
        print(f"beta={beta}: mean {mu:.10f} std {sd:.10f} skew {sk:.7f} "
              f"exkurt {ku:.7f} F(0) {float(cdf(beta, 0.0)):.10f} "
              f"F(mean) {float(cdf(beta, mu)):.7f}")
