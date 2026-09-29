"""Spectral quantities of the universal cover of a finite base graph.

For a finite simple base graph B, the resolvent of the adjacency operator
of its universal cover is determined by the branch values y_{u->v}(z), the
Herglotz solution of

    y_{u->v}(z) = 1 / (z - sum_{w ~ v, w != u} y_{v->w}(z))

over the directed edges of B. The diagonal Green's function at a lift of
v is G_v(z) = 1 / (z - sum_{w ~ v} y_{v->w}(z)), and the limiting spectral
measure of the new eigenvalues of random covers of B is the average over
the base vertices of the spectral measures at the lifts,

    mu_B = (1/|V(B)|) sum_v mu_v,   d mu_v / dx = -(1/pi) Im G_v(x + i0).

Near its right edge rho(B) the density behaves like (C_B/pi) sqrt(rho - x),
and the square-root edge rule predicts fluctuations of the largest new
eigenvalue of a degree k cover at scale (C_B |V(B)| k)^(-2/3). For a
d-regular base, C_B = d (d-1)^(1/4) / (d-2)^2 and the rule reproduces the
constant c(d) of Huang, McKenzie and Yau.
"""
import numpy as np

K5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
K5_MINUS_E = [e for e in K5 if e != (0, 1)]
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
K4_MINUS_E = [e for e in K4 if e != (0, 1)]


def _structure(edges):
    directed = list(edges) + [(v, u) for u, v in edges]
    index = {e: i for i, e in enumerate(directed)}
    following = [[index[(v, w)] for (a, w) in directed if a == v and w != u]
                 for (u, v) in directed]
    vertices = sorted({u for e in edges for u in e})
    out_edges = {v: [index[(v, w)] for (a, w) in directed if a == v]
                 for v in vertices}
    return directed, following, vertices, out_edges


def branch_values(edges, z, y0=None, tol=1e-14, max_iter=200):
    """Newton's method for the branch values at complex z."""
    directed, following, _, _ = _structure(edges)
    m = len(directed)
    y = np.full(m, 1 / z, dtype=complex) if y0 is None else np.array(y0, complex)
    for _ in range(max_iter):
        s = np.array([y[f].sum() for f in following])
        F = y * (z - s) - 1
        J = np.diag(z - s).astype(complex)
        for i, f in enumerate(following):
            J[i, f] -= y[i]
        dy = np.linalg.solve(J, F)
        y = y - dy
        if np.max(np.abs(dy)) < tol:
            break
    return y


def density(edges, x, eta_final=1e-12):
    """Averaged spectral density of the universal cover at real points x,
    continued from the upper half plane."""
    _, _, vertices, out_edges = _structure(edges)
    out = []
    for xx in np.atleast_1d(x):
        y = None
        for eta in (1.0, 0.3, 0.1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, eta_final):
            y = branch_values(edges, complex(xx, eta), y)
        z = complex(xx, eta_final)
        G = [1 / (z - y[out_edges[v]].sum()) for v in vertices]
        out.append(-np.mean([g.imag for g in G]) / np.pi)
    return np.array(out)


def edge_constant(edges, rho, deltas=(1e-4, 1e-5, 1e-6)):
    """C_B with density ~ (C_B/pi) sqrt(rho - x), from the smallest delta."""
    vals = [np.pi * density(edges, rho - d)[0] / np.sqrt(d) for d in deltas]
    return vals[-1]
